# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import io
import logging
import math
import os
import errno
import random
import sys
import types
import traceback
import wave
from contextlib import contextmanager
from collections import deque
from functools import partial
import json
import time
import subprocess
import threading
from multiprocessing import shared_memory
from typing import Any
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image, ImageOps
import torch.nn.functional as F
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import subprocess
from diffusers import FlowMatchEulerDiscreteScheduler
from .wan_2_2.distributed.fsdp import shard_model
from .wan_2_2.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .wan_2_2.distributed.util import get_world_size
from .causal_audio_encoder import AudioEncoder
from .causal_audio_encoder import AudioEncoder_Training
from .causal_model_s2v import CausalWanModel_S2V, sp_attn_forward_s2v
from .wan_2_2.modules.t5 import T5EncoderModel
from .wan_2_2.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .wan_2_2.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...utils.load_weight_utils import load_state_dict
from liveavatar.utils.router.utils import process_masks_to_routing_logits
from .inference_utils import STREAMING_VAE
from .live_stream_runtime import (
    LiveaudioRuntimeConfig,
    chunk_conditioning_target_frames,
    chunk_encode_batch_frames,
    effective_async_producer_mode,
    liveaudio_allow_long_clips,
    liveaudio_max_clip_frames,
    pending_clip_target,
    reply_boundary_prefill_wait_sec,
)
from .live_stream_filler_audio import build_filler_pcm_f32
from .live_stream_prompting import (
    build_stream_idle_prompt_text,
    merge_stream_clip_kinds,
    normalize_stream_clip_kind,
    prompt_switch_clip_kind_for_chunk,
    stream_clip_prefers_idle_prompt,
    stream_prompt_switch_enabled,
)
from .live_stream_trace import LiveaudioTrace
from .post_vae_enhancer import PostVAEEnhancer
from avalife.model.infer_cancel import InferenceCancelled, raise_if_infer_cancelled, request_infer_cancel
from avalife.core.speech_mask import speech_intervals_from_alignment
from avalife.worker.live_raw_shm import (
    LIVE_RAW_SHM_HEADER_BYTES,
    live_raw_shm_frame_region,
    live_raw_shm_total_bytes,
    live_raw_shm_write_header,
)


def _required_int_env(name: str) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        return int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from e


def _required_float_env(name: str) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        return float(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid float env {name}={raw!r}") from e


def _optional_int_env(name: str) -> int | None:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from e


def _liveavatar_audio_sample_m_env(default: int = 0) -> int:
    raw = str(os.getenv("LIVEAVATAR_AUDIO_SAMPLE_M", str(int(default))) or str(int(default))).strip()
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(0, min(4, int(value)))


def _liveaudio_stream_total_clips(*, max_repeat: int) -> int:
    """Resolve total liveaudio clip budget without confusing it with queue depth.

    LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS bounds the in-memory feeder queue only.
    Long RTMP sessions must not end when that queue-sized budget is exhausted.
    """
    max_repeat_i = max(1, int(max_repeat or 1))
    total = _optional_int_env("LIVE_AUDIO_STREAM_MAX_TOTAL_CLIPS")
    if total is None:
        # Backward-compatible legacy knob. Keep it only as an explicit override;
        # the default live stream lifetime is controlled by max_repeat.
        total = _optional_int_env("LIVE_AUDIO_STREAM_MAX_CLIPS")
    if total is None or int(total) <= 0:
        return int(max_repeat_i)
    return int(max(1, min(int(total), int(max_repeat_i))))


try:
    import fcntl  # Linux-only; used to enlarge FIFO pipe for live RAW streaming.
except Exception:
    fcntl = None

if STREAMING_VAE:
    from .wan_2_2.modules.vae_streaming import WanVAE as Wan2_1_VAE
else:
    from .wan_2_2.modules.vae2_1 import Wan2_1_VAE



def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _use_joint_sp_denoise(sp_size: int | None) -> bool:
    return int(sp_size or 1) > 1


def _resolve_tpp_stage_peers(
    *,
    local_gpu_id: int,
    num_gpus_dit: int,
    enable_vae_parallel: bool,
    joint_sp_denoise: bool,
) -> tuple[int | None, int | None]:
    if joint_sp_denoise:
        return None, None

    tgt_gpu_id = local_gpu_id + 1
    src_gpu_id = local_gpu_id - 1
    if local_gpu_id == num_gpus_dit - 1 + int(enable_vae_parallel):
        tgt_gpu = None
    else:
        tgt_gpu = tgt_gpu_id

    if local_gpu_id == 0:
        src_gpu = None
    else:
        src_gpu = src_gpu_id
    return src_gpu, tgt_gpu


def _resolve_tpp_step_range_for_rank(
    *,
    step_rank: int,
    num_steps: int,
    num_gpus_dit: int,
    joint_sp_denoise: bool,
) -> tuple[int, int]:
    if joint_sp_denoise:
        return 0, int(num_steps)

    base, rem = divmod(int(num_steps), int(num_gpus_dit))
    start = int(step_rank) * base + min(int(step_rank), rem)
    end = start + base + (1 if int(step_rank) < rem else 0)
    return start, end


class WanS2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        merged_noise_model_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=None,
        t5_cpu=False,
        init_on_cpu=True,
        drop_part_motion_frames=0.3,
        convert_model_dtype=False,
        is_training=False,
        single_gpu=False,
        offload_kv_cache=False
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.is_training = is_training
        self.num_train_timesteps = config.num_train_timesteps # 1000
        self.num_frames_per_block = config.num_frames_per_block
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.noise_model_checkpoint_dir = merged_noise_model_dir or checkpoint_dir
        self.drop_part_motion_frames = drop_part_motion_frames
        self.single_gpu = single_gpu
        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,dtype=self.param_dtype)

        if self.is_training:
            from liveavatar.models.wan.flow_match import FlowMatchScheduler_Omni
            self.scheduler = FlowMatchScheduler_Omni(shift=5, sigma_min=0.0, extra_one_step=True)
            self.scheduler.set_timesteps(1000, training=True)
        else:
            if config.sample_solver == 'euler':
                self.sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=3)
            elif config.sample_solver == 'unipc':#default
                self.sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            elif config.sample_solver == 'dpm++':
                self.sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            else:
                raise NotImplementedError("Unsupported solver.")

        logging.info(
            "Creating WanModel from %s (noise_model_dir=%s)",
            checkpoint_dir,
            self.noise_model_checkpoint_dir,
        )
        if not dit_fsdp:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                self.noise_model_checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                self.noise_model_checkpoint_dir, torch_dtype=self.param_dtype)
        
        self.noise_model.freqs.to(device=self.device)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            sp_size=sp_size,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        self.noise_model.num_frame_per_block = self.num_frames_per_block

        if not self.is_training:
            self.audio_encoder = AudioEncoder(
                model_id=os.path.join(checkpoint_dir,
                                    "wav2vec2-large-xlsr-53-english"))
        else:
            self.audio_encoder = AudioEncoder_Training(
                model_id=os.path.join(checkpoint_dir
                ,
                                    "wav2vec2-large-xlsr-53-english"))

        if use_sp:
            self.sp_size = sp_size if sp_size is not None else get_world_size()
        else:
            self.sp_size = 1
        self.joint_sp_denoise = _use_joint_sp_denoise(self.sp_size)

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = _liveavatar_audio_sample_m_env(0)
        self.tgt_gpu_id = 0
        self._static_reply_cond_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._neg_prompt_context_cache: dict[str, list[torch.Tensor]] = {}
        self._post_vae_enhancer: PostVAEEnhancer | None = None
        self._post_vae_face_restore: float = 0.0
        self._post_vae_background_restore: float = 0.0

    def _enhance_live_raw_frames(
        self,
        frames_tchw: torch.Tensor,
        *,
        clip_kind: str = "speech",
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> torch.Tensor | None:
        try:
            face_restore = float(max(0.0, min(1.0, float(self._post_vae_face_restore or 0.0))))
            background_restore = float(max(0.0, min(1.0, float(self._post_vae_background_restore or 0.0))))
            if face_restore <= 0.0 and background_restore <= 0.0:
                return None
            if self._post_vae_enhancer is None:
                self._post_vae_enhancer = PostVAEEnhancer(device=self.device)
            if not bool(self._post_vae_enhancer.enabled):
                return None
            self._post_vae_enhancer.set_restore_strengths(
                face_restore=float(face_restore),
                background_restore=float(background_restore),
            )
            return self._post_vae_enhancer.enhance_batch_tchw(
                frames_tchw,
                output_height=output_height,
                output_width=output_width,
            )
        except Exception as e:
            logging.exception("Post-VAE enhancer failed during frame batch: %s", e)
            return None

    def _prewarm_live_post_vae_face_restore(self, *, height: int, width: int) -> None:
        if float(self._post_vae_face_restore) <= 0.0:
            return
        try:
            if self._post_vae_enhancer is None:
                self._post_vae_enhancer = PostVAEEnhancer(device=self.device)
            if not bool(self._post_vae_enhancer.enabled):
                return
            self._post_vae_enhancer.set_restore_strengths(
                face_restore=float(self._post_vae_face_restore),
                background_restore=float(self._post_vae_background_restore),
            )
            self._post_vae_enhancer.prewarm_face_restore(
                height=int(height),
                width=int(width),
            )
        except Exception as e:
            logging.warning(
                "Post-VAE face restore prewarm failed: size=%dx%d err=%s",
                int(width),
                int(height),
                e,
            )
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None, load_only=False, load_lora_weight_only=False):
        if not load_only:
            self.lora_alpha = lora_alpha
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True if init_lora_weights == "kaiming" else init_lora_weights,
                target_modules=lora_target_modules.split(","),
            )
            model = get_peft_model(model, lora_config)
                
        if pretrained_lora_path is not None:
            ori_pretrained_lora_path = pretrained_lora_path
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            
            # get_peft_model adds "base_model.model." prefix to the keys
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith("base_model.model."):
                state_dict = {f"base_model.model.{k}": v for k, v in state_dict.items()}

            if load_lora_weight_only:
                state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {ori_pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
            
            # Merge weights and return base model
            print(f"Merging LoRA weights from {ori_pretrained_lora_path}...")
            model = model.merge_and_unload()
            print(f"LoRA merged successfully.")
            
        return model
    
    def set_all_model_to_dtype_device(self, dtype, device):
        models = [
            self.noise_model,
            self.text_encoder.model,
            self.vae.model,
            self.audio_encoder.model
        ]
        
        for model in models:
            for param in model.parameters():
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad = param.grad.to(device=device, dtype=dtype)
            
    
    def set_requires_grad(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.noise_model.requires_grad_(requires_grad)
        self.text_encoder.model.requires_grad_(requires_grad)
        self.vae.model.requires_grad_(requires_grad)
        self.audio_encoder.model.requires_grad_(requires_grad)
    
    def set_eval(self):
        self.noise_model.eval()
        self.text_encoder.model.eval()
        self.vae.model.eval()
        self.audio_encoder.model.eval()
    
    def set_train(self):
        self.noise_model.train()
        self.text_encoder.model.train()
        self.vae.model.train()
        self.audio_encoder.model.train()
    
    def set_device_dtype(self, device, dtype):
        self.noise_model.to(device, dtype)
        self.text_encoder.model.to(device, dtype)
        self.vae.model.to(device, dtype)
        self.audio_encoder.model.to(device, dtype)
    
    def _configure_model(self, model, use_sp, sp_size, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            sp_size (`int`):
                Sequence parallel size to use instead of world_size.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        if use_sp:
            # Store sp_size in model for use in forward pass
            model.sp_size = sp_size
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_s2v, block.self_attn)
                # Store sp_size in each block for access in forward pass
                block.sp_size = sp_size
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _process_timestep(self, timestep: torch.Tensor, type: str="causal_video") -> torch.Tensor:
        """
        copy from liveavatar/dmd.py:
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frames_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def get_size_less_than_area(self,
                                height,
                                width,
                                target_area=1024 * 704,
                                divisor=64):
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
                2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area /
                                  (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int(
                (target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int(
                (target_area / aspect_ratio)**0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    @staticmethod
    def _resize_cover_crop_pil(image: Image.Image, height: int, width: int) -> Image.Image:
        target_h = int(height)
        target_w = int(width)
        if target_h <= 0 or target_w <= 0:
            return image
        src_w, src_h = image.size
        if int(src_w) <= 0 or int(src_h) <= 0:
            return image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
        new_w = max(target_w, int(math.ceil(float(src_w) * float(scale))))
        new_h = max(target_h, int(math.ceil(float(src_h) * float(scale))))
        resized = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        left = max(0, (int(new_w) - int(target_w)) // 2)
        top = max(0, (int(new_h) - int(target_h)) // 2)
        return resized.crop((left, top, left + target_w, top + target_h))

    def prepare_default_cond_input(self,
                                   map_shape=[3, 12, 64, 64],
                                   motion_frames=5,
                                   lat_motion_frames=2,
                                   enable_mano=False,
                                   enable_kp=False,
                                   enable_pose=False):
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = torch.ones(
                    map_shape, dtype=self.param_dtype, device=self.device) * d
                cond_lat = torch.cat([
                    map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1),
                    map_value
                ],
                                     dim=2)
                cond_lat = torch.stack(
                    self.vae.encode(cond_lat.to(
                        self.param_dtype)))[:, :, lat_motion_frames:].to(
                            self.param_dtype)

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = torch.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path, infer_frames):
        assert self.is_training is False
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def encode_audio_from_array(self, audio_array, infer_frames):
        """
        Encode audio from in-memory mono float32 array (16kHz) for live stream mode.
        This avoids repeated librosa file decode path for tiny wav chunks.
        """
        assert self.is_training is False
        z = self.audio_encoder.extract_audio_feat_from_array(
            audio_array,
            return_all_layers=True,
            dtype=self.param_dtype,
        )
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m
        )
        audio_embed_bucket = audio_embed_bucket.to(self.device, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat
    
    def encode_audio_training(self, audio_tensor, infer_frames,fps,audio_sample_m = 0):
        assert self.is_training is True
        z = self.audio_encoder.extract_audio_feat_training(
            audio_tensor, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=fps, batch_frames=infer_frames, m=audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self,
                           video_path,
                           n_frames,
                           target_fps=16,
                           reverse=False):
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order. 
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))

        required_span = (n_frames - 1) * interval

        start_frame = max(0, total_frames - required_span -
                          1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def encode_prompt(self, input_prompt, n_prompt=None, offload_model=True):
        """
        编码文本提示词
        三种情况：只有正向，cfg但是提供空 neg，提供 neg
        
        Args:
            input_prompt (str): 正面文本提示词
            n_prompt (str): 负面文本提示词，默认为空字符串
            offload_model (bool): 是否在编码后将模型卸载到CPU，默认为True
            
        Returns:
            tuple: (context, context_null) - 编码后的正面和负面文本上下文
        """
        context_null = None
        if n_prompt == "": #case 3
            n_prompt = self.sample_neg_prompt
            
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if n_prompt is not None:
                cached_null = self._neg_prompt_context_cache.get(str(n_prompt))
                if cached_null is None:
                    context_null = self.text_encoder([n_prompt], self.device)
                    self._neg_prompt_context_cache[str(n_prompt)] = [t.detach() for t in context_null]
                else:
                    context_null = [t.clone() for t in cached_null]
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            if n_prompt is not None:
                cached_null = self._neg_prompt_context_cache.get(str(n_prompt))
                if cached_null is None:
                    context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                    context_null = [t.to(self.device) for t in context_null]
                    self._neg_prompt_context_cache[str(n_prompt)] = [t.detach() for t in context_null]
                else:
                    context_null = [t.clone() for t in cached_null]
                
        return context, context_null

    def _get_static_reply_condition(
        self,
        *,
        ref_image_path: str,
        size: tuple[int, int],
        infer_frames: int,
        drop_motion_noisy: bool,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        height, width = int(size[0]), int(size[1])
        cache_key = (
            os.path.abspath(str(ref_image_path)),
            int(height),
            int(width),
            int(infer_frames),
            int(self.motion_frames),
            str(self.param_dtype),
            str(self.device),
            1 if bool(drop_motion_noisy) else 0,
        )
        cached = self._static_reply_cond_cache.get(cache_key)
        if cached is not None:
            out = {
                "ref_latents": cached["ref_latents"].clone(),
                "motion_latents": cached["motion_latents"].clone(),
                "motion_frames_pixels": cached["motion_frames_pixels"].to(
                    dtype=self.vae.dtype,
                    device=self.vae.device,
                ).clone(),
                "cond_zero": cached["cond_zero"].clone(),
            }
            if bool(drop_motion_noisy) and ("zero_motion_latents" in cached):
                out["zero_motion_latents"] = cached["zero_motion_latents"].clone()
            return out, True

        tensor_trans = transforms.ToTensor()

        ref_image = ImageOps.exif_transpose(Image.open(ref_image_path)).convert("RGB")
        model_pic = self._resize_cover_crop_pil(ref_image, int(height), int(width))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae.dtype, device=self.vae.device)

        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))
        motion_frames_pixels = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
        motion_latents = torch.stack(self.vae.encode(motion_frames_pixels))

        cond = -torch.ones([1, 3, int(infer_frames), int(height), int(width)])
        cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
        cond_zero = torch.stack(
            self.vae.encode(cond.to(dtype=self.param_dtype, device=self.device))
        )[:, :, 1:].cpu()

        cache_entry: dict[str, torch.Tensor] = {
            "ref_latents": ref_latents.detach(),
            "motion_latents": motion_latents.detach(),
            "motion_frames_pixels": motion_frames_pixels.detach().cpu(),
            "cond_zero": cond_zero.detach(),
        }
        if bool(drop_motion_noisy):
            cache_entry["zero_motion_latents"] = torch.zeros_like(motion_latents).detach()
        self._static_reply_cond_cache[cache_key] = cache_entry

        out = {
            "ref_latents": cache_entry["ref_latents"].clone(),
            "motion_latents": cache_entry["motion_latents"].clone(),
            "motion_frames_pixels": cache_entry["motion_frames_pixels"].to(
                dtype=self.vae.dtype,
                device=self.vae.device,
            ).clone(),
            "cond_zero": cache_entry["cond_zero"].clone(),
        }
        if bool(drop_motion_noisy) and ("zero_motion_latents" in cache_entry):
            out["zero_motion_latents"] = cache_entry["zero_motion_latents"].clone()
        return out, False

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, size):
        HEIGHT, WIDTH = size
        if not pose_video is None:
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True)

            resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
            crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
            tensor_trans = transforms.ToTensor()

            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(
                1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat([
                cond_tensor,
                - torch.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])
            ],
                                    dim=2)

            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond],
                             dim=2)
            cond_lat = torch.stack(
                self.vae.encode(
                    cond.to(dtype=self.param_dtype,
                            device=self.device)))[:, :,
                                                  1:].cpu()  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        if size is not None:
            if isinstance(size, str):
                raw = str(size).strip().lower().replace("x", "*")
                parts = raw.split("*")
                if len(parts) != 2:
                    raise ValueError(f"invalid generate size: {size!r}")
                HEIGHT, WIDTH = int(parts[0]), int(parts[1])
            else:
                HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(
                    pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(
            HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    @staticmethod
    def _estimate_cond_cache_size(height: int, width: int) -> int:
        """
        Estimate the conditioning token budget for sink/prefill KV cache.

        The conditioning path contains:
        - one ref-image patch sequence
        - frame-pack motion buckets at 1x, 2x and 4x temporal scales

        The previous fixed size `3000` only fit around the old 720x400-ish
        internal resolution. Wider render profiles such as SmartBlog
        `render_video` landscape can legitimately need more tokens.
        """
        lat_h = max(1, int(height) // 8)
        lat_w = max(1, int(width) // 8)
        ref_tokens = max(1, (lat_h // 2) * (lat_w // 2))
        motion_post_tokens = max(1, (lat_h // 2) * (lat_w // 2))
        motion_2x_tokens = max(1, (lat_h // 4) * (lat_w // 4))
        motion_4x_tokens = max(1, 4 * (lat_h // 8) * (lat_w // 8))
        total = int(ref_tokens + motion_post_tokens + motion_2x_tokens + motion_4x_tokens)
        return int(max(256, math.ceil(float(total) / 128.0) * 128))

    @staticmethod
    def _resolve_stream_kv_cache_size(
        *,
        max_seq_len: int,
        frame_seq_length: int,
        num_frames_per_block: int,
    ) -> tuple[int, int, int]:
        """
        Optionally cap live streaming self-attention history without changing
        the external clip shape. `LIVE_STREAM_KV_CACHE_FRAMES=24` means keep a
        rolling 24-video-frame KV window while the request can still use
        INFER_FRAMES=48 for producer scheduling/buffering.
        """
        max_seq_len = int(max(1, int(max_seq_len)))
        frame_seq_length = int(max(1, int(frame_seq_length)))
        num_frames_per_block = int(max(1, int(num_frames_per_block)))
        try:
            cap_frames = int(os.getenv("LIVE_STREAM_KV_CACHE_FRAMES", "0") or 0)
        except Exception:
            cap_frames = 0
        if int(cap_frames) <= 0:
            return max_seq_len, 0, int(math.ceil(float(max_seq_len) / float(frame_seq_length)))

        cap_latent_frames = int(math.ceil(float(max(1, int(cap_frames))) / 4.0))
        cap_latent_frames = int(
            max(
                num_frames_per_block,
                math.ceil(float(cap_latent_frames) / float(num_frames_per_block)) * num_frames_per_block,
            )
        )
        cap_seq_len = int(cap_latent_frames * frame_seq_length)
        kv_cache_size = int(min(max_seq_len, max(num_frames_per_block * frame_seq_length, cap_seq_len)))
        effective_latent_frames = int(math.ceil(float(kv_cache_size) / float(frame_seq_length)))
        return kv_cache_size, int(cap_frames), int(effective_latent_frames)

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=13500, cond_cache_size=3000):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        gpu_id : "1","2","3","4"
        """
        kv_cache1 = []
        cond_cache_size = int(max(256, int(cond_cache_size or 3000)))

        for _ in range(self.noise_model.num_layers):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "cond_k": torch.zeros([batch_size, cond_cache_size, 40, 128], dtype=dtype, device=device),
                "cond_v": torch.zeros([batch_size, cond_cache_size, 40, 128], dtype=dtype, device=device),
                "cond_end": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
        self._kv_cache_step_ids = tuple()
        self._kv_cache_size = int(kv_cache_size)
        self._kv_cache_batch_size = int(batch_size)
        self._kv_cache_cond_cache_size = int(cond_cache_size)
        self._kv_cache_shape_key = tuple()

    def _initialize_kv_cache_by_steps(
        self,
        *,
        step_ids,
        batch_size,
        dtype,
        device,
        kv_cache_size=13500,
        cond_cache_size=3000,
        shape_key=(),
    ):
        """
        Initialize per-diffusion-step KV caches.

        The original TPP design assumes one diffusion step per GPU (so one KV cache
        per rank). When running with fewer GPUs than sampling steps, a rank will
        execute multiple steps sequentially. In that case we must keep *separate*
        KV caches per step; otherwise caches from a later step overwrite those of
        an earlier step, and subsequent blocks attend to mismatched history
        (causing progressive "scaly/mosaic" artifacts).

        We share the (step-invariant) conditioning caches (cond_k/cond_v/cond_end)
        across steps for memory efficiency.
        """
        step_ids = [int(s) for s in step_ids]
        if len(step_ids) == 0:
            self.kv_cache_by_step = {}
            self.kv_cache1 = None
            return
        cond_cache_size = int(max(256, int(cond_cache_size or 3000)))

        # Per-layer shared conditioning caches (keys/values for ref+motion tokens).
        shared_cond = []
        for _ in range(self.noise_model.num_layers):
            shared_cond.append(
                {
                    "cond_k": torch.zeros(
                        [batch_size, cond_cache_size, 40, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "cond_v": torch.zeros(
                        [batch_size, cond_cache_size, 40, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "cond_end": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        kv_cache_by_step = {}
        for step_id in step_ids:
            kv_cache_layers = []
            for layer_idx in range(self.noise_model.num_layers):
                kv_cache_layers.append(
                    {
                        "k": torch.zeros(
                            [batch_size, kv_cache_size, 40, 128],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, kv_cache_size, 40, 128],
                            dtype=dtype,
                            device=device,
                        ),
                        "cond_k": shared_cond[layer_idx]["cond_k"],
                        "cond_v": shared_cond[layer_idx]["cond_v"],
                        "cond_end": shared_cond[layer_idx]["cond_end"],
                    }
                )
            kv_cache_by_step[step_id] = kv_cache_layers

        self.kv_cache_by_step = kv_cache_by_step
        # Back-compat: keep an alias to the first step's cache (used for sink/prefill).
        self.kv_cache1 = kv_cache_by_step[min(step_ids)]
        self._kv_cache_step_ids = tuple(int(s) for s in sorted(step_ids))
        self._kv_cache_size = int(kv_cache_size)
        self._kv_cache_batch_size = int(batch_size)
        self._kv_cache_cond_cache_size = int(cond_cache_size)
        self._kv_cache_shape_key = tuple(shape_key) if isinstance(shape_key, (list, tuple)) else (shape_key,)
    
    def _move_kv_cache_to_working_gpu(self,moved_id, gpu_id=0):
        """
        Move the KV cache to the working GPU.
        move_id : "1","2","3","4"
        """
        if gpu_id == 0: #move to working device
            tgt_device =f"cuda:{gpu_id}"
        else: #offload
            tgt_device = "cpu" if self.single_gpu else f"cuda:{gpu_id}"
            
        kv_cache1 = self.kv_cache1[str(moved_id)]
        for layer in kv_cache1:
            layer["k"] = layer["k"].to(tgt_device)
            layer["v"] = layer["v"].to(tgt_device)
            layer["cond_k"] = layer["cond_k"].to(tgt_device)
            layer["cond_v"] = layer["cond_v"].to(tgt_device)
        self.kv_cache1[str(moved_id)] = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.noise_model.num_layers):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache
        self._crossattn_cache_batch_size = int(batch_size)

    def _reset_kv_cache_by_steps(self):
        """
        Reuse allocated per-step KV caches by zeroing in-place.
        Avoids per-request realloc jitter on long-running live streams.
        """
        if not isinstance(getattr(self, "kv_cache_by_step", None), dict):
            return
        for layers in self.kv_cache_by_step.values():
            for layer in layers:
                if torch.is_tensor(layer.get("k", None)):
                    layer["k"].zero_()
                if torch.is_tensor(layer.get("v", None)):
                    layer["v"].zero_()
                if torch.is_tensor(layer.get("cond_k", None)):
                    layer["cond_k"].zero_()
                if torch.is_tensor(layer.get("cond_v", None)):
                    layer["cond_v"].zero_()
                if torch.is_tensor(layer.get("cond_end", None)):
                    layer["cond_end"].zero_()

    def _reset_crossattn_cache(self):
        """
        Reset cross-attention cache state for a new request while keeping tensors allocated.
        """
        if not isinstance(getattr(self, "crossattn_cache", None), list):
            return
        for layer in self.crossattn_cache:
            k = layer.get("k", None)
            v = layer.get("v", None)
            if torch.is_tensor(k):
                layer["k"] = k[:, :0].contiguous()
            if torch.is_tensor(v):
                layer["v"] = v[:, :0].contiguous()
            layer["is_init"] = False

    def _release_stream_attention_caches(
        self,
        *,
        reason: str,
        clear_model_precompute: bool = False,
    ) -> None:
        """
        Fully release request-profile-dependent attention caches before allocating
        a new profile.

        Reusing caches is correct only while sampling steps, batch, KV length and
        latent shape are unchanged. When those knobs change, simply replacing the
        Python references can leave large freed blocks in PyTorch's CUDA allocator
        until the next pressure point. On memory-shared all-in-one pods this can
        briefly keep both old and new KV cache profiles resident and cause OOM.
        """

        def _mem_stats() -> tuple[float, float]:
            try:
                dev = self.device if getattr(self, "device", None) is not None else torch.cuda.current_device()
                allocated = float(torch.cuda.memory_allocated(dev)) / (1024.0 ** 3)
                reserved = float(torch.cuda.memory_reserved(dev)) / (1024.0 ** 3)
                return allocated, reserved
            except Exception:
                return -1.0, -1.0

        before_alloc, before_reserved = _mem_stats()

        self.kv_cache_by_step = {}
        self.kv_cache1 = None
        self.crossattn_cache = None
        self._kv_cache_step_ids = tuple()
        self._kv_cache_size = 0
        self._kv_cache_batch_size = 0
        self._kv_cache_cond_cache_size = 0
        self._kv_cache_shape_key = tuple()
        self._crossattn_cache_batch_size = 0

        if bool(clear_model_precompute):
            for cache_name in ("_stream_rope_precompute_cache", "_stream_static_precompute_cache"):
                cache_obj = getattr(self.noise_model, cache_name, None)
                if isinstance(cache_obj, dict):
                    cache_obj.clear()
            if isinstance(getattr(self.noise_model, "rope_cache", None), dict):
                self.noise_model.rope_cache.clear()
            if hasattr(self.noise_model, "_stream_rope_cond_grid_key"):
                self.noise_model._stream_rope_cond_grid_key = None
            if hasattr(self.noise_model, "block_mask"):
                self.noise_model.block_mask = None

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
        except Exception:
            pass

        after_alloc, after_reserved = _mem_stats()
        logging.info(
            "Released stream attention caches: reason=%s clear_model_precompute=%d "
            "cuda_alloc %.2f->%.2f GiB cuda_reserved %.2f->%.2f GiB",
            str(reason or "-"),
            1 if bool(clear_model_precompute) else 0,
            float(before_alloc),
            float(after_alloc),
            float(before_reserved),
            float(after_reserved),
        )
    
    def _initialize_comm_group(self, num_gpus_dit=4, enable_vae_parallel=False):
        local_gpu_id = torch.distributed.get_rank()
        self.src_gpu, self.tgt_gpu = _resolve_tpp_stage_peers(
            local_gpu_id=int(local_gpu_id),
            num_gpus_dit=int(num_gpus_dit),
            enable_vae_parallel=bool(enable_vae_parallel),
            joint_sp_denoise=bool(getattr(self, "joint_sp_denoise", False)),
        )

    def _safe_barrier(self):
        """
        A barrier that avoids NCCL collective barrier hangs observed under Gradio.

        For 2 ranks, use a simple CUDA send/recv handshake (P2P) instead of
        dist.barrier() (collective). For other sizes, fall back to dist.barrier().
        """
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        rank = dist.get_rank()
        if world_size == 2:
            token = torch.tensor([1], device=self.device, dtype=torch.int32)
            if rank == 0:
                dist.send(token, dst=1)
                dist.recv(token, src=1)
            else:
                dist.recv(token, src=0)
                dist.send(token, dst=0)
            return

        # Compatibility path for >2 ranks.
        try:
            dist.barrier(device_ids=[self.device.index] if self.device.type == "cuda" else None)
        except TypeError:
            dist.barrier()

    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        lipsync_audio_path=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,
        pose_video=None,
        generate_size=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        video_prompt=None,
        idle_prompt=None,
        seed=-1,
        offload_model=True,
        init_first_frame=False,
        use_dataset=False,
        dataset_sample_idx=0,
        drop_motion_noisy=False,
        num_gpus_dit=4,
        max_repeat=1000000,
        enable_vae_parallel=False,
        mask=None,
        input_video_for_sam2=None,
        enable_online_decode=False,
        live_hls_dir=None,
        live_raw_dir=None,
        post_vae_face_restore=0.0,
        post_vae_background_restore=0.0,
        job_id=None,
        stream_file_output_path=None,
        stream_file_output_width=0,
        stream_file_output_height=0,
        stream_file_output_fps=0.0,
        stream_file_trim_duration_sec=0.0,
        stream_file_interpolation=None,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # ------------------------------------Step 1: prepare conditional inputs--------------------------------------
        dataset_info = {}
        
        size = self.get_gen_size(
            size=generate_size,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        self._post_vae_face_restore = float(max(0.0, min(1.0, float(post_vae_face_restore or 0.0))))
        self._post_vae_background_restore = float(max(0.0, min(1.0, float(post_vae_background_restore or 0.0))))
        HEIGHT, WIDTH = size
        rank = dist.get_rank()
        if rank == 0 and str(audio_path or "").startswith("liveaudio://"):
            print(
                f"TPP live generate resolved: job={str(job_id or '-')} "
                f"requested_size={str(generate_size or '-')} size={int(HEIGHT)}x{int(WIDTH)} "
                f"ref={os.path.basename(str(ref_image_path or '')) or '-'}",
                flush=True,
            )
        # HEIGHT, WIDTH = map(int, generate_size.split('*'))
        # size = (HEIGHT, WIDTH)
        channel = 3
        profile_total_t0 = time.perf_counter()
        profile_audio_t0 = time.perf_counter()
        profile_audio_s = 0.0
        profile_static_cond_s = 0.0
        profile_prompt_s = 0.0
        profile_scheduler_comm_s = 0.0
        profile_loop_s = 0.0
        profile_active_clips = 0
        profile_last_num_blocks = 0
        profile_dit_blocks = 0
        profile_vae_blocks = 0
        profile_steps = 0
        profile_core_s = 0.0
        profile_recv_s = 0.0
        profile_denoise_s = 0.0
        profile_send_s = 0.0
        profile_vae_recv_s = 0.0
        profile_vae_decode_s = 0.0
        profile_rgb_pack_s = 0.0
        profile_cpu_pack_s = 0.0
        profile_raw_enqueue_s = 0.0
        profile_raw_write_s = 0.0
        profile_post_barrier_s = 0.0
        profile_concat_s = 0.0

        # extract audio emb
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        # audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        stream_audio_mode = False
        stream_audio_dir = ""
        stream_audio_clips: deque[torch.Tensor] = deque()
        stream_audio_clip_kinds: deque[str] = deque()
        stream_audio_clip_source_idxs: deque[int] = deque()
        stream_audio_clip_ref_paths: deque[str] = deque()
        stream_audio_clip_pcms: deque[bytes] = deque()
        stream_audio_clip_sample_rates: deque[int] = deque()
        stream_audio_clip_audible_samples: deque[int] = deque()
        stream_audio_clip_visible_start_frames: deque[int] = deque()
        stream_audio_clip_visible_frames: deque[int] = deque()
        stream_audio_done = False
        stream_audio_cancelled = False
        stream_audio_done_status = "ok"
        stream_audio_done_chunks_total = -1
        stream_audio_next_idx = 1
        stream_audio_seen_chunks = 0
        stream_audio_tail: torch.Tensor | None = None
        stream_audio_tail_kind: str | None = None
        stream_audio_tail_source_idx: int | None = None
        stream_audio_tail_ref_path = ""
        stream_audio_tail_pcm: bytes | None = None
        stream_audio_tail_sample_rate = int(max(1, int(_required_int_env("WORKER_AUDIO_SAMPLE_RATE"))))
        stream_audio_tail_audible_samples = 0
        stream_audio_tail_visible_frames = 0
        stream_audio_tail_flushed = False
        stream_audio_recent_context: torch.Tensor | None = None
        stream_audio_recent_context_pcm: bytes | None = None
        stream_audio_recent_context_sample_rate = int(stream_audio_tail_sample_rate)
        stream_audio_recent_context_ref_path = ""
        stream_audio_world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
        stream_runtime = LiveaudioRuntimeConfig.from_env(
            infer_frames=int(infer_frames),
            world_size=int(stream_audio_world_size),
        )
        stream_tail_frames = int(stream_runtime.tail_frames)
        stream_audio_poll_sec = float(stream_runtime.poll_sec)
        stream_audio_immediate_silence = bool(stream_runtime.immediate_silence)
        stream_audio_is_always_on = False
        stream_audio_reply_start_min_clips = int(stream_runtime.reply_start_min_clips)
        stream_audio_tail_fill_mode = str(os.getenv("LIVE_AUDIO_STREAM_TAIL_FILL_MODE", "zero") or "zero").strip().lower()
        if stream_audio_tail_fill_mode in ("noise", "white_noise", "white-noise"):
            stream_audio_tail_fill_mode = "noise"
        elif stream_audio_tail_fill_mode in ("smooth_noise", "smooth-noise"):
            stream_audio_tail_fill_mode = "smooth_noise"
        elif stream_audio_tail_fill_mode in ("hold", "repeat", "last"):
            stream_audio_tail_fill_mode = "hold"
        else:
            stream_audio_tail_fill_mode = "zero"
        stream_audio_tail_preroll = str(
            os.getenv("LIVE_AUDIO_STREAM_TAIL_PREROLL", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            stream_audio_tail_preroll_frames = int(
                os.getenv("LIVE_AUDIO_STREAM_TAIL_PREROLL_FRAMES", "0") or "0"
            )
        except Exception:
            stream_audio_tail_preroll_frames = 0
        stream_audio_tail_preroll_frames = max(0, int(stream_audio_tail_preroll_frames))
        stream_audio_fill_noise_std = float(_required_float_env("LIVE_AUDIO_STREAM_FILL_NOISE_STD"))
        stream_audio_fill_noise_std = max(0.0, min(0.05, float(stream_audio_fill_noise_std)))
        stream_audio_fill_noise_seed = int(_required_int_env("LIVE_AUDIO_STREAM_FILL_NOISE_SEED"))
        stream_audio_feature_speech_blend = str(
            os.getenv(
                "LIVE_AUDIO_STREAM_FEATURE_SPEECH_BLEND",
                os.getenv("WORKER_LIVEAUDIO_FEATURE_SPEECH_BLEND", "0"),
            )
            or "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        stream_audio_feature_speech_floor = max(
            0.0,
            min(
                0.8,
                float(os.getenv("LIVE_AUDIO_STREAM_FEATURE_SPEECH_FLOOR", "0.08") or 0.08),
            ),
        )
        stream_audio_feature_speech_fade_frames = max(
            0,
            min(12, int(os.getenv("LIVE_AUDIO_STREAM_FEATURE_SPEECH_FADE_FRAMES", "3") or 3)),
        )
        stream_audio_feature_neutral_mode = str(
            os.getenv("LIVE_AUDIO_STREAM_FEATURE_NEUTRAL_MODE", "zero") or "zero"
        ).strip().lower()
        if stream_audio_feature_neutral_mode in ("noise", "white_noise", "white-noise"):
            stream_audio_feature_neutral_mode = "noise"
        elif stream_audio_feature_neutral_mode in ("smooth_noise", "smooth-noise"):
            stream_audio_feature_neutral_mode = "smooth_noise"
        else:
            stream_audio_feature_neutral_mode = "zero"
        stream_audio_feature_neutral_cache: dict[tuple[int, str], torch.Tensor] = {}
        stream_audio_feature_blend_events = 0
        stream_trace_t0 = time.perf_counter()
        stream_trace_rank = int(dist.get_rank()) if dist.is_initialized() else 0
        stream_trace_tag = ""
        if live_raw_dir:
            try:
                stream_trace_tag = os.path.basename(os.path.abspath(str(live_raw_dir)))
            except Exception:
                stream_trace_tag = str(live_raw_dir or "")
        elif audio_path:
            try:
                stream_trace_tag = os.path.basename(os.path.abspath(str(audio_path)))
            except Exception:
                stream_trace_tag = str(audio_path or "")
        stream_live_trace = LiveaudioTrace(
            rank=stream_trace_rank,
            trace_t0=stream_trace_t0,
            trace_tag=stream_trace_tag,
            enabled=bool(stream_runtime.timing_log),
        )
        stream_first_clip_pop_dt: float | None = None
        stream_first_clip_pop_logged = False
        stream_timing_log = bool(stream_runtime.timing_log)
        # Exact per-phase GPU timings require synchronize() and slow down first-frame latency.
        # Keep that overhead out of the normal reply path; only enable it for explicit deep timing.
        stream_phase_sync_debug = bool(stream_runtime.phase_sync_debug)
        stream_timing_slow_sec = float(stream_runtime.timing_slow_sec)
        stream_block_log_slow_sec = max(1.0, float(stream_timing_slow_sec))
        if not bool(stream_timing_log):
            stream_block_log_slow_sec = max(2.5, float(stream_block_log_slow_sec))
        stream_step_trace = bool(stream_runtime.step_trace)
        stream_audio_max_pending_clips = int(stream_runtime.max_pending_clips)
        # Refill inside each denoise block can introduce periodic stalls when many
        # TTS chunks are already queued. Keep it disabled by default for smoother
        # frame cadence; clip-boundary refill remains active.
        stream_audio_refill_during_denoise = bool(stream_runtime.refill_during_denoise)
        stream_audio_refill_block_interval = int(stream_runtime.refill_block_interval)
        # Refill is synchronous in the inference thread. If one refill pass loads
        # many chunks, it can stall frame generation and produce visible micro-freezes.
        # Keep per-call refill small and frequent.
        stream_audio_refill_max_chunks_per_call = int(stream_runtime.refill_max_chunks_per_call)
        # Producer/consumer mode: audio refill+encode runs in a background thread,
        # while denoise loop only consumes ready embedding clips.
        stream_audio_async_producer_requested = bool(stream_runtime.async_producer)
        stream_audio_async_producer = bool(stream_audio_async_producer_requested)
        stream_audio_async_start_after_first_clip = bool(stream_runtime.async_start_after_first_clip)
        stream_audio_distributed_clip_broadcast = bool(stream_runtime.distributed_clip_broadcast)
        stream_audio_encode_rank = int(stream_runtime.encode_rank)
        stream_audio_is_encode_rank = (not stream_audio_distributed_clip_broadcast) or (
            int(dist.get_rank()) == int(stream_audio_encode_rank)
        )
        stream_audio_silence_clip: torch.Tensor | None = None
        stream_audio_tail_fill_clip: torch.Tensor | None = None
        stream_audio_producer_thread: threading.Thread | None = None
        stream_audio_producer_stop: threading.Event | None = None
        stream_audio_producer_error: str | None = None
        stream_audio_queue_cv = None
        stream_audio_refill_lock = threading.Lock()
        stream_audio_prompt_switch = bool(stream_prompt_switch_enabled())
        stream_audio_last_prompt_mode: str | None = None
        stream_audio_skip_before_chunk_idx = 0
        stream_audio_startup_error: str | None = None
        audio_encode_path = str(lipsync_audio_path or "").strip() or audio_path

        if isinstance(audio_path, str) and str(audio_path).startswith("liveaudio://"):
            stream_audio_mode = True
            stream_audio_dir = os.path.abspath(str(audio_path)[len("liveaudio://") :].strip())
            stream_audio_is_always_on = str(stream_audio_dir).endswith("_alwayson")
        stream_audio_dir_name = os.path.basename(stream_audio_dir) if stream_audio_dir else ""
        stream_audio_is_warmup = str(stream_audio_dir_name).startswith("warmup_liveaudio_")
        stream_audio_async_producer = effective_async_producer_mode(
            requested_async=bool(stream_audio_async_producer_requested),
            is_always_on=bool(stream_audio_is_always_on),
        )
        # In multi-rank mode, run liveaudio encode/refill only on one rank and
        # broadcast ready clip tensors to others to avoid duplicate GPU audio work.
        if stream_audio_distributed_clip_broadcast and (not stream_audio_is_encode_rank):
            stream_audio_async_producer = False
        # Regular reply startup only needs the first encoded clip. Starting the
        # async producer before that clip is consumed lets it encode several
        # future chunks on the same GPU and delays first-frame latency.
        stream_audio_defer_async_start = bool(
            stream_audio_async_producer
            and stream_audio_async_start_after_first_clip
            and stream_audio_is_encode_rank
            and (not stream_audio_is_always_on)
        )
        stream_audio_queue_cv = threading.Condition() if stream_audio_async_producer else None

        def _stream_done_marker_exists() -> bool:
            if not stream_audio_dir:
                return False
            return (
                os.path.exists(os.path.join(stream_audio_dir, "done.json"))
                or os.path.exists(os.path.join(stream_audio_dir, ".done"))
            )

        def _stream_done_marker_meta() -> dict:
            if not stream_audio_dir:
                return {}
            path = os.path.join(stream_audio_dir, "done.json")
            if not os.path.exists(path):
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        def _stream_write_progress_marker() -> None:
            if not stream_audio_dir:
                return
            payload = {
                "next_idx": int(max(1, int(stream_audio_next_idx))),
                "seen_chunks": int(max(0, int(stream_audio_seen_chunks))),
                "queue_depth": int(max(0, int(len(stream_audio_clips)))),
                "done": bool(stream_audio_done),
                "ts_ms": int(time.time() * 1000.0),
            }
            try:
                tmp = os.path.join(stream_audio_dir, "progress.json.tmp")
                final = os.path.join(stream_audio_dir, "progress.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                os.replace(tmp, final)
            except Exception:
                pass

        def _stream_chunk_path(idx: int) -> str:
            return os.path.join(stream_audio_dir, f"{int(idx):06d}.wav")

        def _stream_chunk_meta_path(idx: int) -> str:
            return os.path.join(stream_audio_dir, f"{int(idx):06d}.meta.json")

        def _stream_chunk_meta(idx: int) -> dict:
            meta_path = _stream_chunk_meta_path(idx)
            if meta_path:
                for _ in range(5):
                    if os.path.exists(meta_path):
                        break
                    time.sleep(0.005)
            try:
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    return meta if isinstance(meta, dict) else {}
            except Exception:
                pass
            return {}

        def _stream_soft_break_path() -> str:
            return os.path.join(stream_audio_dir, "soft_break.json")

        def _stream_soft_break_chunk_idx() -> int:
            if not stream_audio_dir:
                return 0
            path = _stream_soft_break_path()
            if not os.path.exists(path):
                return 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return int(max(0, int(data.get("skip_before_chunk_idx") or 0)))
            except Exception:
                pass
            return 0

        def _stream_chunk_kind(idx: int) -> str:
            try:
                meta = _stream_chunk_meta(idx)
                if isinstance(meta, dict) and meta:
                    try:
                        source_samples = int(meta.get("source_samples") or 0)
                    except Exception:
                        source_samples = 0
                    turn_done = bool(meta.get("turn_done", False))
                    full_chunk_samples = int(_required_int_env("WORKER_LIVEAUDIO_MICRO_CHUNK_SCHEDULE_SAMPLES"))
                    return prompt_switch_clip_kind_for_chunk(
                        kind=str(meta.get("kind") or "speech"),
                        source_samples=int(source_samples),
                        full_chunk_samples=int(max(1, int(full_chunk_samples))),
                        turn_done=bool(turn_done),
                    )
            except Exception:
                pass
            return "speech"

        def _stream_chunk_avatar_ref_path(idx: int) -> str:
            try:
                meta = _stream_chunk_meta(idx)
                if isinstance(meta, dict) and meta:
                    path = str(
                        meta.get("avatar_ref_path")
                        or meta.get("avatarRefPath")
                        or meta.get("ref_image_path")
                        or meta.get("refImagePath")
                        or ""
                    ).strip()
                    if path and os.path.exists(path):
                        return os.path.abspath(path)
            except Exception:
                pass
            return ""

        def _stream_chunk_visual_prompts(idx: int) -> tuple[str, str]:
            try:
                meta = _stream_chunk_meta(idx)
                if isinstance(meta, dict) and meta:
                    prompt = ""
                    for key in ("visual_prompt", "visualPrompt", "video_prompt", "videoPrompt", "prompt"):
                        value = str(meta.get(key) or "").strip()
                        if value:
                            prompt = value
                            break
                    negative = ""
                    for key in (
                        "negative_prompt",
                        "negativePrompt",
                        "video_negative_prompt",
                        "videoNegativePrompt",
                    ):
                        value = str(meta.get(key) or "").strip()
                        if value:
                            negative = value
                            break
                    return prompt, negative
            except Exception:
                pass
            return "", ""

        def _build_stream_fill_clip(*, mode: str, seed_offset: int = 0) -> torch.Tensor | None:
            """
            Build one synthetic conditioning clip with exactly `infer_frames` length.
            Used either for always-on idle fill or for tail padding experiments.
            """
            try:
                sr = 16000
                clip_sec = float(max(0.05, float(infer_frames) / float(max(1, int(self.fps)))))
                n = int(max(1, round(float(sr) * float(clip_sec))))
                fill_np = build_filler_pcm_f32(
                    samples=int(n),
                    mode=str(mode),
                    noise_std=float(stream_audio_fill_noise_std),
                    seed=int(stream_audio_fill_noise_seed) + int(seed_offset),
                )
                audio_emb_s, _ = self.encode_audio_from_array(fill_np, infer_frames=infer_frames)
                if audio_emb_s is None:
                    return None
                try:
                    t = int(audio_emb_s.shape[-1])
                except Exception:
                    t = 0
                if t <= 0:
                    return None
                if t < int(infer_frames):
                    pad_n = int(infer_frames) - int(t)
                    pad = torch.zeros(
                        *audio_emb_s.shape[:-1],
                        int(pad_n),
                        device=audio_emb_s.device,
                        dtype=audio_emb_s.dtype,
                    )
                    audio_emb_s = torch.cat([audio_emb_s, pad], dim=-1)
                return audio_emb_s[..., : int(infer_frames)].contiguous()
            except Exception as e:
                print(
                    f"Rank {dist.get_rank()}: build fill clip failed mode={str(mode)} err={e}",
                    flush=True,
                )
                return None

        def _stream_feature_alignment_from_meta(meta: dict | None) -> dict | None:
            if not isinstance(meta, dict):
                return None
            alignment = meta.get("subtitle_normalized_alignment")
            if isinstance(alignment, dict) and alignment:
                return dict(alignment)
            alignment = meta.get("subtitle_alignment")
            if isinstance(alignment, dict) and alignment:
                return dict(alignment)
            return None

        def _stream_feature_alignment_offset_sec(meta: dict | None, *, sample_rate: int) -> float:
            if not isinstance(meta, dict):
                return 0.0
            base_samples = meta.get("subtitle_alignment_base_samples")
            if not isinstance(base_samples, (int, float)):
                base_samples = meta.get("subtitle_start_samples")
            src_start_samples = meta.get("subtitle_start_samples")
            if not isinstance(src_start_samples, (int, float)):
                src_start_samples = base_samples
            if not isinstance(base_samples, (int, float)) or not isinstance(src_start_samples, (int, float)):
                return 0.0
            return float(int(round(float(base_samples))) - int(round(float(src_start_samples)))) / float(
                max(1, int(sample_rate or 16000))
            )

        def _stream_feature_neutral_clip(frame_count: int) -> torch.Tensor | None:
            frames_i = int(max(1, int(frame_count)))
            key = (int(frames_i), str(stream_audio_feature_neutral_mode))
            cached = stream_audio_feature_neutral_cache.get(key)
            if cached is not None:
                return cached
            try:
                sr = 16000
                clip_sec = float(frames_i) / float(max(1, int(self.fps)))
                samples = int(max(1, round(float(sr) * float(clip_sec))))
                fill_np = build_filler_pcm_f32(
                    samples=int(samples),
                    mode=str(stream_audio_feature_neutral_mode),
                    noise_std=float(stream_audio_fill_noise_std),
                    seed=int(stream_audio_fill_noise_seed) + 20000 + int(frames_i),
                )
                neutral, _ = self.encode_audio_from_array(fill_np, infer_frames=int(frames_i))
                if neutral is None:
                    return None
                t = int(neutral.shape[-1]) if neutral.ndim >= 1 else 0
                if t <= 0:
                    return None
                if t > int(frames_i):
                    neutral = neutral[..., : int(frames_i)].contiguous()
                elif t < int(frames_i):
                    pad_t = int(frames_i) - int(t)
                    last = neutral[..., -1:].contiguous()
                    neutral = torch.cat([neutral, last.repeat(*([1] * (neutral.ndim - 1)), int(pad_t))], dim=-1)
                neutral = neutral.contiguous()
                stream_audio_feature_neutral_cache[key] = neutral
                return neutral
            except Exception as e:
                print(
                    f"Rank {dist.get_rank()}: liveaudio feature neutral encode failed frames={frames_i}: {e}",
                    flush=True,
                )
                return None

        def _stream_feature_speech_envelope(
            meta: dict | None,
            *,
            frame_count: int,
            sample_rate: int,
        ) -> tuple[torch.Tensor | None, int]:
            if not bool(stream_audio_feature_speech_blend):
                return None, 0
            frames_i = int(max(0, int(frame_count)))
            if frames_i <= 0:
                return None, 0
            alignment = _stream_feature_alignment_from_meta(meta)
            if not isinstance(alignment, dict) or not alignment:
                return None, 0
            duration_sec = float(frames_i) / float(max(1, int(self.fps)))
            intervals = speech_intervals_from_alignment(
                alignment,
                alignment_offset_sec=float(_stream_feature_alignment_offset_sec(meta, sample_rate=int(sample_rate))),
                duration_sec=float(duration_sec),
            )
            if not intervals:
                return None, 0
            env = np.full((int(frames_i),), float(stream_audio_feature_speech_floor), dtype=np.float32)
            fade_frames = int(stream_audio_feature_speech_fade_frames)
            for start_sec, end_sec in intervals:
                start = int(max(0, min(frames_i, math.floor(float(start_sec) * float(self.fps)))))
                end = int(max(0, min(frames_i, math.ceil(float(end_sec) * float(self.fps)))))
                if end <= start:
                    continue
                local = np.ones((int(end - start),), dtype=np.float32)
                if fade_frames > 1 and int(local.size) > 1:
                    n = min(int(fade_frames), int(local.size))
                    local[:n] = np.minimum(
                        local[:n],
                        np.linspace(float(stream_audio_feature_speech_floor), 1.0, int(n), dtype=np.float32),
                    )
                    local[-n:] = np.minimum(
                        local[-n:],
                        np.linspace(1.0, float(stream_audio_feature_speech_floor), int(n), dtype=np.float32),
                    )
                env[start:end] = np.maximum(env[start:end], local)
            if float(np.max(env)) <= float(stream_audio_feature_speech_floor) + 1e-6:
                return None, 0
            tensor = torch.as_tensor(env, device=self.device, dtype=self.param_dtype)
            return tensor, int(len(intervals))

        def _apply_stream_feature_speech_blend(
            audio_emb_tensor: torch.Tensor | None,
            meta: dict | None,
            *,
            sample_rate: int,
        ) -> torch.Tensor | None:
            nonlocal stream_audio_feature_blend_events
            if audio_emb_tensor is None or not bool(stream_audio_feature_speech_blend):
                return audio_emb_tensor
            try:
                if audio_emb_tensor.ndim < 2:
                    return audio_emb_tensor
                frames_i = int(audio_emb_tensor.shape[-1])
                if frames_i <= 0:
                    return audio_emb_tensor
                envelope, intervals_n = _stream_feature_speech_envelope(
                    meta,
                    frame_count=int(frames_i),
                    sample_rate=int(sample_rate),
                )
                if envelope is None:
                    return audio_emb_tensor
                neutral = _stream_feature_neutral_clip(int(frames_i))
                if neutral is None or tuple(neutral.shape) != tuple(audio_emb_tensor.shape):
                    return audio_emb_tensor
                view_shape = [1] * int(audio_emb_tensor.ndim)
                view_shape[-1] = int(frames_i)
                env = envelope.reshape(view_shape).to(device=audio_emb_tensor.device, dtype=audio_emb_tensor.dtype)
                out = audio_emb_tensor * env + neutral.to(device=audio_emb_tensor.device, dtype=audio_emb_tensor.dtype) * (1.0 - env)
                stream_audio_feature_blend_events += 1
                if int(stream_audio_feature_blend_events) <= 3:
                    print(
                        f"Rank {dist.get_rank()}: liveaudio feature speech blend applied "
                        f"events={int(stream_audio_feature_blend_events)} frames={frames_i} intervals={int(intervals_n)} "
                        f"floor={float(stream_audio_feature_speech_floor):.2f} neutral={stream_audio_feature_neutral_mode}",
                        flush=True,
                    )
                return out.contiguous()
            except Exception as e:
                print(
                    f"Rank {dist.get_rank()}: liveaudio feature speech blend failed: {e}",
                    flush=True,
                )
                return audio_emb_tensor

        def _stream_sample_range_for_frames(start_frame: int, frame_count: int, sample_rate: int) -> tuple[int, int]:
            fps_i = int(max(1, int(round(float(self.fps)))))
            sr_i = int(max(1, int(sample_rate)))
            start = int((int(max(0, int(start_frame))) * sr_i) // fps_i)
            end = int(((int(max(0, int(start_frame))) + int(max(0, int(frame_count)))) * sr_i) // fps_i)
            return int(start), int(max(start, end))

        def _stream_pcm_target_bytes(frame_count: int, sample_rate: int) -> int:
            start, end = _stream_sample_range_for_frames(0, int(frame_count), int(sample_rate))
            return int(max(0, int(end - start)) * 2)

        def _fit_pcm16le_to_frames(pcm: bytes | bytearray | memoryview | None, *, sample_rate: int, frame_count: int) -> bytes:
            target_bytes = int(_stream_pcm_target_bytes(int(frame_count), int(sample_rate)))
            if target_bytes <= 0:
                return b""
            payload = bytes(pcm or b"")
            if len(payload) >= target_bytes:
                return bytes(payload[:target_bytes])
            return payload + (b"\x00" * int(target_bytes - len(payload)))

        def _slice_pcm16le_for_frames(
            pcm: bytes | bytearray | memoryview | None,
            *,
            sample_rate: int,
            start_frame: int,
            frame_count: int,
        ) -> bytes:
            start_sample, end_sample = _stream_sample_range_for_frames(
                int(start_frame),
                int(frame_count),
                int(sample_rate),
            )
            start_b = int(start_sample * 2)
            end_b = int(end_sample * 2)
            target_bytes = int(max(0, end_b - start_b))
            if target_bytes <= 0:
                return b""
            payload = bytes(pcm or b"")
            out = bytes(payload[start_b:end_b])
            if len(out) < target_bytes:
                out += b"\x00" * int(target_bytes - len(out))
            return out

        def _read_wav_mono16(wav_path: str) -> tuple[np.ndarray, bytes, int] | None:
            """
            Fast WAV read path for liveaudio chunks:
            - mono/stereo PCM16
            - returns float32 in [-1, 1], PCM16LE bytes, sample_rate
            """
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    ch = int(wf.getnchannels() or 1)
                    sw = int(wf.getsampwidth() or 2)
                    sr = int(wf.getframerate() or 16000)
                    n = int(wf.getnframes() or 0)
                    if n <= 0:
                        return None
                    raw = wf.readframes(n)
                if sw != 2:
                    return None
                x = np.frombuffer(raw, dtype=np.int16)
                if x.size <= 0:
                    return None
                if ch > 1:
                    keep = (x.size // ch) * ch
                    if keep <= 0:
                        return None
                    x = x[:keep].reshape(-1, ch).mean(axis=1).astype(np.int16)
                arr = (x.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
                return arr, x.astype(np.int16, copy=False).tobytes(), int(max(1, int(sr)))
            except Exception:
                return None

        def _read_wav_mono16_float32(wav_path: str) -> np.ndarray | None:
            info = _read_wav_mono16(wav_path)
            return None if info is None else info[0]

        def _remember_stream_audio_context(
            audio_emb_tensor: torch.Tensor | None,
            pcm16le: bytes | bytearray | memoryview | None,
            *,
            sample_rate: int,
            start_frame: int,
            end_frame: int,
            avatar_ref_path: str,
        ) -> None:
            nonlocal stream_audio_recent_context, stream_audio_recent_context_pcm
            nonlocal stream_audio_recent_context_sample_rate, stream_audio_recent_context_ref_path
            if audio_emb_tensor is None or audio_emb_tensor.ndim < 4:
                return
            total_t = int(audio_emb_tensor.shape[-1])
            start_i = int(max(0, min(int(total_t), int(start_frame))))
            end_i = int(max(0, min(int(total_t), int(end_frame))))
            if end_i <= start_i:
                return
            max_context = max(4, int(self.num_frames_per_block) * 4)
            if (end_i - start_i) > int(max_context):
                start_i = int(end_i) - int(max_context)
            try:
                stream_audio_recent_context = audio_emb_tensor[..., int(start_i): int(end_i)].detach().contiguous()
                stream_audio_recent_context_pcm = _slice_pcm16le_for_frames(
                    pcm16le,
                    sample_rate=int(sample_rate),
                    start_frame=int(start_i),
                    frame_count=int(end_i - start_i),
                )
                stream_audio_recent_context_sample_rate = int(max(1, int(sample_rate)))
                stream_audio_recent_context_ref_path = str(avatar_ref_path or "")
            except Exception:
                stream_audio_recent_context = None
                stream_audio_recent_context_pcm = None
                stream_audio_recent_context_ref_path = ""

        def _stream_tail_preroll_context(
            *,
            needed_frames: int,
            sample_rate: int,
            avatar_ref_path: str,
            device: torch.device | str,
            dtype: torch.dtype,
        ) -> tuple[torch.Tensor | None, bytes, int]:
            if not bool(stream_audio_tail_preroll):
                return None, b"", 0
            need_i = int(max(0, int(needed_frames)))
            if need_i <= 0:
                return None, b"", 0
            ctx = stream_audio_recent_context
            if ctx is None or ctx.ndim < 4:
                return None, b"", 0
            if str(stream_audio_recent_context_ref_path or "") != str(avatar_ref_path or ""):
                return None, b"", 0
            if int(stream_audio_recent_context_sample_rate) != int(sample_rate):
                return None, b"", 0
            ctx_t = int(ctx.shape[-1])
            if ctx_t <= 0:
                return None, b"", 0
            take = int(min(int(need_i), int(ctx_t)))
            try:
                ctx_tail = ctx[..., int(ctx_t - take): int(ctx_t)].to(device=device, dtype=dtype).contiguous()
                ctx_pcm = _slice_pcm16le_for_frames(
                    stream_audio_recent_context_pcm,
                    sample_rate=int(sample_rate),
                    start_frame=int(ctx_t - take),
                    frame_count=int(take),
                )
                if take < need_i:
                    pad_n = int(need_i - take)
                    pad = ctx_tail[..., :1].expand(*ctx_tail.shape[:-1], int(pad_n)).contiguous()
                    ctx_tail = torch.cat([pad, ctx_tail], dim=-1)
                    first_pcm = _slice_pcm16le_for_frames(
                        stream_audio_recent_context_pcm,
                        sample_rate=int(sample_rate),
                        start_frame=int(max(0, int(ctx_t - take))),
                        frame_count=1,
                    )
                    frame_samples = max(1, int(round(float(sample_rate) / float(max(1, int(self.fps))))))
                    ctx_pcm = (bytes(first_pcm[:2]) * int(frame_samples * pad_n)) + bytes(ctx_pcm)
                    take = int(need_i)
                return ctx_tail, bytes(ctx_pcm), int(take)
            except Exception:
                return None, b"", 0

        def _enqueue_stream_audio_tail_clip(*, reason: str) -> int:
            nonlocal stream_audio_tail, stream_audio_tail_kind, stream_audio_tail_source_idx
            nonlocal stream_audio_tail_ref_path, stream_audio_tail_pcm, stream_audio_tail_sample_rate
            nonlocal stream_audio_tail_audible_samples, stream_audio_tail_visible_frames, stream_audio_tail_flushed
            tail_t = int(stream_audio_tail.shape[-1]) if (stream_audio_tail is not None and stream_audio_tail.ndim >= 4) else 0
            if tail_t <= 0:
                return 0
            clip_tail = stream_audio_tail
            block_frames = max(4, int(self.num_frames_per_block) * 4)
            min_tail_frames = int(max(block_frames, int(math.ceil(float(tail_t) / float(block_frames)) * block_frames)))
            visible_start_frames = 0
            visible_frames = int(max(0, min(int(stream_audio_tail_visible_frames or tail_t), int(tail_t))))
            pre_requested = 0
            pre_taken_actual = 0
            pre_kind = "none"
            tiny_lead_trim_frames = 0
            if bool(stream_audio_tail_preroll):
                fill_needed = int(max(0, int(min_tail_frames) - int(tail_t)))
                min_context = int(stream_audio_tail_preroll_frames)
                if min_context <= 0:
                    min_context = int(fill_needed)
                desired_context = int(max(int(fill_needed), int(min_context)))
                if int(fill_needed) > 0 and int(tail_t) + int(desired_context) > int(min_tail_frames):
                    # Avoid growing a near-complete tail into one more decode block just
                    # because the configured minimum context is a few frames larger than
                    # the block-alignment fill.
                    desired_context = int(fill_needed)
                pre_requested = int(min(int(block_frames), max(0, int(desired_context))))
                pre, pre_pcm, pre_taken = _stream_tail_preroll_context(
                    needed_frames=int(pre_requested),
                    sample_rate=int(stream_audio_tail_sample_rate),
                    avatar_ref_path=str(stream_audio_tail_ref_path or ""),
                    device=clip_tail.device,
                    dtype=clip_tail.dtype,
                )
                if (pre is None or int(pre_taken) <= 0) and int(pre_requested) > 0:
                    try:
                        pre = clip_tail[..., :1].expand(*clip_tail.shape[:-1], int(pre_requested)).contiguous()
                        pre_taken = int(pre_requested)
                        pre_kind = "synthetic"
                    except Exception:
                        pre = None
                        pre_taken = 0
                if pre is not None and int(pre_taken) > 0:
                    clip_tail = torch.cat([pre, clip_tail], dim=-1)
                    visible_start_frames = int(pre_taken)
                    pre_taken_actual = int(pre_taken)
                    if pre_kind == "none":
                        pre_kind = "context"
                    min_tail_frames = int(
                        max(
                            int(block_frames),
                            int(
                                math.ceil(
                                    float(int(tail_t) + int(visible_start_frames)) / float(block_frames)
                                )
                            )
                            * int(block_frames),
                        )
                    )
            tail_fill_kind = "zero"
            clip_t_now = int(clip_tail.shape[-1])
            if clip_t_now < int(min_tail_frames):
                pad_n = int(min_tail_frames) - int(clip_t_now)
                pad = None
                if stream_audio_tail_fill_clip is not None:
                    try:
                        pad = stream_audio_tail_fill_clip[..., : int(pad_n)].to(
                            device=clip_tail.device,
                            dtype=clip_tail.dtype,
                        ).contiguous()
                        if int(pad.shape[-1]) == int(pad_n):
                            tail_fill_kind = str(stream_audio_tail_fill_mode)
                        else:
                            pad = None
                    except Exception:
                        pad = None
                if pad is None:
                    if str(stream_audio_tail_fill_mode) == "hold" and int(tail_t) > 0:
                        pad = clip_tail[..., -1:].expand(*clip_tail.shape[:-1], int(pad_n)).contiguous()
                        tail_fill_kind = "hold"
                    else:
                        pad = torch.zeros(
                            *clip_tail.shape[:-1],
                            int(pad_n),
                            device=clip_tail.device,
                            dtype=clip_tail.dtype,
                        )
                clip_tail = torch.cat([clip_tail, pad], dim=-1)
            clip_tail_kind = normalize_stream_clip_kind(stream_audio_tail_kind)
            boundary_reason = str(reason or "").strip().lower()
            if (
                boundary_reason == "avatar_ref_boundary"
                and int(visible_start_frames) > 0
                and int(visible_frames) > 0
            ):
                try:
                    tiny_threshold = int(
                        os.getenv("LIVE_AUDIO_STREAM_TAIL_TINY_LEADING_TRIM_FRAMES", "5") or "5"
                    )
                except Exception:
                    tiny_threshold = 5
                tiny_threshold = int(max(0, min(int(block_frames) - 1, int(tiny_threshold))))
                visible_offset = int(visible_start_frames) % int(block_frames)
                if int(tiny_threshold) > 0 and int(visible_offset) > 0:
                    leading_visible_frames = int(min(int(visible_frames), int(block_frames) - int(visible_offset)))
                    if (
                        int(leading_visible_frames) > 0
                        and int(leading_visible_frames) <= int(tiny_threshold)
                        and int(visible_frames) > int(leading_visible_frames)
                    ):
                        visible_start_frames = int(visible_start_frames) + int(leading_visible_frames)
                        visible_frames = int(visible_frames) - int(leading_visible_frames)
                        tiny_lead_trim_frames = int(leading_visible_frames)
            tail_pcm_source = stream_audio_tail_pcm
            tail_audible_samples = int(stream_audio_tail_audible_samples)
            if int(tiny_lead_trim_frames) > 0:
                tail_pcm_source = _slice_pcm16le_for_frames(
                    stream_audio_tail_pcm,
                    sample_rate=int(stream_audio_tail_sample_rate),
                    start_frame=int(tiny_lead_trim_frames),
                    frame_count=int(max(0, int(tail_t) - int(tiny_lead_trim_frames))),
                )
                trim_start_samples, trim_end_samples = _stream_sample_range_for_frames(
                    0,
                    int(tiny_lead_trim_frames),
                    int(stream_audio_tail_sample_rate),
                )
                tail_audible_samples = int(
                    max(0, int(tail_audible_samples) - int(max(0, int(trim_end_samples) - int(trim_start_samples))))
                )
            stream_audio_clips.append(clip_tail.contiguous())
            stream_audio_clip_kinds.append(str(clip_tail_kind))
            stream_audio_clip_source_idxs.append(
                int(max(0, int(stream_audio_tail_source_idx or stream_audio_next_idx)))
            )
            stream_audio_clip_ref_paths.append(str(stream_audio_tail_ref_path or ""))
            tail_pcm = _fit_pcm16le_to_frames(
                tail_pcm_source,
                sample_rate=int(stream_audio_tail_sample_rate),
                frame_count=int(min_tail_frames),
            )
            stream_audio_clip_pcms.append(bytes(tail_pcm))
            stream_audio_clip_sample_rates.append(int(stream_audio_tail_sample_rate))
            stream_audio_clip_audible_samples.append(
                int(max(0, min(int(tail_audible_samples), int(len(tail_pcm) // 2))))
            )
            stream_audio_clip_visible_start_frames.append(int(visible_start_frames))
            stream_audio_clip_visible_frames.append(int(visible_frames))
            stream_audio_tail = None
            stream_audio_tail_kind = None
            stream_audio_tail_source_idx = None
            stream_audio_tail_ref_path = ""
            stream_audio_tail_pcm = None
            stream_audio_tail_audible_samples = 0
            stream_audio_tail_visible_frames = 0
            if stream_timing_log:
                print(
                    f"Rank {dist.get_rank()}: liveaudio flushed tail clips_added=1 "
                    f"(reason={str(reason or '-')} tail={int(tail_t)} min_tail={int(min_tail_frames)} "
                    f"visible_start={int(visible_start_frames)} "
                    f"visible={int(stream_audio_clip_visible_frames[-1] if len(stream_audio_clip_visible_frames) > 0 else tail_t)} "
                    f"tiny_lead_trim={int(tiny_lead_trim_frames)} "
                    f"infer_frames={int(infer_frames)} fill={str(tail_fill_kind)} "
                    f"preroll_requested={int(pre_requested)} preroll={int(pre_taken_actual)} "
                    f"preroll_kind={str(pre_kind)})",
                    flush=True,
                )
            return 1

        def _flush_stream_audio_tail_clip(*, reason: str, mark_flushed: bool) -> int:
            nonlocal stream_audio_tail_flushed
            added = int(_enqueue_stream_audio_tail_clip(reason=str(reason or "flush")))
            if added <= 0 and bool(mark_flushed):
                stream_audio_tail_flushed = True
            elif added > 0 and bool(mark_flushed):
                stream_audio_tail_flushed = True
            return int(added)

        def _stream_expected_video_frames_from_wav(wav_path: str) -> int:
            """
            Estimate how many video-frame conditions this WAV should produce at self.fps.
            This prevents per-chunk over-padding inside audio bucketing from accumulating
            lip-sync drift when chunks are fed incrementally.
            """
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    sr = int(wf.getframerate() or 0)
                    n = int(wf.getnframes() or 0)
                if sr <= 0 or n <= 0:
                    return 1
                # Use ceil to avoid cumulative frame under-run across many chunks.
                est = int(math.ceil((float(n) * float(self.fps)) / float(sr)))
                try:
                    # Keep liveaudio timing deterministic across ranks.
                    pad_frames = int(os.getenv("LIVE_AUDIO_STREAM_EXP_FRAMES_PAD", "0") or 0)
                except Exception:
                    pad_frames = 0
                pad_frames = max(0, min(8, int(pad_frames)))
                est += int(pad_frames)
                return max(1, int(est))
            except Exception:
                return 1

        def _stream_expected_video_frames_from_meta_or_wav(chunk_idx: int, wav_path: str) -> int:
            try:
                meta = _stream_chunk_meta(int(chunk_idx))
                conditioning_frames = int(meta.get("conditioning_frames") or 0)
                if conditioning_frames > 0:
                    return int(conditioning_frames)
                frames = int(meta.get("source_frames") or 0)
                if frames > 0:
                    return int(frames)
            except Exception:
                pass
            return int(_stream_expected_video_frames_from_wav(str(wav_path)))

        def _append_stream_audio_clips(
            audio_emb_tensor: torch.Tensor,
            *,
            clip_kind: str = "speech",
            source_chunk_idx: int = 0,
            pcm16le: bytes | bytearray | memoryview | None = None,
            sample_rate: int | None = None,
            audible_samples: int | None = None,
            visible_start_frames: int | None = None,
            visible_frames: int | None = None,
            embedded_visible_start_frames: bool = False,
            avatar_ref_path: str | None = None,
        ) -> int:
            nonlocal stream_audio_tail, stream_audio_tail_kind, stream_audio_tail_source_idx
            nonlocal stream_audio_tail_ref_path, stream_audio_tail_pcm, stream_audio_tail_sample_rate, stream_audio_tail_audible_samples
            nonlocal stream_audio_tail_visible_frames
            nonlocal stream_audio_tail_flushed
            added = 0
            if audio_emb_tensor is None:
                return 0
            clip_kind_norm = normalize_stream_clip_kind(clip_kind)
            merged_head_kind: str | None = None
            merged_head_source_idx: int | None = None
            avatar_ref_path_s = str(avatar_ref_path or "").strip()
            if avatar_ref_path_s and os.path.exists(avatar_ref_path_s):
                avatar_ref_path_s = os.path.abspath(avatar_ref_path_s)
            else:
                avatar_ref_path_s = ""
            tail_ref_path_s = str(stream_audio_tail_ref_path or "").strip()
            if stream_audio_tail is not None:
                if not avatar_ref_path_s and tail_ref_path_s:
                    # Some generated TTS chunks do not repeat the avatar ref. Treat a
                    # missing ref as "same current avatar"; flushing here creates a
                    # cold-start tail clip and visible face jumps near speech endings.
                    avatar_ref_path_s = str(tail_ref_path_s)
                elif avatar_ref_path_s and not tail_ref_path_s:
                    stream_audio_tail_ref_path = str(avatar_ref_path_s)
                    tail_ref_path_s = str(avatar_ref_path_s)
            if (
                stream_audio_tail is not None
                and avatar_ref_path_s
                and tail_ref_path_s
                and avatar_ref_path_s != tail_ref_path_s
            ):
                added += int(
                    _flush_stream_audio_tail_clip(
                        reason="avatar_ref_boundary",
                        mark_flushed=False,
                    )
                )
            sample_rate_i = int(max(1, int(sample_rate or stream_audio_tail_sample_rate or 16000)))
            input_t = int(audio_emb_tensor.shape[-1]) if audio_emb_tensor.ndim >= 4 else 0
            try:
                current_visible_start_i = int(visible_start_frames if visible_start_frames is not None else 0)
            except Exception:
                current_visible_start_i = 0
            current_visible_start_i = int(max(0, min(int(current_visible_start_i), int(input_t))))
            try:
                current_visible_frames_i = int(visible_frames if visible_frames is not None else input_t)
            except Exception:
                current_visible_frames_i = int(input_t)
            current_visible_frames_i = int(
                max(0, min(int(current_visible_frames_i), int(input_t) - int(current_visible_start_i)))
            )
            pcm_payload = _fit_pcm16le_to_frames(pcm16le, sample_rate=int(sample_rate_i), frame_count=int(input_t))
            try:
                audible_samples_i = int(audible_samples if audible_samples is not None else (len(pcm_payload) // 2))
            except Exception:
                audible_samples_i = int(len(pcm_payload) // 2)
            audible_samples_i = int(max(0, min(int(audible_samples_i), int(len(pcm_payload) // 2))))
            visible_start_frames_i = int(current_visible_start_i)
            visible_frames_i = int(current_visible_frames_i)
            if stream_audio_tail is not None:
                try:
                    try:
                        merged_head_source_idx = int(max(0, int(stream_audio_tail_source_idx or 0)))
                    except Exception:
                        merged_head_source_idx = None
                    if int(stream_audio_tail_sample_rate) != int(sample_rate_i):
                        raise RuntimeError("liveaudio tail PCM sample-rate mismatch")
                    audio_emb_tensor = torch.cat([stream_audio_tail, audio_emb_tensor], dim=-1)
                    pcm_payload = bytes(stream_audio_tail_pcm or b"") + bytes(pcm_payload)
                    audible_samples_i = int(max(0, int(stream_audio_tail_audible_samples))) + int(audible_samples_i)
                    visible_start_frames_i = 0
                    visible_frames_i = int(max(0, int(stream_audio_tail_visible_frames))) + int(current_visible_frames_i)
                    merged_head_kind = merge_stream_clip_kinds(stream_audio_tail_kind, clip_kind_norm)
                except Exception:
                    # If concat fails for any reason, drop stale tail instead of poisoning stream.
                    stream_audio_tail = None
                    stream_audio_tail_kind = None
                    stream_audio_tail_source_idx = None
                    merged_head_source_idx = None
                    stream_audio_tail_pcm = None
                    stream_audio_tail_audible_samples = 0
                    stream_audio_tail_visible_frames = 0
                    stream_audio_tail_ref_path = ""
                    visible_start_frames_i = int(current_visible_start_i)
                    visible_frames_i = int(current_visible_frames_i)
            total_t = int(audio_emb_tensor.shape[-1]) if audio_emb_tensor.ndim >= 4 else 0
            if total_t <= 0:
                return 0
            visible_start_frames_i = int(max(0, min(int(visible_start_frames_i), int(total_t))))
            if (
                int(visible_start_frames_i) > 0
                and stream_audio_tail is None
                and not bool(embedded_visible_start_frames)
            ):
                pre, _pre_pcm, pre_taken = _stream_tail_preroll_context(
                    needed_frames=int(visible_start_frames_i),
                    sample_rate=int(sample_rate_i),
                    avatar_ref_path=str(avatar_ref_path_s or stream_audio_tail_ref_path or ""),
                    device=audio_emb_tensor.device,
                    dtype=audio_emb_tensor.dtype,
                )
                if pre is not None and int(pre_taken) > 0:
                    pre_taken_i = int(min(int(visible_start_frames_i), int(pre_taken)))
                    keep_t = int(max(0, int(total_t) - int(pre_taken_i)))
                    audio_emb_tensor = torch.cat(
                        [
                            pre[..., -int(pre_taken_i) :].contiguous(),
                            audio_emb_tensor[..., : int(keep_t)].contiguous(),
                        ],
                        dim=-1,
                    ).contiguous()
                    total_t = int(audio_emb_tensor.shape[-1])
                    visible_start_frames_i = int(pre_taken_i)
                    if stream_timing_log:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio boundary preroll applied "
                            f"chunk={int(source_chunk_idx)} pre={int(pre_taken_i)} "
                            f"visible={int(visible_frames_i)} total={int(total_t)}",
                            flush=True,
                        )
                else:
                    # Without an actual previous context, do not hide the start of this chunk.
                    visible_start_frames_i = 0
            visible_frames_i = int(
                max(0, min(int(visible_frames_i), int(total_t) - int(visible_start_frames_i)))
            )
            pcm_payload = _fit_pcm16le_to_frames(pcm_payload, sample_rate=int(sample_rate_i), frame_count=int(total_t))
            audible_samples_i = int(max(0, min(int(audible_samples_i), int(len(pcm_payload) // 2))))
            block_frames = max(4, int(self.num_frames_per_block) * 4)
            allow_single_clip = bool(
                liveaudio_allow_long_clips()
                and int(total_t) >= int(infer_frames)
                and int(total_t) <= int(liveaudio_max_clip_frames(int(infer_frames)))
                and (int(total_t) % int(block_frames)) == 0
            )
            if bool(allow_single_clip):
                stream_audio_clips.append(audio_emb_tensor.contiguous())
                stream_audio_clip_pcms.append(bytes(pcm_payload))
                stream_audio_clip_sample_rates.append(int(sample_rate_i))
                stream_audio_clip_audible_samples.append(
                    int(max(0, min(int(audible_samples_i), int(len(pcm_payload) // 2))))
                )
                stream_audio_clip_visible_start_frames.append(int(visible_start_frames_i))
                stream_audio_clip_visible_frames.append(int(max(0, min(int(visible_frames_i), int(total_t)))))
                if merged_head_kind is not None:
                    stream_audio_clip_kinds.append(str(merged_head_kind))
                    clip_source_idx = (
                        int(merged_head_source_idx)
                        if merged_head_source_idx is not None and int(merged_head_source_idx) > 0
                        else int(max(0, int(source_chunk_idx)))
                    )
                else:
                    stream_audio_clip_kinds.append(str(clip_kind_norm))
                    clip_source_idx = int(max(0, int(source_chunk_idx)))
                stream_audio_clip_source_idxs.append(int(max(0, int(clip_source_idx))))
                stream_audio_clip_ref_paths.append(str(avatar_ref_path_s or stream_audio_tail_ref_path or ""))
                _remember_stream_audio_context(
                    audio_emb_tensor,
                    pcm_payload,
                    sample_rate=int(sample_rate_i),
                    start_frame=0,
                    end_frame=int(total_t),
                    avatar_ref_path=str(avatar_ref_path_s or stream_audio_tail_ref_path or ""),
                )
                stream_audio_tail = None
                stream_audio_tail_kind = None
                stream_audio_tail_source_idx = None
                stream_audio_tail_ref_path = ""
                stream_audio_tail_pcm = None
                stream_audio_tail_audible_samples = 0
                stream_audio_tail_visible_frames = 0
                return int(added + 1)
            pos = 0
            while (total_t - pos) >= int(infer_frames):
                clip = audio_emb_tensor[..., pos : pos + int(infer_frames)]
                stream_audio_clips.append(clip.contiguous())
                clip_pcm = _slice_pcm16le_for_frames(
                    pcm_payload,
                    sample_rate=int(sample_rate_i),
                    start_frame=int(pos),
                    frame_count=int(infer_frames),
                )
                clip_start_samples, clip_end_samples = _stream_sample_range_for_frames(
                    int(pos),
                    int(infer_frames),
                    int(sample_rate_i),
                )
                clip_audible = int(max(0, min(int(clip_end_samples), int(audible_samples_i)) - int(clip_start_samples)))
                visible_global_start = int(visible_start_frames_i)
                visible_global_end = int(visible_global_start) + int(visible_frames_i)
                clip_global_start = int(pos)
                clip_global_end = int(pos) + int(infer_frames)
                clip_visible_start = int(max(0, int(max(clip_global_start, visible_global_start)) - int(clip_global_start)))
                clip_visible = int(
                    max(
                        0,
                        int(min(clip_global_end, visible_global_end)) - int(max(clip_global_start, visible_global_start)),
                    )
                )
                stream_audio_clip_pcms.append(bytes(clip_pcm))
                stream_audio_clip_sample_rates.append(int(sample_rate_i))
                stream_audio_clip_audible_samples.append(int(clip_audible))
                stream_audio_clip_visible_start_frames.append(int(clip_visible_start))
                stream_audio_clip_visible_frames.append(int(clip_visible))
                if pos == 0 and merged_head_kind is not None:
                    stream_audio_clip_kinds.append(str(merged_head_kind))
                    clip_source_idx = (
                        int(merged_head_source_idx)
                        if merged_head_source_idx is not None and int(merged_head_source_idx) > 0
                        else int(max(0, int(source_chunk_idx)))
                    )
                else:
                    stream_audio_clip_kinds.append(str(clip_kind_norm))
                    clip_source_idx = int(max(0, int(source_chunk_idx)))
                stream_audio_clip_source_idxs.append(int(max(0, int(clip_source_idx))))
                stream_audio_clip_ref_paths.append(str(avatar_ref_path_s or stream_audio_tail_ref_path or ""))
                _remember_stream_audio_context(
                    audio_emb_tensor,
                    pcm_payload,
                    sample_rate=int(sample_rate_i),
                    start_frame=int(pos),
                    end_frame=int(pos) + int(infer_frames),
                    avatar_ref_path=str(avatar_ref_path_s or stream_audio_tail_ref_path or ""),
                )
                added += 1
                pos += int(infer_frames)
            rem = int(total_t - pos)
            if rem > 0:
                stream_audio_tail = audio_emb_tensor[..., pos:].contiguous()
                stream_audio_tail_kind = str(clip_kind_norm)
                stream_audio_tail_source_idx = int(max(0, int(source_chunk_idx)))
                stream_audio_tail_ref_path = str(avatar_ref_path_s or stream_audio_tail_ref_path or "")
                stream_audio_tail_pcm = _slice_pcm16le_for_frames(
                    pcm_payload,
                    sample_rate=int(sample_rate_i),
                    start_frame=int(pos),
                    frame_count=int(rem),
                )
                tail_start_samples, tail_end_samples = _stream_sample_range_for_frames(
                    int(pos),
                    int(rem),
                    int(sample_rate_i),
                )
                stream_audio_tail_sample_rate = int(sample_rate_i)
                stream_audio_tail_audible_samples = int(
                    max(0, min(int(tail_end_samples), int(audible_samples_i)) - int(tail_start_samples))
                )
                stream_audio_tail_visible_frames = int(max(0, min(int(rem), int(visible_frames_i) - int(pos))))
                stream_audio_tail_flushed = False
            else:
                stream_audio_tail = None
                stream_audio_tail_kind = None
                stream_audio_tail_source_idx = None
                stream_audio_tail_ref_path = ""
                stream_audio_tail_pcm = None
                stream_audio_tail_audible_samples = 0
                stream_audio_tail_visible_frames = 0
            return int(added)

        def _stream_audio_pending_clip_target() -> int:
            return int(
                pending_clip_target(
                    max_pending_clips=int(stream_audio_max_pending_clips),
                    is_always_on=bool(stream_audio_is_always_on),
                )
            )

        def _stream_refill_audio_clips(*, min_required: int = 1, block_until_ready: bool = False) -> int:
            nonlocal stream_audio_done, stream_audio_done_status, stream_audio_done_chunks_total
            nonlocal stream_audio_cancelled
            nonlocal stream_audio_next_idx, stream_audio_seen_chunks
            nonlocal stream_audio_tail, stream_audio_tail_flushed, stream_audio_tail_kind
            nonlocal stream_audio_tail_source_idx, stream_audio_skip_before_chunk_idx
            nonlocal stream_audio_tail_ref_path, stream_audio_tail_pcm, stream_audio_tail_sample_rate, stream_audio_tail_audible_samples
            nonlocal stream_audio_tail_visible_frames
            min_required = max(0, int(min_required))
            with stream_audio_refill_lock:
                while True:
                    queue_target = int(_stream_audio_pending_clip_target())
                    skip_before_chunk_idx = int(max(0, _stream_soft_break_chunk_idx()))
                    if int(skip_before_chunk_idx) > int(stream_audio_skip_before_chunk_idx):
                        stream_audio_skip_before_chunk_idx = int(skip_before_chunk_idx)
                    if (
                        stream_audio_tail is not None
                        and stream_audio_tail_source_idx is not None
                        and int(stream_audio_tail_source_idx) < int(stream_audio_skip_before_chunk_idx)
                    ):
                        stream_audio_tail = None
                        stream_audio_tail_kind = None
                        stream_audio_tail_source_idx = None
                        stream_audio_tail_ref_path = ""
                        stream_audio_tail_pcm = None
                        stream_audio_tail_audible_samples = 0
                        stream_audio_tail_visible_frames = 0
                    while (
                        len(stream_audio_clips) > 0
                        and len(stream_audio_clip_source_idxs) > 0
                        and int(stream_audio_clip_source_idxs[0]) < int(stream_audio_skip_before_chunk_idx)
                    ):
                        try:
                            stream_audio_clips.popleft()
                        except Exception:
                            break
                        try:
                            if len(stream_audio_clip_kinds) > 0:
                                stream_audio_clip_kinds.popleft()
                        except Exception:
                            pass
                        try:
                            stream_audio_clip_source_idxs.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_ref_paths) > 0:
                                stream_audio_clip_ref_paths.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_pcms) > 0:
                                stream_audio_clip_pcms.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_sample_rates) > 0:
                                stream_audio_clip_sample_rates.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_audible_samples) > 0:
                                stream_audio_clip_audible_samples.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_visible_start_frames) > 0:
                                stream_audio_clip_visible_start_frames.popleft()
                        except Exception:
                            pass
                        try:
                            if len(stream_audio_clip_visible_frames) > 0:
                                stream_audio_clip_visible_frames.popleft()
                        except Exception:
                            pass
                        _stream_write_progress_marker()
                    loaded_any = False
                    loaded_count_this_call = 0
                    while True:
                        if int(loaded_count_this_call) >= int(stream_audio_refill_max_chunks_per_call):
                            break
                        if len(stream_audio_clips) >= int(queue_target):
                            break
                        if int(stream_audio_next_idx) < int(stream_audio_skip_before_chunk_idx):
                            stream_audio_next_idx += 1
                            _stream_write_progress_marker()
                            loaded_any = True
                            continue
                        chunk_path = _stream_chunk_path(stream_audio_next_idx)
                        if not os.path.exists(chunk_path):
                            break
                        stream_audio_seen_chunks += 1
                        try:
                            enc_t0 = time.perf_counter()
                            expected_frames = _stream_expected_video_frames_from_meta_or_wav(
                                stream_audio_next_idx,
                                chunk_path,
                            )
                            chunk_kind = _stream_chunk_kind(stream_audio_next_idx)
                            chunk_avatar_ref_path = _stream_chunk_avatar_ref_path(stream_audio_next_idx)
                            expected_frames = chunk_conditioning_target_frames(
                                expected_frames=int(expected_frames),
                                infer_frames=int(infer_frames),
                                clip_kind=str(normalize_stream_clip_kind(str(chunk_kind))),
                            )
                            chunk_batch_frames = chunk_encode_batch_frames(
                                expected_frames=int(expected_frames),
                                infer_frames=int(infer_frames),
                            )
                            chunk_meta = _stream_chunk_meta(stream_audio_next_idx)
                            chunk_info = _read_wav_mono16(chunk_path)
                            chunk_arr = chunk_info[0] if chunk_info is not None else None
                            chunk_pcm = chunk_info[1] if chunk_info is not None else b""
                            chunk_sample_rate = int(chunk_info[2]) if chunk_info is not None else int(stream_audio_tail_sample_rate)
                            lipsync_audio_path = str(
                                chunk_meta.get("lipsync_audio_path")
                                or chunk_meta.get("lipsyncAudioPath")
                                or ""
                            ).strip()
                            if lipsync_audio_path and (not os.path.exists(lipsync_audio_path)):
                                lipsync_audio_path = ""
                            encode_info = _read_wav_mono16(lipsync_audio_path) if lipsync_audio_path else chunk_info
                            encode_arr = encode_info[0] if encode_info is not None else None
                            try:
                                chunk_source_samples = int(chunk_meta.get("source_samples") or (len(chunk_pcm) // 2))
                            except Exception:
                                chunk_source_samples = int(len(chunk_pcm) // 2)
                            chunk_audible = int(max(0, min(int(chunk_source_samples), int(len(chunk_pcm) // 2))))
                            try:
                                chunk_visible_frames = int(
                                    chunk_meta.get("visible_frames")
                                    or chunk_meta.get("source_frames")
                                    or expected_frames
                                )
                            except Exception:
                                chunk_visible_frames = int(expected_frames)
                            try:
                                chunk_visible_start_frames = int(
                                    chunk_meta.get("visible_start_frames")
                                    or chunk_meta.get("visible_start")
                                    or 0
                                )
                            except Exception:
                                chunk_visible_start_frames = 0
                            chunk_embedded_visible_start_frames = bool(
                                (chunk_meta or {}).get("embedded_visible_start_frames", False)
                            )
                            chunk_visible_start_frames = int(
                                max(0, min(int(chunk_visible_start_frames), int(expected_frames)))
                            )
                            chunk_visible_frames = int(
                                max(
                                    0,
                                    min(
                                        int(chunk_visible_frames),
                                        int(expected_frames) - int(chunk_visible_start_frames),
                                    ),
                                )
                            )
                            if bool(chunk_meta) and (
                                chunk_meta.get("audible") is False
                                or normalize_stream_clip_kind(chunk_kind) in {"gap_fill", "filler", "silence"}
                            ):
                                chunk_audible = 0
                            enc_src = "lipsync_array" if lipsync_audio_path else "array"
                            if encode_arr is not None:
                                audio_emb_i, _ = self.encode_audio_from_array(
                                    encode_arr, infer_frames=int(chunk_batch_frames)
                                )
                            else:
                                enc_src = "lipsync_file" if lipsync_audio_path else "file"
                                audio_emb_i, _ = self.encode_audio(
                                    lipsync_audio_path or chunk_path,
                                    infer_frames=int(chunk_batch_frames),
                                )
                            enc_dt = float(time.perf_counter() - enc_t0)
                            try:
                                emb_t = int(audio_emb_i.shape[-1]) if audio_emb_i is not None else 0
                            except Exception:
                                emb_t = 0
                            # Keep liveaudio chunk timing exact at the WAV-derived frame count.
                            # Overrun creates visual tail drift; underrun accumulates into a
                            # visibly truncated ending on longer replies. Clamp down when the
                            # encoder overshoots, and pad up when it undershoots.
                            orig_emb_t = int(emb_t)
                            if emb_t > 0 and expected_frames > 0:
                                target_frames = int(expected_frames)
                                if emb_t > target_frames:
                                    audio_emb_i = audio_emb_i[..., :target_frames].contiguous()
                                    emb_t = int(target_frames)
                                elif emb_t < target_frames:
                                    pad_t = int(target_frames) - int(emb_t)
                                    pad = torch.zeros(
                                        *audio_emb_i.shape[:-1],
                                        int(pad_t),
                                        device=audio_emb_i.device,
                                        dtype=audio_emb_i.dtype,
                                    )
                                    audio_emb_i = torch.cat([audio_emb_i, pad], dim=-1).contiguous()
                                    emb_t = int(target_frames)
                            if orig_emb_t > 0 and expected_frames > 0 and orig_emb_t != int(expected_frames):
                                print(
                                    f"Rank {dist.get_rank()}: liveaudio expected/emb mismatch "
                                    f"chunk={stream_audio_next_idx} expected={int(expected_frames)} emb={int(orig_emb_t)} corrected={int(emb_t)}",
                                    flush=True,
                                )
                            if bool(stream_audio_feature_speech_blend) and int(chunk_audible) > 0:
                                audio_emb_i = _apply_stream_feature_speech_blend(
                                    audio_emb_i,
                                    chunk_meta,
                                    sample_rate=int(chunk_sample_rate),
                                )
                            added = _append_stream_audio_clips(
                                audio_emb_i,
                                clip_kind=str(chunk_kind),
                                source_chunk_idx=int(stream_audio_next_idx),
                                pcm16le=_fit_pcm16le_to_frames(
                                    chunk_pcm,
                                    sample_rate=int(chunk_sample_rate),
                                    frame_count=int(expected_frames),
                                ),
                                sample_rate=int(chunk_sample_rate),
                                audible_samples=int(chunk_audible),
                                visible_start_frames=int(chunk_visible_start_frames),
                                visible_frames=int(chunk_visible_frames),
                                embedded_visible_start_frames=bool(chunk_embedded_visible_start_frames),
                                avatar_ref_path=str(chunk_avatar_ref_path or ""),
                            )
                            stream_live_trace.note_chunk_loaded(
                                chunk_idx=int(stream_audio_next_idx),
                                clips_added=int(added),
                                expected_frames=int(expected_frames),
                                enc_src=str(enc_src),
                                enc_dt=float(enc_dt),
                                queue_depth=int(len(stream_audio_clips)),
                                seen_chunks=int(stream_audio_seen_chunks),
                            )
                        except Exception as e:
                            print(
                                f"Rank {dist.get_rank()}: liveaudio encode failed chunk={stream_audio_next_idx}: {e}",
                                flush=True,
                            )
                        stream_audio_next_idx += 1
                        _stream_write_progress_marker()
                        loaded_any = True
                        loaded_count_this_call += 1

                    if _stream_done_marker_exists():
                        stream_audio_done = True
                        done_meta = _stream_done_marker_meta()
                        try:
                            stream_audio_done_status = str(done_meta.get("status") or "ok").strip().lower() or "ok"
                        except Exception:
                            stream_audio_done_status = "ok"
                        try:
                            stream_audio_done_chunks_total = int(done_meta.get("chunks_total", -1))
                        except Exception:
                            stream_audio_done_chunks_total = -1
                        if str(stream_audio_done_status) == "cancelled":
                            stream_audio_cancelled = True

                        if bool(stream_audio_cancelled):
                            if len(stream_audio_clips) > 0 or stream_audio_tail is not None:
                                try:
                                    stream_audio_clips.clear()
                                    stream_audio_clip_kinds.clear()
                                    stream_audio_clip_source_idxs.clear()
                                    stream_audio_clip_ref_paths.clear()
                                    stream_audio_clip_pcms.clear()
                                    stream_audio_clip_sample_rates.clear()
                                    stream_audio_clip_audible_samples.clear()
                                    stream_audio_clip_visible_start_frames.clear()
                                    stream_audio_clip_visible_frames.clear()
                                except Exception:
                                    pass
                                stream_audio_tail = None
                                stream_audio_tail_kind = None
                                stream_audio_tail_source_idx = None
                                stream_audio_tail_ref_path = ""
                                stream_audio_tail_pcm = None
                                stream_audio_tail_audible_samples = 0
                                stream_audio_tail_visible_frames = 0
                                stream_audio_tail_flushed = True
                                _stream_write_progress_marker()

                    next_chunk_pending = bool(os.path.exists(_stream_chunk_path(stream_audio_next_idx)))
                    stream_audio_exhausted = bool(stream_audio_done) and (not bool(next_chunk_pending))

                    if bool(stream_audio_cancelled):
                        return len(stream_audio_clips)

                    # Flush residual tail only once, when producer marked stream done.
                    # IMPORTANT: do not gate this on local queue depth. With async producer,
                    # per-rank queue levels can differ transiently; queue-depth-gated tail
                    # flush can make ranks diverge in clip count and deadlock mid-stream.
                    if stream_audio_exhausted and (not stream_audio_tail_flushed):
                        tail_t = int(stream_audio_tail.shape[-1]) if (stream_audio_tail is not None and stream_audio_tail.ndim >= 4) else 0
                        if tail_t > 0:
                            added_tail = int(
                                _flush_stream_audio_tail_clip(
                                    reason="refill-done",
                                    mark_flushed=True,
                                )
                            )
                            if added_tail > 0:
                                loaded_any = True
                        else:
                            stream_audio_tail_flushed = True
                    if len(stream_audio_clips) >= min_required:
                        return len(stream_audio_clips)
                    if stream_audio_done and next_chunk_pending and len(stream_audio_clips) < int(queue_target):
                        if stream_timing_log and bool(block_until_ready) and len(stream_audio_clips) < int(min_required):
                            print(
                                f"Rank {dist.get_rank()}: liveaudio startup draining pending chunks "
                                f"q={int(len(stream_audio_clips))} need={int(min_required)} next={int(stream_audio_next_idx)}",
                                flush=True,
                            )
                        continue
                    if stream_audio_done:
                        return len(stream_audio_clips)
                    if not block_until_ready:
                        return len(stream_audio_clips)
                    if not loaded_any:
                        time.sleep(float(stream_audio_poll_sec))

        def _stream_async_notify() -> None:
            if stream_audio_queue_cv is None:
                return
            try:
                with stream_audio_queue_cv:
                    stream_audio_queue_cv.notify_all()
            except Exception:
                pass

        def _stream_start_async_producer() -> None:
            nonlocal stream_audio_producer_thread, stream_audio_producer_stop, stream_audio_producer_error
            if (not stream_audio_mode) or (not stream_audio_async_producer):
                return
            if stream_audio_producer_thread is not None:
                return
            stream_audio_producer_stop = threading.Event()
            stream_audio_producer_error = None

            def _producer_main() -> None:
                nonlocal stream_audio_producer_error
                try:
                    while True:
                        if stream_audio_producer_stop is not None and stream_audio_producer_stop.is_set():
                            break
                        qlen_before = int(len(stream_audio_clips))
                        qlen_after = int(
                            _stream_refill_audio_clips(
                                min_required=1,
                                block_until_ready=False,
                            )
                        )
                        queue_target = int(_stream_audio_pending_clip_target())
                        _stream_async_notify()
                        if bool(stream_audio_done) and int(qlen_after) <= 0:
                            break
                        if int(qlen_after) >= int(queue_target):
                            if stream_audio_queue_cv is not None:
                                with stream_audio_queue_cv:
                                    stream_audio_queue_cv.wait(timeout=float(stream_audio_poll_sec))
                            else:
                                time.sleep(float(stream_audio_poll_sec))
                        elif int(qlen_after) <= int(qlen_before):
                            time.sleep(float(stream_audio_poll_sec))
                except Exception as e:
                    stream_audio_producer_error = str(e)
                    print(
                        f"Rank {dist.get_rank()}: liveaudio async producer failed: {e}",
                        flush=True,
                    )
                finally:
                    _stream_async_notify()

            stream_audio_producer_thread = threading.Thread(
                target=_producer_main,
                name=f"liveaudio-producer-r{dist.get_rank()}",
                daemon=True,
            )
            stream_audio_producer_thread.start()

        def _stream_stop_async_producer() -> None:
            nonlocal stream_audio_producer_thread
            if stream_audio_producer_stop is not None:
                try:
                    stream_audio_producer_stop.set()
                except Exception:
                    pass
            _stream_async_notify()
            t = stream_audio_producer_thread
            if t is not None:
                try:
                    t.join()
                except Exception:
                    pass
            stream_audio_producer_thread = None

        def _stream_wait_for_audio_clips(*, min_required: int, block_until_ready: bool) -> int:
            min_required = max(0, int(min_required))
            if (not stream_audio_async_producer) or (stream_audio_queue_cv is None):
                return int(
                    _stream_refill_audio_clips(
                        min_required=int(min_required),
                        block_until_ready=bool(block_until_ready),
                    )
                )
            wait_started = 0.0
            next_wait_log_at = 0.0
            while True:
                qlen_now = int(len(stream_audio_clips))
                if (
                    bool(block_until_ready)
                    and stream_audio_is_encode_rank
                    and qlen_now < int(min_required)
                    and not stream_audio_producer_error
                ):
                    prev_qlen = int(qlen_now)
                    prev_seen = int(stream_audio_seen_chunks)
                    qlen_now = int(
                        _stream_refill_audio_clips(
                            min_required=int(min_required),
                            block_until_ready=False,
                        )
                    )
                    if qlen_now > prev_qlen or int(stream_audio_seen_chunks) > prev_seen:
                        _stream_async_notify()
                if qlen_now >= int(min_required):
                    if stream_timing_log and wait_started > 0.0:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio wait done reason=ready "
                            f"q={int(qlen_now)} need={int(min_required)} "
                            f"waited={float(time.perf_counter() - wait_started):.3f}s",
                            flush=True,
                        )
                    return int(qlen_now)
                if bool(stream_audio_done):
                    if stream_timing_log and wait_started > 0.0:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio wait done reason=done "
                            f"q={int(qlen_now)} need={int(min_required)} "
                            f"waited={float(time.perf_counter() - wait_started):.3f}s",
                            flush=True,
                        )
                    return int(qlen_now)
                if stream_audio_producer_error:
                    if wait_started > 0.0:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio wait done reason=producer_error "
                            f"q={int(qlen_now)} need={int(min_required)} "
                            f"waited={float(time.perf_counter() - wait_started):.3f}s",
                            flush=True,
                        )
                    return int(qlen_now)
                if not bool(block_until_ready):
                    if stream_timing_log and wait_started > 0.0:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio wait done reason=nonblocking "
                            f"q={int(qlen_now)} need={int(min_required)} "
                            f"waited={float(time.perf_counter() - wait_started):.3f}s",
                            flush=True,
                        )
                    return int(qlen_now)
                now_t = float(time.perf_counter())
                if wait_started <= 0.0:
                    wait_started = now_t
                    next_wait_log_at = now_t + 0.5
                elif now_t >= next_wait_log_at:
                    if stream_timing_log:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio wait q={int(qlen_now)} "
                            f"need={int(min_required)} seen_chunks={int(stream_audio_seen_chunks)} "
                            f"done={1 if stream_audio_done else 0} "
                            f"waited={float(now_t - wait_started):.3f}s",
                            flush=True,
                        )
                    next_wait_log_at = now_t + 0.5
                with stream_audio_queue_cv:
                    stream_audio_queue_cv.wait(timeout=float(stream_audio_poll_sec))

        def _stream_try_reply_boundary_prefill(*, min_required: int) -> int:
            target = max(0, int(min_required))
            if (
                int(target) <= 0
                or bool(stream_audio_async_producer)
                or bool(stream_audio_is_always_on)
                or (not stream_audio_is_encode_rank)
                or bool(stream_audio_done)
                or bool(stream_audio_producer_error)
            ):
                return int(len(stream_audio_clips))
            qlen_now = int(
                _stream_refill_audio_clips(
                    min_required=int(target),
                    block_until_ready=False,
                )
            )
            if int(qlen_now) >= int(target):
                return int(qlen_now)
            deadline = float(time.perf_counter()) + float(reply_boundary_prefill_wait_sec())
            sleep_step = float(min(max(float(stream_audio_poll_sec), 0.005), 0.02))
            while (
                int(qlen_now) < int(target)
                and (not bool(stream_audio_done))
                and (not bool(stream_audio_producer_error))
                and float(time.perf_counter()) < float(deadline)
            ):
                time.sleep(float(sleep_step))
                qlen_now = int(
                    _stream_refill_audio_clips(
                        min_required=int(target),
                        block_until_ready=False,
                    )
                )
            return int(qlen_now)

        def _stream_pop_audio_clip() -> tuple[torch.Tensor | None, str, int, str, bytes, int, int, int, int]:
            nonlocal stream_audio_skip_before_chunk_idx
            stream_audio_skip_before_chunk_idx = max(
                int(stream_audio_skip_before_chunk_idx),
                int(_stream_soft_break_chunk_idx()),
            )
            while (
                len(stream_audio_clips) > 0
                and len(stream_audio_clip_source_idxs) > 0
                and int(stream_audio_clip_source_idxs[0]) < int(stream_audio_skip_before_chunk_idx)
            ):
                try:
                    stream_audio_clips.popleft()
                except Exception:
                    break
                try:
                    if len(stream_audio_clip_kinds) > 0:
                        stream_audio_clip_kinds.popleft()
                except Exception:
                    pass
                try:
                    stream_audio_clip_source_idxs.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_ref_paths) > 0:
                        stream_audio_clip_ref_paths.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_pcms) > 0:
                        stream_audio_clip_pcms.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_sample_rates) > 0:
                        stream_audio_clip_sample_rates.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_audible_samples) > 0:
                        stream_audio_clip_audible_samples.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_visible_start_frames) > 0:
                        stream_audio_clip_visible_start_frames.popleft()
                except Exception:
                    pass
                try:
                    if len(stream_audio_clip_visible_frames) > 0:
                        stream_audio_clip_visible_frames.popleft()
                except Exception:
                    pass
            if len(stream_audio_clips) <= 0:
                return None, "speech", 0, "", b"", int(stream_audio_tail_sample_rate), 0, 0, 0
            clip = stream_audio_clips.popleft()
            clip_kind = normalize_stream_clip_kind(
                stream_audio_clip_kinds.popleft() if len(stream_audio_clip_kinds) > 0 else "speech"
            )
            clip_source_idx = int(
                max(0, int(stream_audio_clip_source_idxs.popleft() if len(stream_audio_clip_source_idxs) > 0 else 0))
            )
            clip_ref_path = str(
                stream_audio_clip_ref_paths.popleft() if len(stream_audio_clip_ref_paths) > 0 else ""
            ).strip()
            clip_pcm = bytes(stream_audio_clip_pcms.popleft() if len(stream_audio_clip_pcms) > 0 else b"")
            clip_sample_rate = int(
                max(1, int(stream_audio_clip_sample_rates.popleft() if len(stream_audio_clip_sample_rates) > 0 else stream_audio_tail_sample_rate))
            )
            clip_audible_samples = int(
                max(0, int(stream_audio_clip_audible_samples.popleft() if len(stream_audio_clip_audible_samples) > 0 else 0))
            )
            try:
                clip_visible_start_frames = int(
                    stream_audio_clip_visible_start_frames.popleft()
                    if len(stream_audio_clip_visible_start_frames) > 0
                    else 0
                )
            except Exception:
                clip_visible_start_frames = 0
            try:
                clip_visible_frames = int(
                    stream_audio_clip_visible_frames.popleft()
                    if len(stream_audio_clip_visible_frames) > 0
                    else int(clip.shape[-1])
                )
            except Exception:
                clip_visible_frames = int(clip.shape[-1]) if clip is not None and clip.ndim >= 4 else 0
            clip_visible_frames = int(
                max(
                    0,
                    min(
                        int(clip_visible_frames),
                        int(clip.shape[-1]) if clip is not None and clip.ndim >= 4 else int(clip_visible_frames),
                    ),
                )
            )
            clip_visible_start_frames = int(
                max(
                    0,
                    min(
                        int(clip_visible_start_frames),
                        int(clip.shape[-1]) if clip is not None and clip.ndim >= 4 else int(clip_visible_start_frames),
                    ),
                )
            )
            if clip is not None and clip.ndim >= 4:
                clip_visible_frames = int(
                    max(
                        0,
                        min(
                            int(clip_visible_frames),
                            int(clip.shape[-1]) - int(clip_visible_start_frames),
                        ),
                    )
                )
            _stream_write_progress_marker()
            _stream_async_notify()
            return (
                clip,
                clip_kind,
                clip_source_idx,
                clip_ref_path,
                clip_pcm,
                clip_sample_rate,
                clip_audible_samples,
                clip_visible_start_frames,
                clip_visible_frames,
            )

        if stream_audio_mode:
            if stream_timing_log:
                print(
                    f"Rank {dist.get_rank()}: liveaudio mode enabled dir={stream_audio_dir}",
                    flush=True,
                )
            if stream_timing_log and stream_audio_distributed_clip_broadcast:
                print(
                    f"Rank {dist.get_rank()}: liveaudio distributed broadcast "
                    f"source_rank={int(stream_audio_encode_rank)} local_encode={1 if stream_audio_is_encode_rank else 0}",
                    flush=True,
                )
            if stream_audio_is_encode_rank and bool(stream_audio_is_always_on):
                stream_audio_silence_clip = _build_stream_fill_clip(mode="silence")
            else:
                stream_audio_silence_clip = None
            if stream_audio_is_encode_rank and str(stream_audio_tail_fill_mode) in {"noise", "smooth_noise"}:
                stream_audio_tail_fill_clip = _build_stream_fill_clip(
                    mode=str(stream_audio_tail_fill_mode),
                    seed_offset=97,
                )
            else:
                stream_audio_tail_fill_clip = None
            if stream_audio_is_encode_rank:
                # Keep immediate-silence startup only for always-on streams.
                # For regular reply streams, first try to wait a short time for real chunk #1;
                # otherwise the first silent warm clip can add a full denoise cycle of latency.
                if (
                    stream_audio_silence_clip is not None
                    and bool(stream_audio_immediate_silence)
                    and bool(stream_audio_is_always_on)
                ):
                    startup_wait_t0 = time.perf_counter()
                    startup_q = int(
                        _stream_refill_audio_clips(
                            min_required=1,
                            block_until_ready=False,
                        )
                    )
                    startup_wait_dt = float(time.perf_counter() - startup_wait_t0)
                    if stream_timing_log:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio startup wait done "
                            f"mode=always_on need=1 q={int(startup_q)} dt={startup_wait_dt:.3f}s "
                            f"seen_chunks={int(stream_audio_seen_chunks)} done={1 if stream_audio_done else 0}",
                            flush=True,
                        )
                        print(
                            f"Rank {dist.get_rank()}: liveaudio immediate-silence startup enabled (always-on)",
                            flush=True,
                        )
                else:
                    first_wait_min_clips = 1 if bool(stream_audio_is_warmup) else int(stream_audio_reply_start_min_clips)
                    # Startup reply prefill is intentionally config-driven. After fixing
                    # producer progress tracking and startup underfill, a single ready clip
                    # is now a valid latency-first mode for regular reply streams.
                    startup_wait_t0 = time.perf_counter()
                    startup_q = int(
                        _stream_refill_audio_clips(
                            min_required=int(first_wait_min_clips),
                            block_until_ready=True,
                        )
                    )
                    startup_wait_dt = float(time.perf_counter() - startup_wait_t0)
                    if stream_timing_log:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio reply startup clips={len(stream_audio_clips)} "
                            f"target={int(first_wait_min_clips)} q={int(startup_q)} "
                            f"dt={startup_wait_dt:.3f}s seen_chunks={int(stream_audio_seen_chunks)} "
                            f"done={1 if stream_audio_done else 0}",
                            flush=True,
                        )
                    if (
                        (not stream_audio_async_producer)
                        and stream_audio_is_encode_rank
                        and (not bool(stream_audio_is_always_on))
                    ):
                        # Regular reply streams use synchronous clip-boundary prefetch.
                        # Try to preload one extra clip before the first denoise block so
                        # clip-2 can usually start without waiting, while keeping all audio
                        # encode work out of the hot denoise loop.
                        startup_prefill_cap = max(2, int(first_wait_min_clips) + 1)
                        prev_pending_cap = int(stream_audio_max_pending_clips)
                        try:
                            stream_audio_max_pending_clips = min(
                                int(prev_pending_cap),
                                int(startup_prefill_cap),
                            )
                            _stream_try_reply_boundary_prefill(
                                min_required=min(
                                    int(stream_audio_max_pending_clips),
                                    int(startup_prefill_cap),
                                )
                            )
                        finally:
                            stream_audio_max_pending_clips = int(prev_pending_cap)
                        print(
                            f"Rank {dist.get_rank()}: liveaudio boundary prefill "
                            f"q={int(len(stream_audio_clips))} target={int(startup_prefill_cap)}",
                            flush=True,
                        )
                if stream_audio_producer_error:
                    if stream_audio_distributed_clip_broadcast:
                        stream_audio_startup_error = (
                            f"liveaudio producer failed before startup: {stream_audio_producer_error}"
                        )
                        stream_audio_done = True
                    else:
                        raise RuntimeError(f"liveaudio producer failed before startup: {stream_audio_producer_error}")
                if len(stream_audio_clips) <= 0 and stream_audio_done and stream_audio_silence_clip is None:
                    if int(stream_audio_done_chunks_total) <= 0:
                        print(
                            f"Rank {dist.get_rank()}: liveaudio startup ended before first chunk: "
                            f"status={str(stream_audio_done_status or 'ok')} dir={stream_audio_dir}",
                            flush=True,
                        )
                        if not stream_audio_distributed_clip_broadcast:
                            return None, dataset_info
                    elif not stream_audio_distributed_clip_broadcast:
                        raise RuntimeError(f"liveaudio queue finished without chunks: {stream_audio_dir}")
                if len(stream_audio_clips) <= 0 and (not bool(stream_audio_done)) and (not bool(stream_audio_is_always_on)):
                    if stream_audio_distributed_clip_broadcast:
                        stream_audio_startup_error = f"liveaudio startup has no clips yet: {stream_audio_dir}"
                        stream_audio_done = True
                    else:
                        raise RuntimeError(f"liveaudio startup has no clips yet: {stream_audio_dir}")
            if not bool(stream_audio_defer_async_start):
                _stream_start_async_producer()
            nr = _liveaudio_stream_total_clips(max_repeat=int(max_repeat))
            audio_emb = None
        elif isinstance(audio_encode_path, str) and ('+' in audio_encode_path):
            audio_paths = audio_encode_path.split('+')
            audio_embs = []
            nr_list = []
            
            for path in audio_paths:
                audio_emb_i, nr_i = self.encode_audio(path, infer_frames=infer_frames)
                audio_embs.append(audio_emb_i)
                nr_list.append(nr_i)
            
            min_frames = min(emb.shape[-1] for emb in audio_embs)
            audio_embs = [emb[..., :min_frames] for emb in audio_embs]
            nr = min(nr_list)
            audio_emb = torch.cat(audio_embs, dim=0)

            # Process SAM2 and generate routing_logits if video path is provided
            print(f"rank {dist.get_rank()} processing SAM2")
            input_video_for_sam2 = input_video_for_sam2 if input_video_for_sam2 is not None else ref_image_path
            routing_logits = None
            rank = dist.get_rank()
            
            # Broadcast video path to all ranks
            if rank == 0:
                video_path_bytes = input_video_for_sam2.encode('utf-8')
                path_length = torch.tensor([len(video_path_bytes)], dtype=torch.long, device=self.device)
            else:
                path_length = torch.tensor([0], dtype=torch.long, device=self.device)
            dist.broadcast(path_length, src=0)
            if rank == 0:
                path_tensor = torch.ByteTensor(list(video_path_bytes)).to(self.device)
            else:
                path_tensor = torch.zeros(path_length.item(), dtype=torch.uint8, device=self.device)
            dist.broadcast(path_tensor, src=0)
            video_path = path_tensor.cpu().numpy().tobytes().decode('utf-8')
            print(f"Rank {rank}: video_path: {video_path}")
            
            parent_dir = os.path.dirname(video_path)
            sam2_output_base = parent_dir  
            
            if rank == 0:
                sam2_cmd = [
                    "python",
                    "liveavatar/utils/router/sam2_tools.py",
                    "--video_folder", video_path,
                    "--output_path", sam2_output_base
                ]
                try:
                    subprocess.run(sam2_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Rank {rank}: SAM2 processing failed: {e}")
                    raise e
                dist.barrier()
            else:
                dist.barrier()
            
            base_name = os.path.basename(video_path).split(".")[0]
            tracking_mask_results_dir = os.path.join(
                sam2_output_base,
                base_name,
                "tracking_mask_results"
            )
            print(f"Rank {rank}: Looking for masks in: {tracking_mask_results_dir}")
            
            target_shape = (1, infer_frames // 4, HEIGHT // 8, WIDTH // 8)
            routing_logits = process_masks_to_routing_logits(
                tracking_mask_results_dir,
                shape=target_shape
            )
            num_actors = routing_logits.shape[-1]
            routing_logits = routing_logits.reshape(1, infer_frames // 4, HEIGHT // 8 // 2, WIDTH // 8 // 2, num_actors)
            routing_logits = routing_logits.to(device=self.device, dtype=self.param_dtype)
            mask = routing_logits.permute(4,1,2,3,0)  # [num_actors, t, h, w, 1]

            # 按比例进行二维空间膨胀（允许重叠）
            def dilate_mask_by_ratio(mask_tensor: torch.Tensor, ratio: float = 0.3, thr: float = 0.5) -> torch.Tensor:
                # mask_tensor: [A, T, H, W, 1]，值域 [0,1]
                A, T, H, W, _ = mask_tensor.shape
                out = torch.zeros_like(mask_tensor)
                bin_mask = (mask_tensor > thr).to(dtype=mask_tensor.dtype)
                for a in range(A):
                    for t in range(T):
                        m2d = bin_mask[a, t, :, :, 0]
                        if m2d.any():
                            ys, xs = torch.where(m2d)
                            box_h = int(ys.max() - ys.min() + 1)
                            box_w = int(xs.max() - xs.min() + 1)
                            radius = max(1, int((ratio * max(box_h, box_w) + 0.9999)))
                            k = 2 * radius + 1
                            x = m2d[None, None, :, :]
                            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
                            out[a, t, :, :, 0] = (x[0, 0] > 0).to(mask_tensor.dtype)
                        else:
                            out[a, t, :, :, 0] = m2d
                return out

            mask = dilate_mask_by_ratio(mask, ratio=0.1, thr=0.5)
            mask_bool = mask > 0.5
            total_count = mask_bool.sum(dim=0, keepdim=True)  # [1, t, h, w, 1], 统计该位置有多少角色为1
            others_present = (total_count - mask_bool.to(total_count.dtype)) > 0  # [num_actors, t, h, w, 1]
            mask = (~others_present).to(dtype=mask.dtype)
            m=(mask[0][0].detach().to(torch.float16).cpu().numpy()>0.5).astype(np.uint8)*255; Image.fromarray(m.squeeze()).save("tmp/mask/mask.png")
        else:
            audio_emb, nr = self.encode_audio(audio_encode_path, infer_frames=infer_frames)

        
        # In liveaudio stream mode (especially with async producer), keep audio encoder
        # on device while stream is active; moving it to CPU here breaks subsequent
        # chunk encodes and causes early video stop/freeze.
        if not stream_audio_mode:
            self.audio_encoder.model.to("cpu")
        profile_audio_s = float(time.perf_counter() - profile_audio_t0)
        if stream_audio_mode:
            num_repeat = max(1, int(nr))
        else:
            nr_eff = max(1, int(nr))
            req_repeat = nr_eff if num_repeat is None else max(1, int(num_repeat))
            if req_repeat > nr_eff:
                # Do not clamp requested clip count down to encoder-derived nr.
                # Pad audio embeddings with silence so the generator can cover
                # the full requested tail (prevents early visual freeze before
                # audio playback ends).
                target_t = int(req_repeat) * int(infer_frames)
                try:
                    cur_t = int(audio_emb.shape[-1])
                except Exception:
                    cur_t = 0
                if target_t > cur_t:
                    pad_t = int(target_t - cur_t)
                    pad_shape = list(audio_emb.shape)
                    pad_shape[-1] = int(pad_t)
                    pad = torch.zeros(
                        pad_shape,
                        dtype=audio_emb.dtype,
                        device=audio_emb.device,
                    )
                    audio_emb = torch.cat([audio_emb, pad], dim=-1)
            num_repeat = int(req_repeat)

        lat_motion_frames = (self.motion_frames + 3) // 4
        drop_first_motion = False
        static_cond_t0 = time.perf_counter()
        static_cond_hit = False
        if pose_video is None:
            static_cond, static_cond_hit = self._get_static_reply_condition(
                ref_image_path=ref_image_path,
                size=size,
                infer_frames=infer_frames,
                drop_motion_noisy=bool(drop_motion_noisy),
            )
            ref_latents = static_cond["ref_latents"]
            motion_latents = static_cond["motion_latents"]
            videos_last_frames = static_cond.get("motion_frames_pixels", motion_latents).detach()
            if bool(drop_motion_noisy):
                zero_motion_latents = static_cond["zero_motion_latents"]
            COND = [static_cond["cond_zero"]]
        else:
            tensor_trans = transforms.ToTensor()
            ref_image = ImageOps.exif_transpose(Image.open(ref_image_path)).convert("RGB")
            model_pic = self._resize_cover_crop_pil(ref_image, int(HEIGHT), int(WIDTH))
            ref_pixel_values = tensor_trans(model_pic)
            ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
            ref_pixel_values = ref_pixel_values.to(
                dtype=self.vae.dtype, device=self.vae.device
            )
            ref_latents = torch.stack(self.vae.encode(ref_pixel_values))
            motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
            videos_last_frames = motion_latents.detach()
            motion_latents = torch.stack(self.vae.encode(motion_latents))
            if bool(drop_motion_noisy):
                zero_motion_latents = torch.zeros_like(motion_latents)
            COND = self.load_pose_cond(
                pose_video=pose_video,
                num_repeat=num_repeat,
                infer_frames=infer_frames,
                size=size,
            )
        profile_static_cond_s = float(time.perf_counter() - static_cond_t0)
        stream_current_ref_image_path = os.path.abspath(str(ref_image_path or "")) if ref_image_path else ""
        if rank == 0 and stream_audio_mode:
            print(
                f"TPP static-cond-cache rank=0 mode={'hit' if static_cond_hit else 'miss'} "
                f"dt={float(profile_static_cond_s):.3f}s pose={1 if pose_video else 0}",
                flush=True,
            )

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # process prompt
        profile_prompt_t0 = time.perf_counter()
        context, context_null = self.encode_prompt(input_prompt, n_prompt, offload_model) #list(1):[len,4096]
        idle_context = None
        idle_context_null = None
        stream_visual_context_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        base_prompt_norm = str(input_prompt or "").strip()
        base_negative_norm = str(n_prompt or "").strip()
        try:
            guide_scale_value = float(guide_scale)
        except Exception:
            guide_scale_value = 0.0
        cfg_mode = str(os.getenv("LIVE_STREAM_TPP_CFG_MODE", "off") or "off").strip().lower()
        cfg_mode_enabled = cfg_mode not in {"", "0", "off", "false", "none", "disabled"}
        tpp_cfg_enabled = bool(cfg_mode_enabled and guide_scale_value > 1.0 and context_null is not None)
        if rank == 0 and stream_timing_log:
            print(
                f"TPP CFG mode={cfg_mode or 'off'} enabled={1 if tpp_cfg_enabled else 0} "
                f"guide={guide_scale_value:.3f} batch={'2' if tpp_cfg_enabled else '1'}",
                flush=True,
            )
        if stream_audio_mode and bool(stream_audio_prompt_switch):
            idle_prompt_text = build_stream_idle_prompt_text(
                str(input_prompt or ""),
                idle_prompt=str(idle_prompt or ""),
            )
            idle_prompt_norm = str(idle_prompt_text).strip()
            input_prompt_norm = str(input_prompt or "").strip()
            if idle_prompt_norm and idle_prompt_norm != input_prompt_norm:
                idle_context, idle_context_null = self.encode_prompt(idle_prompt_text, n_prompt, offload_model)
                if rank == 0 and stream_timing_log:
                    print(
                        "TPP liveaudio idle prompt prepared: clip-boundary switch enabled",
                        flush=True,
                    )
            elif idle_prompt_norm:
                idle_context, idle_context_null = context, context_null
                if rank == 0 and stream_timing_log:
                    print(
                        "TPP liveaudio idle prompt prepared: base prompt already idle",
                        flush=True,
                    )
            else:
                stream_audio_prompt_switch = False
        profile_prompt_s = float(time.perf_counter() - profile_prompt_t0)

        def _stream_visual_context(
            prompt_text: str,
            negative_text: str,
        ) -> tuple[Any, Any, str]:
            prompt_s = str(prompt_text or "").strip() or str(base_prompt_norm)
            negative_s = str(negative_text or "").strip() or str(base_negative_norm)
            if prompt_s == str(base_prompt_norm) and negative_s == str(base_negative_norm):
                return context, context_null, "speech"
            key = (str(prompt_s), str(negative_s))
            cached = stream_visual_context_cache.get(key)
            if cached is None:
                enc_t0 = time.perf_counter()
                cached = self.encode_prompt(str(prompt_s), str(negative_s), offload_model)
                stream_visual_context_cache[key] = cached
                if rank == 0:
                    print(
                        f"TPP liveaudio visual prompt prepared: chars={len(prompt_s)} "
                        f"neg_chars={len(negative_s)} dt={float(time.perf_counter() - enc_t0):.3f}s",
                        flush=True,
                    )
            return cached[0], cached[1], f"visual:{abs(hash(key)) % 100000000}"

        dataset_info = {}

        if (not stream_audio_mode) or stream_timing_log:
            print("complete prepare conditional inputs")
        profile_scheduler_comm_t0 = time.perf_counter()
        if sample_solver == 'euler':#default
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=float(shift))
        else:
            raise NotImplementedError("Unsupported solver.")
        self._initialize_comm_group(num_gpus_dit=num_gpus_dit, enable_vae_parallel=enable_vae_parallel)
        world_size = dist.get_world_size()
        expected_world_size = num_gpus_dit + (1 if enable_vae_parallel else 0)
        assert world_size == expected_world_size, (
            f"Invalid distributed setup: got WORLD_SIZE={world_size}, "
            f"but expected {expected_world_size} for num_gpus_dit={num_gpus_dit} "
            f"and enable_vae_parallel={enable_vae_parallel}."
        )

        in_dit_device = rank < num_gpus_dit
        decode_rank = num_gpus_dit if enable_vae_parallel else (num_gpus_dit - 1)

        def _cfg_context_batch(pos_ctx, null_ctx, enabled: bool):
            pos_list = list(pos_ctx or [])
            if not bool(enabled):
                return pos_list[0:1]
            null_list = list(null_ctx or [])
            if len(pos_list) == 0 or len(null_list) == 0:
                return pos_list[0:1]
            return null_list[0:1] + pos_list[0:1]

        def _cfg_batch_tensor(tensor, enabled: bool, *, zero_uncond: bool = False):
            if not bool(enabled):
                return tensor
            if not torch.is_tensor(tensor):
                return tensor
            if tensor.dim() == 0:
                return tensor
            first = tensor[0:1]
            uncond = torch.zeros_like(first) if bool(zero_uncond) else first
            return torch.cat([uncond, first], dim=0)

        def _cfg_latent_inputs(latent_tensor, enabled: bool):
            if not bool(enabled):
                return [latent_tensor]
            return [latent_tensor, latent_tensor]

        def _cfg_timestep_batch(timestep_tensor, enabled: bool):
            if not bool(enabled):
                return timestep_tensor
            if not torch.is_tensor(timestep_tensor) or timestep_tensor.dim() == 0:
                return timestep_tensor
            return timestep_tensor[0:1].repeat(2, *([1] * (timestep_tensor.dim() - 1)))

        def _noise_output_batch_tensor(output):
            if torch.is_tensor(output):
                return output
            if isinstance(output, (list, tuple)) and len(output) > 0:
                parts = list(output)
                if all(torch.is_tensor(item) for item in parts):
                    if parts[0].dim() == 4:
                        return torch.stack(parts, dim=0)
                    return torch.cat(parts, dim=0)
            raise RuntimeError(f"Unexpected noise_model output type for TPP CFG: {type(output)!r}")

        if bool(getattr(self, "joint_sp_denoise", False)):
            print(
                f"TPP sequence-parallel joint denoise enabled: sp_size={int(self.sp_size)} "
                f"num_gpus_dit={int(num_gpus_dit)} stage_p2p=0",
                flush=True,
            )
        # Avoid NCCL collective barrier hangs in server environments.
        self._safe_barrier()  # wait all ranks to finish initialization
        profile_scheduler_comm_s = float(time.perf_counter() - profile_scheduler_comm_t0)

        if bool(live_raw_dir) and int(rank) == int(decode_rank):
            self._prewarm_live_post_vae_face_restore(
                height=int(HEIGHT),
                width=int(WIDTH),
            )

        # Optional: start a live HLS (fMP4) stream writer on the decode rank.
        # This allows watching generation progress in a browser over plain HTTPS (Cloudflare Tunnel).
        hls_proc = None
        hls_log_f = None
        raw_fd = None
        raw_pipe_path = None
        raw_transport_mode = "pipe"
        raw_shm = None
        raw_shm_name = None
        raw_shm_frame_capacity = 0
        raw_shm_header_bytes = int(LIVE_RAW_SHM_HEADER_BYTES)
        raw_ready_path = None
        raw_done_path = None
        raw_done_json_path = None
        raw_progress_path = None
        raw_progress_json_enabled = True
        raw_first_frame_marked = False
        raw_pipe_size_set = False
        raw_backlog_chunks = deque()
        raw_backlog_cv = threading.Condition()
        raw_writer_thread: threading.Thread | None = None
        raw_writer_stop: threading.Event | None = None
        raw_writer_error: str | None = None
        raw_copy_stream = None
        raw_host_tensor_pool: dict[tuple[tuple[int, ...], str], deque[torch.Tensor]] = {}
        try:
            raw_host_tensor_pool_limit = int(os.getenv("LIVE_RAW_PINNED_POOL_SIZE", "4") or 4)
        except Exception:
            raw_host_tensor_pool_limit = 4
        raw_host_tensor_pool_limit = max(0, min(16, int(raw_host_tensor_pool_limit)))
        raw_backlog_bytes = 0
        raw_backlog_warned = False
        raw_frames_streamed = 0
        raw_frames_enqueued = 0
        raw_progress_last_write_ts = 0.0
        raw_progress_last_snapshot = None
        raw_prompt_mode = "speech"
        raw_prompt_mode_seq = 0
        raw_prompt_mode_start_frame = 0
        raw_source_chunk_idx = 0
        raw_source_chunk_start_frame = 0
        remote_edge_sender = None
        remote_edge_enabled = False
        remote_edge_failed = False
        remote_edge_mode = ""
        remote_edge_output = ""
        remote_edge_file_target_frames = 0
        remote_edge_file_result: dict[str, Any] | None = None
        remote_edge_static_audio_sent = False
        remote_edge_poster_sent = False
        remote_edge_audio_sent_chunks: set[int] = set()
        remote_edge_audio_next_chunk_idx = 1
        remote_edge_video_frames_sent = 0
        remote_edge_first_ts_us = int(time.monotonic() * 1_000_000.0)
        remote_edge_stats_t0 = time.perf_counter()
        remote_edge_stats_last_ts = float(remote_edge_stats_t0)
        remote_edge_stats_last_frames = 0
        remote_edge_stats_last_audio_chunks = 0
        remote_edge_last_send_dt = 0.0
        remote_edge_last_pace_sleep_dt = 0.0
        remote_edge_pace_total_sleep_sec = 0.0
        remote_edge_pace_started_at = 0.0
        remote_edge_pace_last_log = 0.0
        remote_edge_last_block_dt = 0.0
        remote_edge_last_denoise_dt = 0.0
        remote_edge_last_recv_dt = 0.0
        remote_edge_last_send_wait_dt = 0.0
        remote_edge_last_clip_frames = 0
        remote_edge_last_num_blocks = 0
        remote_edge_last_kv_cache_size = 0
        remote_edge_last_max_seq_len = 0
        remote_edge_last_kv_cap_frames = 0
        try:
            remote_edge_stats_interval_sec = float(
                os.getenv("REMOTE_EDGE_PRODUCER_STATS_INTERVAL_SEC", "0") or 0
            )
        except Exception:
            remote_edge_stats_interval_sec = 0.0
        remote_edge_stats_interval_sec = max(0.0, min(60.0, float(remote_edge_stats_interval_sec)))
        try:
            remote_edge_producer_max_lead_sec = float(
                os.getenv("REMOTE_EDGE_PRODUCER_MAX_LEAD_SEC", "45.0") or 45.0
            )
        except Exception:
            remote_edge_producer_max_lead_sec = 45.0
        remote_edge_producer_max_lead_sec = max(0.0, min(600.0, float(remote_edge_producer_max_lead_sec)))
        try:
            remote_edge_producer_prebuffer_frames = int(
                os.getenv(
                    "REMOTE_EDGE_PRODUCER_PREBUFFER_FRAMES",
                    os.getenv("REMOTE_EDGE_START_PREBUFFER_FRAMES", "360") or 360,
                )
                or 360
            )
        except Exception:
            remote_edge_producer_prebuffer_frames = 360
        remote_edge_producer_prebuffer_frames = max(0, min(60 * max(1, int(self.fps)), int(remote_edge_producer_prebuffer_frames)))
        try:
            remote_edge_producer_pace_sleep_max_sec = float(
                os.getenv("REMOTE_EDGE_PRODUCER_PACE_SLEEP_MAX_SEC", "0.25") or 0.25
            )
        except Exception:
            remote_edge_producer_pace_sleep_max_sec = 0.25
        remote_edge_producer_pace_sleep_max_sec = max(0.01, min(2.0, float(remote_edge_producer_pace_sleep_max_sec)))
        try:
            tpp_stage_stats_interval_sec = float(
                os.getenv("LIVE_AUDIO_TPP_STAGE_STATS_INTERVAL_SEC", "0") or 0
            )
        except Exception:
            tpp_stage_stats_interval_sec = 0.0
        tpp_stage_stats_interval_sec = max(0.0, min(60.0, float(tpp_stage_stats_interval_sec)))
        tpp_stage_stats_t0 = time.perf_counter()
        tpp_stage_stats_last_ts = float(tpp_stage_stats_t0)
        tpp_stage_stats_blocks = 0
        tpp_stage_stats_last_blocks = 0
        tpp_stage_stats_last_core_dt = 0.0
        tpp_stage_stats_last_recv_dt = 0.0
        tpp_stage_stats_last_denoise_dt = 0.0
        tpp_stage_stats_last_send_dt = 0.0
        tpp_stage_stats_last_steps = 0
        stream_file_path = os.path.abspath(str(stream_file_output_path or "").strip()) if str(stream_file_output_path or "").strip() else ""
        stream_file_output_w = int(max(0, int(stream_file_output_width or 0)))
        stream_file_output_h = int(max(0, int(stream_file_output_height or 0)))
        stream_file_fps = float(stream_file_output_fps or 0.0)
        if float(stream_file_fps) <= 0.0:
            stream_file_fps = float(self.fps)
        stream_file_trim_sec = float(max(0.0, float(stream_file_trim_duration_sec or 0.0)))
        stream_file_interpolation_mode = str(stream_file_interpolation or "").strip().lower()
        if stream_file_interpolation_mode in {"none", "off", "0", "false", "no"}:
            stream_file_interpolation_mode = ""
        stream_file_enabled = bool(stream_file_path) and int(rank) == int(decode_rank)
        stream_file_queue: deque[dict[str, Any]] = deque()
        stream_file_cv = threading.Condition()
        stream_file_stop = threading.Event()
        stream_file_thread: threading.Thread | None = None
        stream_file_error: str | None = None
        stream_file_frames_in = 0
        stream_file_frames_out = 0
        stream_file_blocks = 0
        stream_file_enqueue_s = 0.0
        stream_file_rife_s = 0.0
        stream_file_resize_s = 0.0
        stream_file_pack_s = 0.0
        stream_file_write_s = 0.0
        stream_file_started_s = 0.0
        stream_file_finished_s = 0.0
        stream_file_last_frame: torch.Tensor | None = None
        stream_file_progress_path = f"{stream_file_path}.progress.json" if bool(stream_file_path) else ""
        raw_frame_bytes = max(1, int(WIDTH) * int(HEIGHT) * 3)
        try:
            raw_use_shm = str(os.getenv("LIVE_RAW_USE_SHM", "1") or "1").strip().lower() not in ("0", "false", "no", "off", "")
        except Exception:
            raw_use_shm = True
        try:
            raw_shm_target_bytes = int(os.getenv("LIVE_RAW_SHM_BYTES", str(256 * 1024 * 1024)) or (256 * 1024 * 1024))
        except Exception:
            raw_shm_target_bytes = 256 * 1024 * 1024
        raw_shm_target_bytes = max(int(raw_frame_bytes * 64), min(2 * 1024 * 1024 * 1024, int(raw_shm_target_bytes)))
        raw_shm_frame_capacity = max(64, int(raw_shm_target_bytes // max(1, int(raw_frame_bytes))))
        raw_shm_nbytes = int(
            live_raw_shm_total_bytes(
                frame_bytes=int(raw_frame_bytes),
                frame_capacity=int(raw_shm_frame_capacity),
            )
        )
        try:
            raw_pipe_target_bytes = int(os.getenv("LIVE_RAW_PIPE_SIZE_BYTES", "8388608") or 8388608)
        except Exception:
            raw_pipe_target_bytes = 8388608
        raw_pipe_target_bytes = max(0, min(64 * 1024 * 1024, int(raw_pipe_target_bytes)))
        try:
            raw_backlog_max_bytes = int(os.getenv("LIVE_RAW_BACKLOG_MAX_BYTES", str(512 * 1024 * 1024)) or (512 * 1024 * 1024))
        except Exception:
            raw_backlog_max_bytes = 512 * 1024 * 1024
        raw_backlog_max_bytes = max(0, min(4 * 1024 * 1024 * 1024, int(raw_backlog_max_bytes)))
        if live_hls_dir and rank == decode_rank:
            try:
                os.makedirs(live_hls_dir, exist_ok=True)
                audio_for_hls = (
                    audio_path.split("+")[0]
                    if isinstance(audio_path, str) and "+" in audio_path
                    else audio_path
                )
                if audio_for_hls and os.path.exists(audio_for_hls):
                    log_path = os.path.join(live_hls_dir, "ffmpeg.log")
                    hls_log_f = open(log_path, "w", encoding="utf-8")

                    seg_pattern = os.path.join(live_hls_dir, "seg_%05d.m4s")
                    playlist_path = os.path.join(live_hls_dir, "index.m3u8")

                    fps = int(self.fps)
                    # HLS segment duration for live preview.
                    #
                    # NOTE: With ffmpeg's HLS muxer, using sub-second segments can result in
                    # `#EXT-X-TARGETDURATION:0` in the playlist, which breaks some players
                    # and can cause hls.js to jump to the tail or freeze. Keep it >= 1s.
                    try:
                        hls_time = float(os.getenv("LIVE_PREVIEW_HLS_TIME", "1.0") or 1.0)
                    except Exception:
                        hls_time = 1.0
                    hls_time = float(min(max(hls_time, 1.00), 4.00))
                    try:
                        hls_init_time = float(
                            os.getenv("LIVE_PREVIEW_HLS_INIT_TIME", str(min(0.35, hls_time))) or min(0.35, hls_time)
                        )
                    except Exception:
                        hls_init_time = min(0.35, hls_time)
                    hls_init_time = float(min(max(hls_init_time, 0.10), hls_time))
                    gop = max(1, int(math.ceil(float(fps) * float(hls_time))))
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "warning",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{WIDTH}x{HEIGHT}",
                        "-r",
                        str(fps),
                        "-thread_queue_size",
                        "512",
                        "-i",
                        "pipe:0",
                        "-thread_queue_size",
                        "512",
                        "-i",
                        audio_for_hls,
                        # The preview video can end slightly after the audio (we intentionally pad num_clip
                        # to avoid truncation). Pad audio with silence so ffmpeg doesn't terminate early
                        # and close the rawvideo pipe (broken pipe).
                        "-af",
                        "apad",
                        "-shortest",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-tune",
                        "zerolatency",
                        "-pix_fmt",
                        "yuv420p",
                        "-g",
                        str(gop),
                        "-keyint_min",
                        str(gop),
                        "-sc_threshold",
                        "0",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-ar",
                        "48000",
                        "-ac",
                        "1",
                        "-f",
                        "hls",
                        "-hls_init_time",
                        str(hls_init_time),
                        "-hls_time",
                        str(hls_time),
                        # Keep all segments listed (EVENT playlist that grows). This avoids
                        # hls.js treating the preview as a sliding live window and "skipping"
                        # older segments to catch up.
                        "-hls_list_size",
                        "0",
                        # Ensure playlist/segments are updated atomically enough for HTTP serving:
                        # - append_list: avoid rewriting the whole playlist on each segment
                        # - temp_file: write segments as temp files and rename when complete
                        # - program_date_time: improves debugging / player behavior in some browsers
                        "-hls_flags",
                        "append_list+temp_file+program_date_time",
                        "-hls_playlist_type",
                        "event",
                        "-hls_segment_type",
                        "fmp4",
                        "-hls_fmp4_init_filename",
                        "init.mp4",
                        "-hls_segment_filename",
                        seg_pattern,
                        playlist_path,
                    ]
                    hls_proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=hls_log_f,
                    )
                    print(
                        f"Rank {rank}: live HLS started at {playlist_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"Rank {rank}: live_hls_dir set but audio not found: {audio_for_hls}",
                        flush=True,
                    )
            except Exception as e:
                print(f"Rank {rank}: failed to start live HLS: {e}", flush=True)
                try:
                    if hls_log_f is not None:
                        hls_log_f.close()
                except Exception:
                    pass
                hls_proc = None
                hls_log_f = None

        # Optional: start raw RGB24 live stream (FIFO + metadata) on decode rank.
        # This bypasses HLS mux/demux and allows direct frame push to LiveKit worker.
        if live_raw_dir and rank == decode_rank:
            try:
                os.makedirs(live_raw_dir, exist_ok=True)
                raw_pipe_path = os.path.join(live_raw_dir, "frames.rgb")
                meta_path = os.path.join(live_raw_dir, "stream.json")
                raw_ready_path = os.path.join(live_raw_dir, ".first_frame_ready")
                raw_done_path = os.path.join(live_raw_dir, ".done")
                raw_done_json_path = os.path.join(live_raw_dir, "done.json")
                raw_progress_path = os.path.join(live_raw_dir, "progress.json")
                try:
                    if os.path.exists(raw_pipe_path):
                        os.remove(raw_pipe_path)
                except Exception:
                    pass
                try:
                    if raw_ready_path and os.path.exists(raw_ready_path):
                        os.remove(raw_ready_path)
                except Exception:
                    pass
                try:
                    if raw_done_path and os.path.exists(raw_done_path):
                        os.remove(raw_done_path)
                except Exception:
                    pass
                try:
                    if raw_done_json_path and os.path.exists(raw_done_json_path):
                        os.remove(raw_done_json_path)
                except Exception:
                    pass
                try:
                    if raw_progress_path and os.path.exists(raw_progress_path):
                        os.remove(raw_progress_path)
                except Exception:
                    pass
                raw_transport_mode = "pipe"
                if bool(raw_use_shm):
                    raw_transport_mode = "shm_ring"
                    raw_shm_name = f"avalife_live_raw_{os.getpid()}_{int(time.time() * 1000.0)}"
                    raw_shm = shared_memory.SharedMemory(
                        name=str(raw_shm_name),
                        create=True,
                        size=int(raw_shm_nbytes),
                    )
                    live_raw_shm_write_header(
                        raw_shm.buf,
                        written_frames=0,
                        enqueued_frames=0,
                        backlog_bytes=0,
                        prompt_mode="speech",
                        mode_seq=0,
                        mode_start_frame=0,
                        source_chunk_idx=0,
                        source_chunk_start_frame=0,
                        done=False,
                    )
                else:
                    os.mkfifo(raw_pipe_path, 0o666)
                raw_progress_json_enabled = bool(
                    str(raw_transport_mode) != "shm_ring"
                    or str(os.getenv("LIVE_RAW_PROGRESS_JSON_FALLBACK", "0") or "0").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                meta = {
                    "version": 1,
                    "width": int(WIDTH),
                    "height": int(HEIGHT),
                    "fps": float(self.fps),
                    "pix_fmt": "rgb24",
                    "transport": str(raw_transport_mode),
                    "pipe_path": os.path.abspath(raw_pipe_path) if str(raw_transport_mode) == "pipe" else "",
                    "shm_name": str(raw_shm_name or ""),
                    "shm_header_bytes": int(raw_shm_header_bytes) if str(raw_transport_mode) == "shm_ring" else 0,
                    "ring_frame_capacity": int(raw_shm_frame_capacity) if str(raw_transport_mode) == "shm_ring" else 0,
                    "frame_bytes": int(raw_frame_bytes),
                    "created_at_ms": int(time.time() * 1000.0),
                }
                tmp_meta = meta_path + ".tmp"
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                os.replace(tmp_meta, meta_path)
                # Do not open FIFO in RDWR mode: it creates a local read-end and can deadlock
                # writes when external consumer disconnects. Open WRONLY only when reader is present.
                raw_fd = None
                print(
                    f"Rank {rank}: live RAW started at {live_raw_dir} transport={raw_transport_mode} "
                    f"{f'shm={raw_shm_name} frames={raw_shm_frame_capacity}' if raw_transport_mode == 'shm_ring' else ''}",
                    flush=True,
                )
            except Exception as e:
                print(f"Rank {rank}: failed to start live RAW: {e}", flush=True)
                raw_pipe_path = None
                raw_transport_mode = "pipe"
                if raw_shm is not None:
                    try:
                        raw_shm.close()
                    except Exception:
                        pass
                    try:
                        raw_shm.unlink()
                    except Exception:
                        pass
                raw_shm = None
                raw_shm_name = None
                raw_shm_frame_capacity = 0
                raw_fd = None
                raw_ready_path = None

        def _remote_edge_env_enabled() -> bool:
            return str(os.getenv("REMOTE_EDGE_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes", "on")

        def _remote_edge_load_manifest() -> dict[str, Any]:
            if not live_raw_dir:
                return {}
            path = os.path.join(str(live_raw_dir), "remote_edge.json")
            if not os.path.exists(path):
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                return dict(obj) if isinstance(obj, dict) else {}
            except Exception as e:
                print(f"Rank {rank}: failed to read remote edge manifest: {e}", flush=True)
                return {}

        def _remote_edge_poster_payload(manifest: dict[str, Any], *, width: int, height: int) -> bytes | None:
            path = str(manifest.get("poster_path") or "").strip()
            if not path or not os.path.exists(path):
                return None
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img).convert("RGB")
                src_w, src_h = img.size
                tgt_w, tgt_h = int(width), int(height)
                if src_w <= 0 or src_h <= 0 or tgt_w <= 0 or tgt_h <= 0:
                    return None

                scale = max(float(tgt_w) / float(src_w), float(tgt_h) / float(src_h))
                new_w = max(1, int(round(float(src_w) * float(scale))))
                new_h = max(1, int(round(float(src_h) * float(scale))))
                img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
                left = max(0, (new_w - tgt_w) // 2)
                top = max(0, (new_h - tgt_h) // 2)
                img = img.crop((left, top, left + tgt_w, top + tgt_h))
                arr = np.asarray(img, dtype=np.uint8)
                if arr.shape != (tgt_h, tgt_w, 3):
                    return None
                return arr.tobytes()
            except Exception as e:
                print(f"Rank {rank}: failed to build remote edge poster: {e}", flush=True)
                return None

        def _remote_edge_is_warmup_job() -> bool:
            return str(job_id or "").strip().lower().startswith("warmup_")

        def _remote_edge_start() -> bool:
            nonlocal remote_edge_sender, remote_edge_enabled, remote_edge_failed, remote_edge_mode, remote_edge_output
            nonlocal remote_edge_poster_sent, remote_edge_static_audio_sent
            nonlocal remote_edge_pace_started_at, remote_edge_file_target_frames
            if int(rank) != int(decode_rank):
                return False
            if remote_edge_sender is not None and bool(remote_edge_enabled):
                return True
            if bool(remote_edge_failed):
                return False
            manifest = _remote_edge_load_manifest()
            enabled = bool(manifest.get("enabled")) or bool(_remote_edge_env_enabled())
            if not enabled:
                return False
            host = str(manifest.get("host") or os.getenv("REMOTE_EDGE_HOST", "") or "").strip()
            port = int(manifest.get("port") or os.getenv("REMOTE_EDGE_PORT", "8787") or 8787)
            output = str(manifest.get("output") or ("rtmp" if str(manifest.get("rtmp_url") or "").strip() else "livekit")).strip().lower()
            if output not in {"livekit", "rtmp", "file"}:
                output = "livekit"
            remote_edge_output = str(output)
            try:
                target_duration = float(manifest.get("file_target_duration_sec") or 0.0)
            except Exception:
                target_duration = 0.0
            remote_edge_file_target_frames = (
                int(max(1, round(float(target_duration) * float(max(1, int(self.fps))))))
                if str(output) == "file" and float(target_duration) > 0.0
                else 0
            )
            livekit_url = str(manifest.get("livekit_url") or "").strip()
            livekit_token = str(manifest.get("livekit_token") or "").strip()
            rtmp_url = str(manifest.get("rtmp_url") or "").strip()
            rtmp_urls_raw = manifest.get("rtmp_urls")
            rtmp_urls = []
            if isinstance(rtmp_urls_raw, (list, tuple)):
                for item in rtmp_urls_raw:
                    item_s = str(item or "").strip()
                    if item_s and item_s not in rtmp_urls:
                        rtmp_urls.append(item_s)
            if rtmp_url and rtmp_url not in rtmp_urls:
                rtmp_urls.insert(0, str(rtmp_url))
            mode = str(manifest.get("mode") or os.getenv("REMOTE_EDGE_STREAM_MODE", "latents") or "latents").strip().lower()
            if mode not in {"latents", "rgb24"}:
                mode = "latents"
            remote_edge_mode = str(mode)
            missing_credentials = bool(
                (str(output) == "livekit" and (not livekit_url or not livekit_token))
                or (str(output) == "rtmp" and not rtmp_urls)
                or (str(output) == "file" and not str(manifest.get("file_upload_url") or "").strip())
            )
            if not host or bool(missing_credentials):
                if _remote_edge_is_warmup_job():
                    # Startup warmup runs before reply_handler writes the live remote_edge.json
                    # manifest. Keep the producer warm without treating absent publish
                    # credentials as a broken live edge.
                    return False
                remote_edge_failed = True
                print(
                    f"Rank {rank}: remote edge disabled: missing host/output credentials "
                    f"host={host or '-'} output={output} livekit={1 if livekit_url else 0} "
                    f"token={1 if livekit_token else 0} rtmp={1 if rtmp_urls else 0}",
                    flush=True,
                )
                return False
            timeout_sec = float(manifest.get("connect_timeout_sec") or os.getenv("REMOTE_EDGE_CONNECT_TIMEOUT_SEC", "5.0") or 5.0)
            write_timeout_sec = float(
                manifest.get("write_timeout_sec")
                or os.getenv("REMOTE_EDGE_WRITE_TIMEOUT_SEC", "10.0")
                or 10.0
            )
            write_timeout_sec = float(max(1.0, min(300.0, write_timeout_sec)))
            try:
                from avalife.remote.sync_sender import AsyncRemoteStreamSender, SyncRemoteStreamSender

                sender = SyncRemoteStreamSender(host=host, port=int(port), timeout_sec=float(timeout_sec))
                sender.connect()
                sender.hello(
                    session_id=str(manifest.get("session_id") or job_id or f"remote-{int(time.time() * 1000)}"),
                    job_id=str(manifest.get("job_id") or job_id or ""),
                    livekit_url=str(livekit_url),
                    livekit_token=str(livekit_token),
                    output=str(output),
                    rtmp_url=str(rtmp_urls[0] if rtmp_urls else ""),
                    rtmp_urls=list(rtmp_urls),
                    live_session_id=str(manifest.get("live_session_id") or manifest.get("session_id") or ""),
                    workspace_id=str(manifest.get("workspace_id") or ""),
                    persona_id=str(manifest.get("persona_id") or ""),
                    width=int(manifest.get("width") or WIDTH),
                    height=int(manifest.get("height") or HEIGHT),
                    fps=int(manifest.get("fps") or self.fps),
                    sample_rate=int(manifest.get("sample_rate") or 16000),
                    mode=str(mode),
                    auth_token=str(manifest.get("auth_token") or os.getenv("REMOTE_EDGE_SHARED_SECRET", "") or ""),
                    file_upload_url=str(manifest.get("file_upload_url") or ""),
                    file_upload_path=str(manifest.get("file_upload_path") or ""),
                    file_public_url=str(manifest.get("file_public_url") or ""),
                    file_content_type=str(manifest.get("file_content_type") or "video/mp4"),
                    file_progress_path=str(manifest.get("file_progress_path") or ""),
                    file_output_fps=int(manifest.get("file_output_fps") or 0),
                    file_target_audio_samples=int(manifest.get("file_target_audio_samples") or 0),
                    file_target_duration_sec=float(manifest.get("file_target_duration_sec") or 0.0),
                    watermark_text=str(manifest.get("watermark_text") or ""),
                    file_remote_finalizer=(
                        bool(manifest.get("file_remote_finalizer"))
                        if "file_remote_finalizer" in manifest
                        else None
                    ),
                )
                try:
                    sender.set_timeout(float(write_timeout_sec))
                except Exception:
                    pass
                if str(output) == "file" and not bool(remote_edge_static_audio_sent):
                    static_audio_path = str(audio_path or "").strip()
                    if static_audio_path and (not static_audio_path.startswith("liveaudio://")) and os.path.exists(static_audio_path):
                        sender.send_wav_pcm16le(static_audio_path, source_chunk_idx=0)
                        remote_edge_static_audio_sent = True
                send_startup_poster = str(
                    os.getenv("REMOTE_EDGE_SEND_STARTUP_POSTER", "1") or "1"
                ).strip().lower() in {"1", "true", "yes", "on"}
                poster_payload = (
                    _remote_edge_poster_payload(
                        manifest,
                        width=int(manifest.get("width") or WIDTH),
                        height=int(manifest.get("height") or HEIGHT),
                    )
                    if bool(send_startup_poster)
                    else None
                )
                if poster_payload:
                    sender.send_poster_rgb24(poster_payload)
                    remote_edge_poster_sent = True
                async_sender = str(os.getenv("REMOTE_EDGE_ASYNC_SENDER", "1") or "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                    "",
                )
                remote_sender = AsyncRemoteStreamSender.from_env(sender) if bool(async_sender) else sender
                remote_edge_sender = remote_sender
                remote_edge_enabled = True
                remote_edge_mode = str(mode)
                remote_edge_pace_started_at = time.perf_counter()
                print(
                    f"Rank {rank}: remote edge connected host={host}:{int(port)} mode={remote_edge_mode} "
                    f"output={output} size={int(manifest.get('width') or WIDTH)}x{int(manifest.get('height') or HEIGHT)} "
                    f"poster={1 if remote_edge_poster_sent else 0} async={1 if async_sender else 0}",
                    flush=True,
                )
                return True
            except Exception as e:
                remote_edge_failed = True
                remote_edge_enabled = False
                remote_edge_sender = None
                print(f"Rank {rank}: remote edge connect failed: {e}", flush=True)
                return False

        def _remote_edge_tensor_payload(tensor: torch.Tensor, *, codec: str = "torch.save") -> bytes:
            codec_s = str(codec or "torch.save").strip().lower()
            if codec_s in {"raw", "raw_tensor", "tensor.raw"}:
                raw = tensor.detach().to("cpu").contiguous()
                return raw.view(torch.uint8).numpy().tobytes()
            buf = io.BytesIO()
            torch.save(tensor.detach().to("cpu"), buf)
            return buf.getvalue()

        def _remote_edge_float_env_override(name: str, default: float) -> float:
            raw = str(os.getenv(str(name), "") or "").strip()
            if raw == "":
                return float(default)
            try:
                return float(max(0.0, min(1.0, float(raw))))
            except Exception:
                return float(default)

        def _remote_edge_skip_local_decode() -> bool:
            raw = str(os.getenv("REMOTE_EDGE_SKIP_LOCAL_DECODE", "1") or "1").strip().lower()
            return raw not in {"0", "false", "no", "off", ""}

        def _remote_edge_fail_fatal() -> bool:
            if _remote_edge_is_warmup_job():
                return False
            default = "1" if _remote_edge_skip_local_decode() else "0"
            raw = str(os.getenv("REMOTE_EDGE_FAIL_FATAL", default) or default).strip().lower()
            return raw not in {"0", "false", "no", "off", ""}

        def _remote_edge_abort_for_cancel(reason: str) -> None:
            nonlocal remote_edge_enabled, remote_edge_failed
            sender = remote_edge_sender
            if sender is None:
                return
            remote_edge_enabled = False
            remote_edge_failed = True
            try:
                if hasattr(sender, "abort"):
                    sender.abort()
                else:
                    sender.close(drain=False)
                print(f"Rank {rank}: remote edge aborted after cancel: {reason}", flush=True)
            except TypeError:
                try:
                    sender.close()
                except Exception as close_e:
                    print(f"Rank {rank}: remote edge abort close failed: {close_e}", flush=True)
            except Exception as close_e:
                print(f"Rank {rank}: remote edge abort failed: {close_e}", flush=True)

        def _remote_edge_request_cancel_or_raise(reason: str) -> None:
            reason_s = str(reason or "remote_edge_failed").strip() or "remote_edge_failed"
            _remote_edge_abort_for_cancel(reason_s)
            if dist.is_initialized() and int(rank) != 0:
                try:
                    request_infer_cancel(job_id=job_id, reason=reason_s)
                    print(
                        f"Rank {rank}: requested synchronized inference cancel: {reason_s}",
                        flush=True,
                    )
                    return
                except Exception as cancel_e:
                    print(
                        f"Rank {rank}: failed to request synchronized cancel: {cancel_e}",
                        flush=True,
                    )
            raise InferenceCancelled(reason_s)

        def _remote_edge_latent_decode_delegated() -> bool:
            if not bool(_remote_edge_skip_local_decode()):
                return False
            if bool(remote_edge_failed):
                return False
            mode = str(remote_edge_mode or os.getenv("REMOTE_EDGE_STREAM_MODE", "latents") or "latents").strip().lower()
            # Producer-latent mode is intentionally one-way: this 2-GPU node owns
            # DiT/denoise only. VAE/RGB/PostVAE/publish may be skipped only after
            # an edge sender is actually connected; otherwise startup warmup can
            # finish with zero raw/shm frames and leave the first real live job to
            # warm the path.
            return bool(mode == "latents" and _remote_edge_env_enabled() and bool(remote_edge_enabled))

        def _remote_edge_sender_queue_snapshot_fast() -> tuple[int, int]:
            sender = remote_edge_sender
            if sender is None:
                return -1, -1
            queue_obj = getattr(sender, "queue", None)
            if queue_obj is None:
                return -1, -1
            try:
                qsize = int(queue_obj.qsize())
            except Exception:
                qsize = -1
            try:
                qmax = int(getattr(sender, "max_queue", -1) or -1)
            except Exception:
                qmax = -1
            return int(qsize), int(qmax)

        def _remote_edge_estimated_lead_frames(*, extra_frames: int = 0) -> int:
            fps_i = max(1, int(round(float(self.fps))))
            started_at = float(remote_edge_pace_started_at or remote_edge_stats_t0)
            elapsed = max(0.0, float(time.perf_counter() - started_at))
            # The edge does not publish immediately: it first waits for a real
            # prebuffer. Account for that runway locally so producer pacing is
            # controlled by intended live lead, not by socket backpressure.
            consumed_est = max(
                0,
                int(math.floor(float(elapsed) * float(fps_i))) - int(remote_edge_producer_prebuffer_frames),
            )
            return max(0, int(remote_edge_video_frames_sent) + int(max(0, int(extra_frames))) - int(consumed_est))

        def _remote_edge_pace_before_enqueue(*, extra_frames: int) -> None:
            nonlocal remote_edge_last_pace_sleep_dt, remote_edge_pace_total_sleep_sec, remote_edge_pace_last_log
            if str(remote_edge_output or "").strip().lower() == "file":
                remote_edge_last_pace_sleep_dt = 0.0
                return
            max_lead_sec = float(remote_edge_producer_max_lead_sec)
            if max_lead_sec <= 0.0 or int(extra_frames) <= 0:
                remote_edge_last_pace_sleep_dt = 0.0
                return
            fps_i = max(1, int(round(float(self.fps))))
            max_lead_frames = int(round(float(max_lead_sec) * float(fps_i)))
            slept_total = 0.0
            while True:
                lead_frames = _remote_edge_estimated_lead_frames(extra_frames=int(extra_frames))
                qsize, qmax = _remote_edge_sender_queue_snapshot_fast()
                queue_hot = bool(qmax > 0 and qsize >= int(max(1, int(qmax) * 0.90)))
                over_frames = max(0, int(lead_frames) - int(max_lead_frames))
                if int(over_frames) <= 0 and not bool(queue_hot):
                    break
                if bool(queue_hot):
                    sleep_sec = min(float(remote_edge_producer_pace_sleep_max_sec), 0.05)
                else:
                    sleep_sec = min(
                        float(remote_edge_producer_pace_sleep_max_sec),
                        max(0.01, float(over_frames) / float(fps_i)),
                    )
                time.sleep(float(sleep_sec))
                slept_total += float(sleep_sec)
                checker = getattr(remote_edge_sender, "check_error", None)
                if checker is not None:
                    checker()
                now_log = time.monotonic()
                if now_log - float(remote_edge_pace_last_log) >= 5.0:
                    logging.warning(
                        "Remote edge producer pacing: lead=%.2fs max=%.2fs sleep=%.3fs total=%.3fs sender_q=%d/%d prebuffer_frames=%d",
                        float(lead_frames) / float(fps_i),
                        float(max_lead_sec),
                        float(sleep_sec),
                        float(slept_total),
                        int(qsize),
                        int(qmax),
                        int(remote_edge_producer_prebuffer_frames),
                    )
                    remote_edge_pace_last_log = float(now_log)
            remote_edge_last_pace_sleep_dt = float(slept_total)
            remote_edge_pace_total_sleep_sec += float(slept_total)

        def _remote_edge_send_latents(
            tensor: torch.Tensor,
            *,
            reset_vae: bool = False,
            prime_only: bool = False,
            keep_last_frames: int | None = None,
            segment_id: str | None = None,
            segment_kind: str = "",
            segment_audio_pcm16le: bytes | bytearray | memoryview | None = None,
            segment_sample_rate: int | None = None,
            segment_start_frame: int | None = None,
            segment_frames: int | None = None,
            segment_audible_samples: int | None = None,
            segment_subtitle_text: str | None = None,
            segment_subtitle_start_samples: int | None = None,
            segment_subtitle_end_samples: int | None = None,
            segment_subtitle_total_samples: int | None = None,
            segment_subtitle_alignment: dict | None = None,
            segment_subtitle_normalized_alignment: dict | None = None,
            segment_subtitle_alignment_base_samples: int | None = None,
            segment_turn_done: bool = False,
            avatar_ref_path: str | None = None,
        ) -> None:
            nonlocal remote_edge_video_frames_sent, remote_edge_enabled, remote_edge_failed
            nonlocal remote_edge_last_send_dt
            if not _remote_edge_start() or remote_edge_sender is None or str(remote_edge_mode) != "latents":
                if _remote_edge_fail_fatal() and _remote_edge_env_enabled():
                    _remote_edge_request_cancel_or_raise("remote_edge_unavailable")
                return
            try:
                send_t0 = time.perf_counter()
                frames_to_add = 0
                if not bool(prime_only):
                    if keep_last_frames is not None and int(keep_last_frames) > 0:
                        frames_to_add = int(keep_last_frames)
                    elif tensor.ndim == 4:
                        frames_to_add = int(tensor.shape[1])
                    elif tensor.ndim == 5:
                        frames_to_add = int(tensor.shape[2])
                    if (
                        str(segment_id or "").strip()
                        and segment_audio_pcm16le is not None
                        and segment_frames is not None
                    ):
                        frames_to_add = int(max(0, min(int(frames_to_add), int(segment_frames))))
                    _remote_edge_pace_before_enqueue(extra_frames=int(frames_to_add))
                    if int(frames_to_add) > 0 and not (
                        str(segment_id or "").strip() and segment_audio_pcm16le is not None
                    ):
                        _remote_edge_send_audio_until_for_frames(
                            int(remote_edge_video_frames_sent)
                            + int(frames_to_add)
                            + int(_remote_edge_audio_lead_frames())
                        )
                frame_step_us = int(round(1_000_000.0 / float(max(1, int(self.fps)))))
                ts_us = int(remote_edge_first_ts_us + int(remote_edge_video_frames_sent) * int(frame_step_us))
                latent_codec = str(os.getenv("REMOTE_EDGE_LATENT_CODEC", "torch.save") or "torch.save").strip()
                fields = {
                    "codec": latent_codec,
                    "shape": str(tuple(int(v) for v in tensor.shape)),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "timestamp_us": None if bool(prime_only) else int(ts_us),
                    "keep_last_frames": keep_last_frames,
                    "reset_vae": bool(reset_vae),
                    "prime_only": bool(prime_only),
                    "face_restore": _remote_edge_float_env_override(
                        "REMOTE_EDGE_FACE_RESTORE",
                        float(getattr(self, "_post_vae_face_restore", 0.0) or 0.0),
                    ),
                    "background_restore": _remote_edge_float_env_override(
                        "REMOTE_EDGE_BACKGROUND_RESTORE",
                        float(getattr(self, "_post_vae_background_restore", 0.0) or 0.0),
                    ),
                    "avatar_ref_path": None if avatar_ref_path is None else str(avatar_ref_path),
                }
                segment_id_s = str(segment_id or "").strip()
                if (not bool(prime_only)) and segment_id_s and segment_audio_pcm16le is not None:
                    segment_frame_count = int(segment_frames if segment_frames is not None else frames_to_add)
                    segment_sr = int(max(1, int(segment_sample_rate or _required_int_env("WORKER_AUDIO_SAMPLE_RATE"))))
                    if hasattr(remote_edge_sender, "send_pcm16le"):
                        remote_edge_sender.send_pcm16le(
                            bytes(segment_audio_pcm16le or b""),
                            sample_rate=int(segment_sr),
                            segment_id=str(segment_id_s),
                            segment_kind=str(segment_kind or ""),
                            segment_frames=int(max(0, int(segment_frame_count))),
                            segment_audible_samples=(
                                None if segment_audible_samples is None else int(max(0, int(segment_audible_samples)))
                            ),
                            segment_turn_done=bool(segment_turn_done),
                            subtitle_text=None if segment_subtitle_text is None else str(segment_subtitle_text),
                            subtitle_start_samples=(
                                None
                                if segment_subtitle_start_samples is None
                                else int(max(0, int(segment_subtitle_start_samples)))
                            ),
                            subtitle_end_samples=(
                                None
                                if segment_subtitle_end_samples is None
                                else int(max(0, int(segment_subtitle_end_samples)))
                            ),
                            subtitle_total_samples=(
                                None
                                if segment_subtitle_total_samples is None
                                else int(max(0, int(segment_subtitle_total_samples)))
                            ),
                            subtitle_alignment=(
                                dict(segment_subtitle_alignment)
                                if isinstance(segment_subtitle_alignment, dict)
                                else None
                            ),
                            subtitle_normalized_alignment=(
                                dict(segment_subtitle_normalized_alignment)
                                if isinstance(segment_subtitle_normalized_alignment, dict)
                                else None
                            ),
                            subtitle_alignment_base_samples=(
                                None
                                if segment_subtitle_alignment_base_samples is None
                                else int(max(0, int(segment_subtitle_alignment_base_samples)))
                            ),
                        )
                    fields.update(
                        {
                            "segment_id": str(segment_id_s),
                            "segment_kind": str(segment_kind or ""),
                            "segment_start_frame": None if segment_start_frame is None else int(max(0, int(segment_start_frame))),
                            "segment_frames": int(max(0, int(segment_frame_count))),
                        }
                    )
                if hasattr(remote_edge_sender, "send_latents_tensor"):
                    remote_edge_sender.send_latents_tensor(tensor, **fields)
                else:
                    payload = _remote_edge_tensor_payload(tensor, codec=str(latent_codec))
                    remote_edge_sender.send_latents(payload, **fields)
                remote_edge_last_send_dt = float(time.perf_counter() - send_t0)
                if not bool(prime_only):
                    remote_edge_video_frames_sent += int(frames_to_add)
                    if str(remote_edge_output or "").strip().lower() == "file":
                        target_frames = int(max(1, int(remote_edge_file_target_frames or remote_edge_video_frames_sent)))
                        progress = float(remote_edge_video_frames_sent) / float(target_frames)
                        notifier = getattr(remote_edge_sender, "note_file_progress", None)
                        if notifier is not None:
                            notifier(
                                phase="inference",
                                progress=float(max(0.0, min(1.0, progress))),
                                frames_sent=int(remote_edge_video_frames_sent),
                                target_frames=int(target_frames),
                            )
            except Exception as e:
                remote_edge_failed = True
                remote_edge_enabled = False
                print(f"Rank {rank}: remote edge latent send failed: {e}", flush=True)
                if _remote_edge_fail_fatal():
                    _remote_edge_request_cancel_or_raise(f"remote_edge_send_failed: {e}")

        def _remote_edge_audio_lead_frames() -> int:
            try:
                return max(0, min(240, int(os.getenv("REMOTE_EDGE_AUDIO_LEAD_FRAMES", "48") or 48)))
            except Exception:
                return 48

        def _remote_edge_send_audio_chunk(source_chunk_idx: int) -> None:
            nonlocal remote_edge_enabled, remote_edge_failed
            idx = int(max(0, int(source_chunk_idx or 0)))
            if idx <= 0 or idx in remote_edge_audio_sent_chunks:
                return
            if not _remote_edge_start() or remote_edge_sender is None:
                return
            if not stream_audio_dir:
                return
            path = _stream_chunk_path(idx)
            if not os.path.exists(path):
                return
            try:
                if hasattr(remote_edge_sender, "send_wav_pcm16le"):
                    remote_edge_sender.send_wav_pcm16le(path, source_chunk_idx=int(idx))
                    remote_edge_audio_sent_chunks.add(idx)
                    return
                with wave.open(path, "rb") as wf:
                    sample_rate = int(wf.getframerate())
                    channels = int(wf.getnchannels())
                    sampwidth = int(wf.getsampwidth())
                    pcm = wf.readframes(wf.getnframes())
                if channels != 1 or sampwidth != 2:
                    print(
                        f"Rank {rank}: remote edge audio skip unsupported wav chunk={idx} "
                        f"channels={channels} sampwidth={sampwidth}",
                        flush=True,
                    )
                    return
                remote_edge_sender.send_pcm16le(pcm, sample_rate=int(sample_rate))
                remote_edge_audio_sent_chunks.add(idx)
            except Exception as e:
                remote_edge_failed = True
                remote_edge_enabled = False
                print(f"Rank {rank}: remote edge audio send failed chunk={idx}: {e}", flush=True)
                if _remote_edge_fail_fatal():
                    raise InferenceCancelled(f"remote_edge_audio_send_failed: {e}") from e

        def _remote_edge_send_audio_until_for_frames(target_video_frames: int) -> None:
            nonlocal remote_edge_audio_next_chunk_idx
            if int(target_video_frames) <= 0:
                return
            try:
                fps_i = max(1, int(round(float(self.fps))))
            except Exception:
                fps_i = 12
            sample_rate_i = max(1, int(_required_int_env("WORKER_AUDIO_SAMPLE_RATE")))
            chunk_samples_i = max(1, int(_required_int_env("WORKER_LIVEAUDIO_MICRO_CHUNK_SCHEDULE_SAMPLES")))
            target_samples = int(math.ceil(float(max(0, int(target_video_frames))) * float(sample_rate_i) / float(fps_i)))
            target_chunk_idx = int(math.ceil(float(target_samples) / float(chunk_samples_i)))
            idx = int(max(1, int(remote_edge_audio_next_chunk_idx)))
            try:
                skip_before_idx = int(max(0, int(_stream_soft_break_chunk_idx())))
            except Exception:
                skip_before_idx = 0
            if int(skip_before_idx) > int(idx):
                # A soft break means earlier unsent chunks are stale. The edge
                # protocol is FIFO PCM, so sending old chunks after a break would
                # create audible repeats and A/V drift.
                if remote_edge_sender is not None and hasattr(remote_edge_sender, "drop_audio_before"):
                    try:
                        remote_edge_sender.drop_audio_before(int(skip_before_idx))
                    except Exception:
                        pass
                idx = int(skip_before_idx)
            while int(idx) <= int(target_chunk_idx):
                if idx not in remote_edge_audio_sent_chunks:
                    _remote_edge_send_audio_chunk(int(idx))
                    if idx not in remote_edge_audio_sent_chunks:
                        break
                idx += 1
            remote_edge_audio_next_chunk_idx = int(max(1, int(idx)))

        def _remote_edge_sender_queue_snapshot() -> tuple[int, int]:
            sender = remote_edge_sender
            if sender is None:
                return -1, -1
            queue_obj = getattr(sender, "queue", None)
            if queue_obj is None:
                return -1, -1
            try:
                qsize = int(queue_obj.qsize())
            except Exception:
                qsize = -1
            try:
                qmax = int(getattr(sender, "max_queue", -1) or -1)
            except Exception:
                qmax = -1
            return int(qsize), int(qmax)

        def _remote_edge_log_producer_stats(*, force: bool = False, reason: str = "interval") -> None:
            nonlocal remote_edge_stats_last_ts, remote_edge_stats_last_frames
            nonlocal remote_edge_stats_last_audio_chunks
            if float(remote_edge_stats_interval_sec) <= 0.0:
                return
            now = time.perf_counter()
            if (not bool(force)) and (now - float(remote_edge_stats_last_ts)) < float(remote_edge_stats_interval_sec):
                return
            elapsed = max(0.001, float(now - float(remote_edge_stats_t0)))
            recent_elapsed = max(0.001, float(now - float(remote_edge_stats_last_ts)))
            frames_sent = int(max(0, int(remote_edge_video_frames_sent)))
            recent_frames = int(max(0, frames_sent - int(remote_edge_stats_last_frames)))
            audio_chunks_sent = int(len(remote_edge_audio_sent_chunks))
            recent_audio_chunks = int(max(0, audio_chunks_sent - int(remote_edge_stats_last_audio_chunks)))
            sender_q, sender_qmax = _remote_edge_sender_queue_snapshot()
            try:
                model_q = int(len(stream_audio_clips))
            except Exception:
                model_q = -1
            print(
                f"Rank {rank}: remote edge producer stats: reason={str(reason)} "
                f"frames_sent={int(frames_sent)} fps_recent={float(recent_frames) / float(recent_elapsed):.2f} "
                f"fps_avg={float(frames_sent) / float(elapsed):.2f} audio_chunks_sent={int(audio_chunks_sent)} "
                f"audio_chunks_recent={int(recent_audio_chunks)} sender_q={int(sender_q)}/{int(sender_qmax)} "
                f"model_q={int(model_q)} seen_chunks={int(stream_audio_seen_chunks)} "
                f"done={1 if bool(stream_audio_done) else 0} "
                f"clip_frames={int(remote_edge_last_clip_frames)} "
                f"blocks={int(remote_edge_last_num_blocks)} "
                f"kv={int(remote_edge_last_kv_cache_size)}/{int(remote_edge_last_max_seq_len)} "
                f"kv_cap_frames={int(remote_edge_last_kv_cap_frames)} "
                f"last_block={float(remote_edge_last_block_dt):.3f}s "
                f"denoise={float(remote_edge_last_denoise_dt):.3f}s "
                f"recv={float(remote_edge_last_recv_dt):.3f}s "
                f"send_wait={float(remote_edge_last_send_wait_dt):.3f}s "
                f"pace={float(remote_edge_last_pace_sleep_dt):.3f}s "
                f"pace_total={float(remote_edge_pace_total_sleep_sec):.1f}s "
                f"latent_send={float(remote_edge_last_send_dt):.3f}s",
                flush=True,
            )
            remote_edge_stats_last_ts = float(now)
            remote_edge_stats_last_frames = int(frames_sent)
            remote_edge_stats_last_audio_chunks = int(audio_chunks_sent)

        def _tpp_log_stage_stats(
            *,
            core_dt: float,
            recv_wait_dt: float,
            denoise_total_dt: float,
            send_wait_dt: float,
            steps_executed: int,
            step_start: int,
            step_end: int,
            num_steps: int,
            clip_idx: int,
            active_clips: int,
            block_index: int,
            num_blocks: int,
        ) -> None:
            nonlocal tpp_stage_stats_last_ts, tpp_stage_stats_blocks, tpp_stage_stats_last_blocks
            nonlocal tpp_stage_stats_last_core_dt, tpp_stage_stats_last_recv_dt
            nonlocal tpp_stage_stats_last_denoise_dt, tpp_stage_stats_last_send_dt
            nonlocal tpp_stage_stats_last_steps
            if (not bool(stream_audio_mode)) or float(tpp_stage_stats_interval_sec) <= 0.0:
                return
            tpp_stage_stats_blocks += 1
            tpp_stage_stats_last_core_dt = float(core_dt)
            tpp_stage_stats_last_recv_dt = float(recv_wait_dt)
            tpp_stage_stats_last_denoise_dt = float(denoise_total_dt)
            tpp_stage_stats_last_send_dt = float(send_wait_dt)
            tpp_stage_stats_last_steps = int(steps_executed)
            now = time.perf_counter()
            if (now - float(tpp_stage_stats_last_ts)) < float(tpp_stage_stats_interval_sec):
                return
            recent_elapsed = max(0.001, float(now - float(tpp_stage_stats_last_ts)))
            elapsed = max(0.001, float(now - float(tpp_stage_stats_t0)))
            blocks_recent = int(max(0, int(tpp_stage_stats_blocks) - int(tpp_stage_stats_last_blocks)))
            try:
                model_q = int(len(stream_audio_clips))
            except Exception:
                model_q = -1
            print(
                f"Rank {rank}: TPP stage stats: "
                f"blocks={int(tpp_stage_stats_blocks)} "
                f"blocks_recent={int(blocks_recent)} "
                f"block_rate_recent={float(blocks_recent) / float(recent_elapsed):.2f}/s "
                f"block_rate_avg={float(tpp_stage_stats_blocks) / float(elapsed):.2f}/s "
                f"clip={int(clip_idx) + 1}/{int(active_clips)} "
                f"block={int(block_index) + 1}/{int(num_blocks)} "
                f"steps={int(step_start) + 1}-{int(step_end)}/{int(num_steps)} "
                f"last_core={float(tpp_stage_stats_last_core_dt):.3f}s "
                f"recv={float(tpp_stage_stats_last_recv_dt):.3f}s "
                f"denoise={float(tpp_stage_stats_last_denoise_dt):.3f}s "
                f"send={float(tpp_stage_stats_last_send_dt):.3f}s "
                f"steps_done={int(tpp_stage_stats_last_steps)} "
                f"q={int(model_q)} seen_chunks={int(stream_audio_seen_chunks)} "
                f"done={1 if bool(stream_audio_done) else 0}",
                flush=True,
            )
            tpp_stage_stats_last_ts = float(now)
            tpp_stage_stats_last_blocks = int(tpp_stage_stats_blocks)

        def _stream_file_audio_path() -> str:
            audio_for_file = str(audio_path or "").strip()
            if "+" in audio_for_file:
                audio_for_file = audio_for_file.split("+", 1)[0]
            return audio_for_file

        def _stream_file_expected_output_frames() -> int:
            if float(stream_file_trim_sec) > 0.0 and float(stream_file_fps) > 0.0:
                return max(1, int(round(float(stream_file_trim_sec) * float(stream_file_fps))))
            return 0

        def _stream_file_write_progress(*, phase: str = "inference", error: str = "") -> None:
            if not bool(stream_file_enabled) or not str(stream_file_progress_path or "").strip():
                return
            expected_frames = int(_stream_file_expected_output_frames())
            frac = 0.0
            if expected_frames > 0:
                frac = float(max(0.0, min(1.0, float(stream_file_frames_out) / float(expected_frames))))
            payload = {
                "phase": str(phase or "inference"),
                "progress": float(frac),
                "frames_in": int(stream_file_frames_in),
                "frames_out": int(stream_file_frames_out),
                "expected_frames": int(expected_frames),
                "blocks": int(stream_file_blocks),
                "queue_blocks": int(len(stream_file_queue)),
                "rife_sec": float(stream_file_rife_s),
                "resize_sec": float(stream_file_resize_s),
                "pack_sec": float(stream_file_pack_s),
                "write_sec": float(stream_file_write_s),
                "error": str(error or ""),
                "updated_at_ms": int(time.time() * 1000.0),
            }
            try:
                tmp = str(stream_file_progress_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                os.replace(tmp, stream_file_progress_path)
            except Exception:
                pass

        def _stream_file_encoder_cmd() -> list[str]:
            audio_for_file = _stream_file_audio_path()
            if not audio_for_file or not os.path.exists(audio_for_file):
                raise RuntimeError(f"stream_file_output audio not found: {audio_for_file}")
            out_w = int(stream_file_output_w or WIDTH)
            out_h = int(stream_file_output_h or HEIGHT)
            out_fps = float(stream_file_fps or self.fps)
            gop = max(1, int(round(out_fps * 2.0)))
            encoder = str(os.getenv("SMARTBLOG_STREAM_FILE_VIDEO_ENCODER", "libx264") or "libx264").strip()
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                str(os.getenv("SMARTBLOG_STREAM_FILE_FFMPEG_LOGLEVEL", "warning") or "warning"),
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{int(out_w)}x{int(out_h)}",
                "-r",
                f"{float(out_fps):.6f}",
                "-thread_queue_size",
                "1024",
                "-i",
                "pipe:0",
                "-thread_queue_size",
                "1024",
                "-i",
                str(audio_for_file),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                "setsar=1",
                "-c:v",
                str(encoder),
            ]
            if str(encoder) == "libx264":
                cmd += [
                    "-preset",
                    str(os.getenv("SMARTBLOG_STREAM_FILE_X264_PRESET", "veryfast") or "veryfast"),
                ]
                x264_tune = str(os.getenv("SMARTBLOG_STREAM_FILE_X264_TUNE", "") or "").strip()
                if x264_tune:
                    cmd += ["-tune", x264_tune]
                cmd += [
                    "-crf",
                    str(os.getenv("SMARTBLOG_STREAM_FILE_X264_CRF", "18") or "18"),
                ]
            elif str(encoder).endswith("_nvenc"):
                cmd += [
                    "-preset",
                    str(os.getenv("SMARTBLOG_STREAM_FILE_NVENC_PRESET", "p4") or "p4"),
                    "-rc",
                    "vbr",
                    "-cq",
                    str(os.getenv("SMARTBLOG_STREAM_FILE_NVENC_CQ", "18") or "18"),
                    "-b:v",
                    "0",
                ]
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-r",
                f"{float(out_fps):.6f}",
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-sc_threshold",
                "0",
                "-c:a",
                "aac",
                "-b:a",
                str(os.getenv("SMARTBLOG_STREAM_FILE_AUDIO_BITRATE", "128k") or "128k"),
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
            cmd += ["-shortest", "-movflags", "+faststart", str(stream_file_path)]
            return cmd

        def _stream_file_get_rife_interpolator():
            from avalife.remote.torch_rife import get_shared_torch_rife_interpolator

            model_dir = str(
                os.getenv(
                    "SMARTBLOG_STREAM_FILE_RIFE_MODEL_DIR",
                    os.getenv("REMOTE_EDGE_TORCH_RIFE_MODEL_DIR", "/opt/RIFE-safetensors"),
                )
                or "/opt/RIFE-safetensors"
            )
            weights_path = str(
                os.getenv(
                    "SMARTBLOG_STREAM_FILE_RIFE_WEIGHTS",
                    os.getenv("REMOTE_EDGE_TORCH_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors")),
                )
                or os.path.join(model_dir, "flownet.safetensors")
            )
            device = str(
                os.getenv("SMARTBLOG_STREAM_FILE_RIFE_DEVICE", os.getenv("REMOTE_EDGE_TORCH_RIFE_DEVICE", str(self.device)))
                or str(self.device)
            )
            dtype_name = str(
                os.getenv("SMARTBLOG_STREAM_FILE_RIFE_DTYPE", os.getenv("REMOTE_EDGE_TORCH_RIFE_DTYPE", "float16"))
                or "float16"
            )
            try:
                batch_pairs = int(
                    os.getenv(
                        "SMARTBLOG_STREAM_FILE_RIFE_BATCH_PAIRS",
                        os.getenv("REMOTE_EDGE_TORCH_RIFE_BATCH_PAIRS", "4"),
                    )
                    or 4
                )
            except Exception:
                batch_pairs = 4
            return get_shared_torch_rife_interpolator(
                model_dir=str(model_dir),
                weights_path=str(weights_path),
                device=str(device),
                dtype_name=str(dtype_name),
                batch_pairs=max(1, int(batch_pairs)),
            )

        def _stream_file_interpolate_x2(frames_01: torch.Tensor) -> torch.Tensor:
            nonlocal stream_file_last_frame, stream_file_rife_s
            if int(frames_01.shape[0]) <= 0:
                return frames_01
            if str(stream_file_interpolation_mode) not in {"torch-rife", "rife"}:
                return frames_01
            rife_t0 = time.perf_counter()
            interpolator = _stream_file_get_rife_interpolator()
            if stream_file_last_frame is not None:
                combined = torch.cat(
                    (stream_file_last_frame.to(device=frames_01.device, non_blocking=True), frames_01),
                    dim=0,
                )
                out = interpolator.interpolate_tensor_x2(
                    combined,
                    target_frames=int(combined.shape[0]) * 2 - 1,
                )[1:].contiguous()
            elif int(frames_01.shape[0]) >= 2:
                out = interpolator.interpolate_tensor_x2(
                    frames_01,
                    target_frames=int(frames_01.shape[0]) * 2 - 1,
                )
            else:
                out = frames_01.contiguous()
            stream_file_last_frame = frames_01[-1:].detach().contiguous()
            stream_file_rife_s += float(time.perf_counter() - rife_t0)
            return out

        def _stream_file_resize_output(frames_01: torch.Tensor) -> torch.Tensor:
            nonlocal stream_file_resize_s
            out_w = int(stream_file_output_w or frames_01.shape[3])
            out_h = int(stream_file_output_h or frames_01.shape[2])
            if int(frames_01.shape[2]) == int(out_h) and int(frames_01.shape[3]) == int(out_w):
                return frames_01
            resize_t0 = time.perf_counter()
            out = F.interpolate(
                frames_01,
                size=(int(out_h), int(out_w)),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
            stream_file_resize_s += float(time.perf_counter() - resize_t0)
            return out

        def _stream_file_tensor_to_rgb24(frames_01: torch.Tensor) -> bytes:
            nonlocal stream_file_pack_s
            pack_t0 = time.perf_counter()
            rgb = (frames_01.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            rgb = rgb.permute(0, 2, 3, 1).contiguous()
            payload = rgb.cpu().numpy().tobytes()
            stream_file_pack_s += float(time.perf_counter() - pack_t0)
            return payload

        def _stream_file_write_tensor(proc: subprocess.Popen, frames_01: torch.Tensor) -> int:
            nonlocal stream_file_write_s, stream_file_frames_out
            if proc.stdin is None or int(frames_01.shape[0]) <= 0:
                return 0
            expected_frames = int(_stream_file_expected_output_frames())
            if expected_frames > 0:
                remaining = int(expected_frames) - int(stream_file_frames_out)
                if remaining <= 0:
                    return 0
                if int(frames_01.shape[0]) > int(remaining):
                    frames_01 = frames_01[: int(remaining)].contiguous()
            frames_01 = _stream_file_resize_output(frames_01)
            payload = _stream_file_tensor_to_rgb24(frames_01)
            write_t0 = time.perf_counter()
            proc.stdin.write(payload)
            stream_file_write_s += float(time.perf_counter() - write_t0)
            written_frames = int(frames_01.shape[0])
            stream_file_frames_out += int(written_frames)
            return int(written_frames)

        def _stream_file_start_writer() -> None:
            nonlocal stream_file_thread, stream_file_error, stream_file_started_s, stream_file_finished_s
            if not bool(stream_file_enabled) or stream_file_thread is not None:
                return
            if int(stream_file_output_w or 0) <= 0 or int(stream_file_output_h or 0) <= 0:
                raise RuntimeError("stream_file_output requires positive output width/height")
            os.makedirs(os.path.dirname(str(stream_file_path)) or ".", exist_ok=True)
            stream_file_started_s = float(time.perf_counter())
            stream_file_error = None

            def _writer_main() -> None:
                nonlocal stream_file_error, stream_file_finished_s, stream_file_frames_in, stream_file_blocks
                nonlocal stream_file_last_frame
                proc = None
                log_f = None
                try:
                    cmd = _stream_file_encoder_cmd()
                    log_f = open(str(stream_file_path) + ".ffmpeg.log", "w", encoding="utf-8")
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f)
                    print(
                        f"Rank {rank}: stream-file encoder started: output={stream_file_path} "
                        f"size={int(stream_file_output_w)}x{int(stream_file_output_h)} "
                        f"fps={float(stream_file_fps):.3f} interpolation={stream_file_interpolation_mode or 'off'}",
                        flush=True,
                    )
                    _stream_file_write_progress(phase="inference")
                    last_log = float(time.perf_counter())
                    while True:
                        entry = None
                        with stream_file_cv:
                            while len(stream_file_queue) <= 0 and not bool(stream_file_stop.is_set()):
                                stream_file_cv.wait(timeout=0.25)
                            if len(stream_file_queue) > 0:
                                entry = stream_file_queue.popleft()
                                stream_file_cv.notify_all()
                            elif bool(stream_file_stop.is_set()):
                                break
                        if entry is None:
                            continue
                        event = entry.get("event")
                        if event is not None:
                            event.synchronize()
                        frames_01 = entry.get("tensor")
                        if not torch.is_tensor(frames_01):
                            continue
                        frames_01 = frames_01.detach().to(device=self.device, dtype=torch.float16, non_blocking=True).contiguous()
                        stream_file_frames_in += int(frames_01.shape[0])
                        stream_file_blocks += 1
                        expected_frames = int(_stream_file_expected_output_frames())
                        if expected_frames > 0 and int(stream_file_frames_out) >= int(expected_frames):
                            continue
                        out = _stream_file_interpolate_x2(frames_01)
                        _stream_file_write_tensor(proc, out)
                        _stream_file_write_progress(phase="inference")
                        now = float(time.perf_counter())
                        if now - last_log >= 10.0:
                            print(
                                f"TPP stream-file timing job={str(job_id or '-')} "
                                f"blocks={int(stream_file_blocks)} frames={int(stream_file_frames_in)}->{int(stream_file_frames_out)} "
                                f"q={int(len(stream_file_queue))} rife={float(stream_file_rife_s):.3f}s "
                                f"resize={float(stream_file_resize_s):.3f}s pack={float(stream_file_pack_s):.3f}s "
                                f"write={float(stream_file_write_s):.3f}s",
                                flush=True,
                            )
                            last_log = now
                    if str(stream_file_interpolation_mode) in {"torch-rife", "rife"} and stream_file_last_frame is not None:
                        _stream_file_write_progress(phase="encoding")
                        _stream_file_write_tensor(proc, stream_file_last_frame)
                    expected_frames = int(_stream_file_expected_output_frames())
                    if expected_frames > 0 and stream_file_last_frame is not None:
                        _stream_file_write_progress(phase="encoding")
                        while int(stream_file_frames_out) < int(expected_frames):
                            _stream_file_write_tensor(proc, stream_file_last_frame)
                    if proc.stdin is not None:
                        proc.stdin.close()
                    try:
                        return_code = proc.wait(timeout=float(os.getenv("SMARTBLOG_STREAM_FILE_FFMPEG_WAIT_SEC", "180") or 180))
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        return_code = proc.wait(timeout=15)
                    if int(return_code) != 0:
                        raise RuntimeError(f"stream-file ffmpeg exited with code {return_code}")
                    if not os.path.exists(stream_file_path) or os.path.getsize(stream_file_path) <= 0:
                        raise RuntimeError(f"stream-file encoder produced no output: {stream_file_path}")
                    _stream_file_write_progress(phase="done")
                except Exception as e:
                    stream_file_error = str(e)
                    _stream_file_write_progress(phase="error", error=str(e))
                    print(f"Rank {rank}: stream-file writer failed: {e}", flush=True)
                finally:
                    try:
                        if proc is not None and proc.stdin is not None and not proc.stdin.closed:
                            proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        if log_f is not None:
                            log_f.close()
                    except Exception:
                        pass
                    stream_file_finished_s = float(time.perf_counter())
                    with stream_file_cv:
                        if stream_file_error:
                            stream_file_stop.set()
                        stream_file_cv.notify_all()

            stream_file_thread = threading.Thread(
                target=_writer_main,
                name=f"stream-file-writer-r{rank}",
                daemon=True,
            )
            stream_file_thread.start()

        def _stream_file_enqueue_frames(frames_01: torch.Tensor) -> None:
            nonlocal stream_file_enqueue_s
            if not bool(stream_file_enabled):
                return
            enqueue_t0 = time.perf_counter()
            try:
                max_blocks = int(os.getenv("SMARTBLOG_STREAM_FILE_QUEUE_BLOCKS", "16") or 16)
            except Exception:
                max_blocks = 16
            max_blocks = max(1, min(256, int(max_blocks)))
            event = None
            try:
                if frames_01.device.type == "cuda":
                    event = torch.cuda.Event()
                    torch.cuda.current_stream(device=frames_01.device).record_event(event)
            except Exception:
                event = None
            entry = {"tensor": frames_01.detach(), "event": event}
            with stream_file_cv:
                while len(stream_file_queue) >= int(max_blocks) and not bool(stream_file_stop.is_set()):
                    stream_file_cv.wait(timeout=0.05)
                if bool(stream_file_stop.is_set()):
                    return
                stream_file_queue.append(entry)
                stream_file_cv.notify_all()
            stream_file_enqueue_s += float(time.perf_counter() - enqueue_t0)

        def _stream_file_stop_writer(*, drain: bool) -> None:
            nonlocal stream_file_thread
            if stream_file_thread is None:
                return
            with stream_file_cv:
                if not bool(drain):
                    stream_file_queue.clear()
                _stream_file_write_progress(phase="encoding")
                stream_file_stop.set()
                stream_file_cv.notify_all()
            timeout = float(os.getenv("SMARTBLOG_STREAM_FILE_JOIN_TIMEOUT_SEC", "900") or 900)
            stream_file_thread.join(timeout=max(1.0, float(timeout)))
            if stream_file_thread.is_alive():
                raise RuntimeError("stream-file writer did not stop before timeout")
            stream_file_thread = None

        def _raw_notify() -> None:
            if raw_backlog_cv is None:
                return
            try:
                with raw_backlog_cv:
                    raw_backlog_cv.notify_all()
            except Exception:
                pass

        def _raw_make_bytes_entry(payload: bytes) -> dict[str, Any]:
            view = memoryview(payload)
            return {
                "view": view,
                "nbytes": int(len(view)),
                "frame_count": int(len(view) // raw_frame_bytes),
                "host_tensor": None,
                "np_array": None,
                "event": None,
                "pool_key": None,
            }

        def _raw_host_pool_key(shape: Any, dtype: Any) -> tuple[tuple[int, ...], str]:
            return (
                tuple(int(x) for x in tuple(shape)),
                str(dtype),
            )

        def _raw_acquire_host_tensor_like(rgb_gpu: torch.Tensor) -> tuple[torch.Tensor, tuple[tuple[int, ...], str]]:
            pool_key = _raw_host_pool_key(rgb_gpu.shape, rgb_gpu.dtype)
            if int(raw_host_tensor_pool_limit) > 0:
                with raw_backlog_cv:
                    pool = raw_host_tensor_pool.get(pool_key)
                    while pool:
                        try:
                            host_tensor = pool.popleft()
                        except Exception:
                            break
                        try:
                            if (
                                isinstance(host_tensor, torch.Tensor)
                                and host_tensor.device.type == "cpu"
                                and bool(host_tensor.is_pinned())
                                and tuple(int(x) for x in tuple(host_tensor.shape)) == tuple(int(x) for x in tuple(rgb_gpu.shape))
                                and host_tensor.dtype == rgb_gpu.dtype
                            ):
                                return host_tensor, pool_key
                        except Exception:
                            continue
            return torch.empty_like(rgb_gpu, device="cpu", pin_memory=True), pool_key

        def _raw_release_entry(entry: dict[str, Any] | None) -> None:
            if not isinstance(entry, dict):
                return
            event = entry.get("event")
            if event is not None:
                try:
                    event.synchronize()
                except Exception:
                    pass
            entry["event"] = None
            entry["view"] = None
            entry["np_array"] = None
            host_tensor = entry.get("host_tensor")
            entry["host_tensor"] = None
            if int(raw_host_tensor_pool_limit) <= 0 or host_tensor is None:
                return
            try:
                if (
                    (not isinstance(host_tensor, torch.Tensor))
                    or host_tensor.device.type != "cpu"
                    or (not bool(host_tensor.is_pinned()))
                ):
                    return
            except Exception:
                return
            pool_key = entry.get("pool_key")
            if not isinstance(pool_key, tuple):
                try:
                    pool_key = _raw_host_pool_key(host_tensor.shape, host_tensor.dtype)
                except Exception:
                    return
            with raw_backlog_cv:
                pool = raw_host_tensor_pool.get(pool_key)
                if pool is None:
                    pool = deque()
                    raw_host_tensor_pool[pool_key] = pool
                if len(pool) < int(raw_host_tensor_pool_limit):
                    pool.append(host_tensor)

        def _raw_make_async_entry(rgb_gpu: torch.Tensor) -> dict[str, Any]:
            nonlocal raw_copy_stream
            if raw_copy_stream is None:
                raw_copy_stream = torch.cuda.Stream(device=self.device)
            host_tensor, pool_key = _raw_acquire_host_tensor_like(rgb_gpu)
            current_stream = torch.cuda.current_stream(self.device)
            raw_copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(raw_copy_stream):
                host_tensor.copy_(rgb_gpu, non_blocking=True)
                event = raw_copy_stream.record_event()
            return {
                "view": None,
                "nbytes": int(host_tensor.numel()),
                "frame_count": int(host_tensor.shape[0]),
                "host_tensor": host_tensor,
                "np_array": None,
                "event": event,
                "pool_key": pool_key,
            }

        def _raw_materialize_entry(entry: dict[str, Any]) -> memoryview | None:
            view = entry.get("view")
            if view is not None:
                return view
            event = entry.get("event")
            if event is not None:
                event.synchronize()
                entry["event"] = None
            np_array = entry.get("np_array")
            if np_array is None:
                host_tensor = entry.get("host_tensor")
                if host_tensor is None:
                    return None
                np_array = host_tensor.view(-1).numpy()
                entry["np_array"] = np_array
            view = memoryview(np_array)
            entry["view"] = view
            return view

        def _write_raw_progress_marker(*, done: bool = False) -> None:
            nonlocal raw_progress_last_write_ts, raw_progress_last_snapshot
            payload = {
                "written_frames": int(max(0, int(raw_frames_streamed))),
                "enqueued_frames": int(max(0, int(raw_frames_enqueued))),
                "backlog_bytes": int(max(0, int(raw_backlog_bytes))),
                "prompt_mode": str(raw_prompt_mode or "speech"),
                "mode_seq": int(max(0, int(raw_prompt_mode_seq))),
                "mode_start_frame": int(max(0, int(raw_prompt_mode_start_frame))),
                "source_chunk_idx": int(max(0, int(raw_source_chunk_idx))),
                "source_chunk_start_frame": int(max(0, int(raw_source_chunk_start_frame))),
                "done": bool(done),
                "ts_ms": int(time.time() * 1000.0),
            }
            snapshot = (
                int(payload["written_frames"]),
                int(payload["enqueued_frames"]),
                int(payload["backlog_bytes"]),
                str(payload["prompt_mode"]),
                int(payload["mode_seq"]),
                int(payload["mode_start_frame"]),
                int(payload["source_chunk_idx"]),
                int(payload["source_chunk_start_frame"]),
                bool(payload["done"]),
            )
            if str(raw_transport_mode) == "shm_ring" and raw_shm is not None:
                try:
                    live_raw_shm_write_header(
                        raw_shm.buf,
                        written_frames=int(payload["written_frames"]),
                        enqueued_frames=int(payload["enqueued_frames"]),
                        backlog_bytes=int(payload["backlog_bytes"]),
                        prompt_mode=str(payload["prompt_mode"]),
                        mode_seq=int(payload["mode_seq"]),
                        mode_start_frame=int(payload["mode_start_frame"]),
                        source_chunk_idx=int(payload["source_chunk_idx"]),
                        source_chunk_start_frame=int(payload["source_chunk_start_frame"]),
                        done=bool(payload["done"]),
                    )
                except Exception:
                    pass
            if (not raw_progress_path) or (not bool(raw_progress_json_enabled)):
                raw_progress_last_snapshot = snapshot
                raw_progress_last_write_ts = float(time.perf_counter())
                return
            if not bool(done):
                now_ts = float(time.perf_counter())
                if raw_progress_last_snapshot == snapshot and (now_ts - float(raw_progress_last_write_ts)) < 0.25:
                    return
                if raw_progress_last_snapshot is not None and (now_ts - float(raw_progress_last_write_ts)) < 0.05:
                    return
            try:
                tmp = str(raw_progress_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                os.replace(tmp, raw_progress_path)
                raw_progress_last_write_ts = float(time.perf_counter())
                raw_progress_last_snapshot = snapshot
            except Exception:
                pass

        def _raw_start_writer() -> None:
            nonlocal raw_writer_thread, raw_writer_stop, raw_writer_error
            nonlocal raw_fd, raw_pipe_size_set, raw_backlog_bytes, raw_frames_streamed
            nonlocal raw_first_frame_marked
            if raw_pipe_path is None:
                return
            if raw_writer_thread is not None:
                return
            raw_writer_stop = threading.Event()
            raw_writer_error = None

            def _writer_main() -> None:
                nonlocal raw_fd, raw_pipe_size_set, raw_backlog_bytes, raw_frames_streamed
                nonlocal raw_first_frame_marked, raw_writer_error
                try:
                    while True:
                        if raw_writer_stop is not None and raw_writer_stop.is_set():
                            break

                        chunk0 = None
                        chunk_entry = None
                        with raw_backlog_cv:
                            while (
                                len(raw_backlog_chunks) <= 0
                                and not (raw_writer_stop is not None and raw_writer_stop.is_set())
                            ):
                                # Event-driven wait: avoid periodic wakeups that can add
                                # Python-thread jitter during first denoise block.
                                raw_backlog_cv.wait()
                            if raw_writer_stop is not None and raw_writer_stop.is_set():
                                break
                            if len(raw_backlog_chunks) > 0:
                                chunk_entry = raw_backlog_chunks[0]
                        if chunk_entry is None:
                            continue

                        chunk0 = _raw_materialize_entry(chunk_entry)
                        if chunk0 is None:
                            with raw_backlog_cv:
                                if len(raw_backlog_chunks) > 0 and raw_backlog_chunks[0] is chunk_entry:
                                    raw_backlog_bytes -= int(max(0, int(chunk_entry.get("nbytes") or 0)))
                                    if raw_backlog_bytes < 0:
                                        raw_backlog_bytes = 0
                                    raw_backlog_chunks.popleft()
                            _raw_release_entry(chunk_entry)
                            continue

                        if str(raw_transport_mode) == "shm_ring" and raw_shm is not None:
                            head_nbytes = int(max(0, int(chunk_entry.get("nbytes") or 0)))
                            if head_nbytes <= 0:
                                with raw_backlog_cv:
                                    if len(raw_backlog_chunks) > 0 and raw_backlog_chunks[0] is chunk_entry:
                                        raw_backlog_chunks.popleft()
                                _raw_release_entry(chunk_entry)
                                continue
                            frame_count = int(max(0, int(chunk_entry.get("frame_count") or 0)))
                            if frame_count <= 0:
                                frame_count = int(head_nbytes // max(1, int(raw_frame_bytes)))
                            write_frame_idx = int(max(0, int(raw_frames_streamed)))
                            start_slot = int(write_frame_idx % max(1, int(raw_shm_frame_capacity)))
                            first_frames = min(int(frame_count), int(raw_shm_frame_capacity) - int(start_slot))
                            first_nbytes = int(first_frames * int(raw_frame_bytes))
                            shm_buf = live_raw_shm_frame_region(raw_shm.buf)
                            if first_nbytes > 0:
                                start_byte = int(start_slot * int(raw_frame_bytes))
                                shm_buf[start_byte : start_byte + first_nbytes] = chunk0[:first_nbytes]
                            remain_nbytes = int(max(0, int(head_nbytes) - int(first_nbytes)))
                            if remain_nbytes > 0:
                                shm_buf[0:remain_nbytes] = chunk0[first_nbytes : first_nbytes + remain_nbytes]
                            with raw_backlog_cv:
                                if len(raw_backlog_chunks) > 0 and raw_backlog_chunks[0] is chunk_entry:
                                    raw_backlog_chunks.popleft()
                                    raw_backlog_bytes -= int(head_nbytes)
                                    raw_frames_streamed += int(frame_count)
                                    if raw_backlog_bytes < 0:
                                        raw_backlog_bytes = 0
                            _raw_release_entry(chunk_entry)
                            _write_raw_progress_marker(done=False)

                            if raw_frames_streamed > 0 and (not raw_first_frame_marked) and raw_ready_path:
                                try:
                                    with open(raw_ready_path, "w", encoding="utf-8") as rf:
                                        rf.write(str(int(time.time() * 1000.0)))
                                    raw_first_frame_marked = True
                                    stream_live_trace.note_first_raw_written(
                                        frames_streamed=int(raw_frames_streamed),
                                        backlog_bytes=int(raw_backlog_bytes),
                                    )
                                except Exception:
                                    pass
                            continue

                        if raw_fd is None:
                            try:
                                raw_fd = os.open(raw_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                                if (
                                    raw_fd is not None
                                    and (not raw_pipe_size_set)
                                    and raw_pipe_target_bytes > 0
                                    and fcntl is not None
                                ):
                                    try:
                                        cur_sz = 0
                                        try:
                                            cur_sz = int(fcntl.fcntl(raw_fd, fcntl.F_GETPIPE_SZ))
                                        except Exception:
                                            cur_sz = 0
                                        if cur_sz < int(raw_pipe_target_bytes):
                                            fcntl.fcntl(raw_fd, fcntl.F_SETPIPE_SZ, int(raw_pipe_target_bytes))
                                        try:
                                            new_sz = int(fcntl.fcntl(raw_fd, fcntl.F_GETPIPE_SZ))
                                        except Exception:
                                            new_sz = int(raw_pipe_target_bytes)
                                        print(
                                            f"Rank {rank}: live RAW pipe size={new_sz}B (target={raw_pipe_target_bytes}B)",
                                            flush=True,
                                        )
                                    except Exception as e:
                                        print(f"Rank {rank}: live RAW pipe resize failed: {e}", flush=True)
                                    raw_pipe_size_set = True
                            except OSError as e:
                                if e.errno not in (errno.ENXIO, errno.ENOENT, errno.EAGAIN, errno.EWOULDBLOCK):
                                    print(f"Rank {rank}: live RAW open failed (writer retry): {e}", flush=True)
                                raw_fd = None
                                time.sleep(0.01)
                                continue
                            except Exception as e:
                                print(f"Rank {rank}: live RAW open failed (writer retry): {e}", flush=True)
                                raw_fd = None
                                time.sleep(0.01)
                                continue

                        written = 0
                        try:
                            view = memoryview(chunk0)
                            written = int(os.write(raw_fd, view))
                        except BlockingIOError:
                            written = 0
                        except OSError as e:
                            if e.errno in (errno.EPIPE, errno.EBADF):
                                try:
                                    os.close(raw_fd)
                                except Exception:
                                    pass
                                raw_fd = None
                            else:
                                print(f"Rank {rank}: live RAW write failed: {e}", flush=True)
                            time.sleep(0.001)
                            continue
                        except Exception as e:
                            print(f"Rank {rank}: live RAW write failed: {e}", flush=True)
                            time.sleep(0.001)
                            continue

                        if written <= 0:
                            time.sleep(0.001)
                            continue

                        with raw_backlog_cv:
                            if len(raw_backlog_chunks) <= 0:
                                continue
                            head = raw_backlog_chunks[0]
                            head_nbytes = int(max(0, int(head.get("nbytes") or 0)))
                            if written >= head_nbytes:
                                raw_backlog_chunks.popleft()
                                raw_backlog_bytes -= int(head_nbytes)
                                raw_frames_streamed += int(head_nbytes // raw_frame_bytes)
                                released_entry = head
                            else:
                                remain_view = memoryview(chunk0)[written:]
                                head["view"] = remain_view
                                head["nbytes"] = int(max(0, head_nbytes - int(written)))
                                raw_backlog_bytes -= int(written)
                                raw_frames_streamed += int(written // raw_frame_bytes)
                                released_entry = None
                            if raw_backlog_bytes < 0:
                                raw_backlog_bytes = 0
                        if released_entry is not None:
                            _raw_release_entry(released_entry)
                        _write_raw_progress_marker(done=False)

                        if raw_frames_streamed > 0 and (not raw_first_frame_marked) and raw_ready_path:
                            try:
                                with open(raw_ready_path, "w", encoding="utf-8") as rf:
                                    rf.write(str(int(time.time() * 1000.0)))
                                raw_first_frame_marked = True
                                stream_live_trace.note_first_raw_written(
                                    frames_streamed=int(raw_frames_streamed),
                                    backlog_bytes=int(raw_backlog_bytes),
                                )
                            except Exception:
                                pass
                except Exception as e:
                    raw_writer_error = str(e)
                    print(f"Rank {rank}: live RAW writer failed: {e}", flush=True)
                finally:
                    try:
                        if raw_fd is not None:
                            os.close(raw_fd)
                    except Exception:
                        pass
                    raw_fd = None
                    _raw_notify()

            raw_writer_thread = threading.Thread(
                target=_writer_main,
                name=f"raw-writer-r{rank}",
                daemon=True,
            )
            raw_writer_thread.start()

        def _raw_stop_writer(*, drain: bool) -> None:
            nonlocal raw_writer_thread, raw_writer_stop
            if raw_writer_thread is None:
                return
            if raw_writer_stop is not None:
                if drain:
                    deadline = time.perf_counter() + 6.0
                    while True:
                        with raw_backlog_cv:
                            if len(raw_backlog_chunks) <= 0:
                                break
                        if time.perf_counter() >= deadline:
                            break
                        _raw_notify()
                        time.sleep(0.002)
                try:
                    raw_writer_stop.set()
                except Exception:
                    pass
            _raw_notify()
            try:
                raw_writer_thread.join(timeout=2.0)
            except Exception:
                pass
            raw_writer_thread = None

        if (raw_pipe_path is not None or raw_shm is not None) and rank == decode_rank:
            _raw_start_writer()
            _write_raw_progress_marker(done=False)

        if bool(stream_file_enabled):
            _stream_file_start_writer()

        if live_raw_dir and rank == decode_rank and _remote_edge_env_enabled():
            _remote_edge_start()

        #--------------------------------------Step 2: generate--------------------------------------
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            out = []
            # Streaming VAE keeps an internal causal-conv cache across stream_decode() calls.
            # Reset it per request; otherwise consecutive generations can "bleed" into each
            # other (e.g. the next video partially resembles the previous reference image).
            try:
                if hasattr(self.vae, "model") and hasattr(self.vae.model, "first_decode"):
                    self.vae.model.first_decode = True
            except Exception:
                pass
            request_cache_prepared = False
            active_nr = min(max_repeat, num_repeat)
            profile_active_clips = int(active_nr)
            stream_ref_latents_sync = (
                str(os.getenv("WORKER_STREAM_REF_LATENTS_SYNC", "0") or "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            stream_update_ref_latents = (
                str(os.getenv("LIVE_STREAM_UPDATE_REF_LATENTS", "1") or "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            stream_update_motion_latents = (
                str(os.getenv("LIVE_STREAM_UPDATE_MOTION_LATENTS", "1") or "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            stream_motion_latents_update_mode = str(
                os.getenv("LIVE_STREAM_UPDATE_MOTION_LATENTS_MODE", "latent") or "latent"
            ).strip().lower()
            if stream_motion_latents_update_mode in {"rgb", "image", "images", "decoded", "decode", "vae"}:
                stream_motion_latents_update_mode = "decoded"
            else:
                stream_motion_latents_update_mode = "latent"
            stream_file_motion_update = (
                str(os.getenv("LIVE_STREAM_UPDATE_MOTION_LATENTS_FOR_FILE", "0") or "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            stream_file_motion_update_mode = str(
                os.getenv("LIVE_STREAM_UPDATE_MOTION_LATENTS_FOR_FILE_MODE", "latent") or "latent"
            ).strip().lower()
            if stream_file_motion_update_mode in {"rgb", "image", "images", "decoded", "decode", "vae"}:
                stream_file_motion_update_mode = "decoded"
            else:
                stream_file_motion_update_mode = "latent"
            effective_motion_latents_update_mode = (
                str(stream_motion_latents_update_mode)
                if bool(stream_audio_mode)
                else str(stream_file_motion_update_mode)
            )
            stateful_motion_latents = bool(
                stream_update_motion_latents
                and (
                    bool(stream_audio_mode)
                    or (bool(stream_file_enabled) and bool(stream_file_motion_update))
                )
            )
            stream_update_motion_from_decoded = bool(
                stateful_motion_latents
                and effective_motion_latents_update_mode == "decoded"
            )
            stream_motion_latents_sync = (
                str(os.getenv("LIVE_STREAM_MOTION_LATENTS_SYNC", "1") or "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            try:
                stream_idle_seed_stride = int(os.getenv("WORKER_STREAM_IDLE_SEED_STRIDE", "64") or 64)
            except Exception:
                stream_idle_seed_stride = 64
            stream_idle_seed_stride = int(max(0, int(stream_idle_seed_stride)))
            if rank == 0 and ((not stream_audio_mode) or stream_timing_log):
                # Progress hint for long-running jobs (Gradio otherwise looks "stuck").
                print(
                    f"TPP generate start: clips={active_nr}, infer_frames={infer_frames}, "
                    f"steps={sampling_steps}, size={HEIGHT}x{WIDTH}, num_gpus_dit={num_gpus_dit}, "
                    f"vae_parallel={enable_vae_parallel} stream_mode={1 if stream_audio_mode else 0} "
                    f"ref_update={1 if stream_update_ref_latents else 0} "
                    f"motion_update={str(effective_motion_latents_update_mode)} "
                    f"motion_state={1 if stateful_motion_latents else 0}",
                    flush=True,
                )
            profile_loop_t0 = time.perf_counter()
            # Streaming clips can have different temporal lengths. Use a cumulative
            # block cursor for KV-cache positions instead of r * current_num_blocks;
            # otherwise short tails/variable render chunks reuse old temporal slots.
            stream_global_block_offset = 0
            for r in range(active_nr):
            #-------------------------------------------rollout loop------------------------------------------------------
                cancel_reason = ""
                if int(rank) == 0:
                    try:
                        raise_if_infer_cancelled(job_id=job_id)
                    except InferenceCancelled as e:
                        cancel_reason = str(e or "cancelled")
                cancel_obj = [cancel_reason]
                dist.broadcast_object_list(cancel_obj, src=0)
                cancel_reason = str(cancel_obj[0] or "").strip()
                if cancel_reason:
                    _remote_edge_abort_for_cancel(cancel_reason)
                    raise InferenceCancelled(cancel_reason)
                if rank == 0 and ((not stream_audio_mode) or stream_timing_log):
                    print(f"TPP clip {r+1}/{active_nr} start", flush=True)

                stream_clip_audio_input = None
                stream_clip_is_silence = False
                stream_clip_kind = "speech"
                stream_clip_source_chunk_idx = 0
                stream_clip_avatar_ref_path = ""
                stream_clip_pcm16le = b""
                stream_clip_sample_rate = int(stream_audio_tail_sample_rate)
                stream_clip_audible_samples = 0
                stream_clip_visible_start_frames = 0
                stream_clip_visible_frames = int(infer_frames)
                if stream_audio_mode:
                    if stream_audio_distributed_clip_broadcast:
                        stream_clip_done = False
                        stream_clip_error = ""
                        meta = None
                        if stream_audio_is_encode_rank:
                            # Source rank: resolve exactly one next clip (or done)
                            # then broadcast decision to all ranks.
                            if stream_audio_startup_error:
                                stream_clip_done = True
                                stream_clip_error = str(stream_audio_startup_error)
                            while (not stream_clip_done) and (not stream_clip_error):
                                if len(stream_audio_clips) <= 0:
                                    _stream_wait_for_audio_clips(
                                        min_required=1,
                                        block_until_ready=True,
                                    )
                                if len(stream_audio_clips) <= 0:
                                    # Reconcile marker race: producer may finish slightly before
                                    # done marker becomes visible.
                                    if (not stream_audio_done) and _stream_done_marker_exists():
                                        stream_audio_done = True
                                        _stream_wait_for_audio_clips(
                                            min_required=1,
                                            block_until_ready=False,
                                        )
                                if len(stream_audio_clips) <= 0:
                                    if stream_audio_producer_error:
                                        stream_clip_done = True
                                        stream_clip_error = f"liveaudio producer failed: {stream_audio_producer_error}"
                                        break
                                    if stream_audio_done:
                                        if bool(stream_audio_cancelled) and rank == 0 and stream_timing_log:
                                            print(f"TPP liveaudio cancelled at clip {r}", flush=True)
                                        stream_clip_done = True
                                        break
                                    if bool(stream_audio_is_always_on) and stream_audio_silence_clip is not None:
                                        stream_clip_audio_input = stream_audio_silence_clip
                                        stream_clip_is_silence = True
                                        stream_clip_kind = "filler"
                                        stream_clip_sample_rate = int(stream_audio_tail_sample_rate)
                                        stream_clip_pcm16le = _fit_pcm16le_to_frames(
                                            b"",
                                            sample_rate=int(stream_clip_sample_rate),
                                            frame_count=int(infer_frames),
                                        )
                                        stream_clip_audible_samples = 0
                                        stream_clip_visible_start_frames = 0
                                        stream_clip_visible_frames = int(infer_frames)
                                        break
                                    # Keep strict deterministic flow: wait until real clip or done.
                                    continue
                                (
                                    stream_clip_audio_input,
                                    stream_clip_kind,
                                    stream_clip_source_chunk_idx,
                                    stream_clip_avatar_ref_path,
                                    stream_clip_pcm16le,
                                    stream_clip_sample_rate,
                                    stream_clip_audible_samples,
                                    stream_clip_visible_start_frames,
                                    stream_clip_visible_frames,
                                ) = _stream_pop_audio_clip()
                                _stream_try_reply_boundary_prefill(min_required=1)
                                if stream_clip_audio_input is None:
                                    if stream_audio_producer_error:
                                        stream_clip_done = True
                                        stream_clip_error = f"liveaudio producer failed: {stream_audio_producer_error}"
                                        break
                                    if stream_audio_done:
                                        stream_clip_done = True
                                        break
                                    continue
                                if not bool(stream_first_clip_pop_logged):
                                    stream_first_clip_pop_dt = float(time.perf_counter() - stream_trace_t0)
                                    stream_first_clip_pop_logged = True
                                    stream_live_trace.note_first_clip(
                                        event="selected",
                                        q_after=int(len(stream_audio_clips)),
                                        is_silence=bool(stream_clip_is_silence),
                                        seen_chunks=int(stream_audio_seen_chunks),
                                    )
                                break

                            if stream_clip_done:
                                meta = {"done": 1}
                                if stream_clip_error:
                                    meta["error"] = str(stream_clip_error)
                            else:
                                meta = {
                                    "done": 0,
                                    "is_silence": 1 if stream_clip_is_silence else 0,
                                    "clip_kind": str(stream_clip_kind),
                                    "source_chunk_idx": int(max(0, int(stream_clip_source_chunk_idx))),
                                    "avatar_ref_path": str(stream_clip_avatar_ref_path or ""),
                                    "pcm16le": bytes(stream_clip_pcm16le or b""),
                                    "sample_rate": int(stream_clip_sample_rate),
                                    "audible_samples": int(stream_clip_audible_samples),
                                    "visible_start_frames": int(stream_clip_visible_start_frames),
                                    "visible_frames": int(stream_clip_visible_frames),
                                    "shape": [int(v) for v in list(stream_clip_audio_input.shape)],
                                }

                        obj = [meta]
                        dist.broadcast_object_list(obj, src=int(stream_audio_encode_rank))
                        meta_rx = obj[0] if isinstance(obj, list) and len(obj) > 0 else {}
                        meta_error = str((meta_rx or {}).get("error") or "").strip()
                        if meta_error:
                            raise RuntimeError(meta_error)
                        if bool((meta_rx or {}).get("done", 0)):
                            if rank == 0 and stream_timing_log:
                                print(f"TPP liveaudio completed at clip {r}", flush=True)
                            break

                        stream_clip_is_silence = bool((meta_rx or {}).get("is_silence", 0))
                        stream_clip_kind = normalize_stream_clip_kind((meta_rx or {}).get("clip_kind"))
                        stream_clip_source_chunk_idx = int(max(0, int((meta_rx or {}).get("source_chunk_idx") or 0)))
                        stream_clip_avatar_ref_path = str((meta_rx or {}).get("avatar_ref_path") or "").strip()
                        stream_clip_pcm16le = bytes((meta_rx or {}).get("pcm16le") or b"")
                        stream_clip_sample_rate = int(max(1, int((meta_rx or {}).get("sample_rate") or stream_audio_tail_sample_rate)))
                        stream_clip_audible_samples = int(max(0, int((meta_rx or {}).get("audible_samples") or 0)))
                        stream_clip_visible_start_frames = int(max(0, int((meta_rx or {}).get("visible_start_frames") or 0)))
                        stream_clip_visible_frames = int(max(0, int((meta_rx or {}).get("visible_frames") or 0)))
                        shape_rx = tuple(int(v) for v in ((meta_rx or {}).get("shape") or []))
                        if (not shape_rx) or any(int(v) <= 0 for v in shape_rx):
                            raise RuntimeError(f"Invalid liveaudio broadcast shape: {shape_rx}")

                        if not stream_audio_is_encode_rank:
                            stream_clip_audio_input = torch.empty(
                                shape_rx, device=self.device, dtype=self.param_dtype
                            )
                        else:
                            stream_clip_audio_input = stream_clip_audio_input.to(
                                device=self.device, dtype=self.param_dtype
                            ).contiguous()

                        dist.broadcast(stream_clip_audio_input, src=int(stream_audio_encode_rank))
                        stream_clip_audio_input = stream_clip_audio_input.contiguous()
                        if not bool(stream_first_clip_pop_logged):
                            stream_first_clip_pop_dt = float(time.perf_counter() - stream_trace_t0)
                            stream_first_clip_pop_logged = True
                            stream_live_trace.note_first_clip(
                                event="broadcast",
                                q_after=int(len(stream_audio_clips)),
                                is_silence=bool(stream_clip_is_silence),
                                seen_chunks=int(stream_audio_seen_chunks),
                                clip_shape=tuple(int(v) for v in list(stream_clip_audio_input.shape)),
                            )
                    else:
                        if len(stream_audio_clips) <= 0:
                            _stream_wait_for_audio_clips(
                                min_required=1,
                                block_until_ready=True,
                            )
                        if len(stream_audio_clips) <= 0:
                            # Best-effort reconcile marker race: if producer finished but marker
                            # arrived late, promote to done and flush residual tail once.
                            if (not stream_audio_done) and _stream_done_marker_exists():
                                stream_audio_done = True
                                _stream_wait_for_audio_clips(
                                    min_required=1,
                                    block_until_ready=False,
                                )
                        if len(stream_audio_clips) <= 0:
                            if stream_audio_producer_error:
                                raise RuntimeError(f"liveaudio producer failed: {stream_audio_producer_error}")
                            if stream_audio_done:
                                if bool(stream_audio_cancelled) and rank == 0:
                                    print(f"TPP liveaudio cancelled at clip {r}", flush=True)
                                if rank == 0:
                                    print(f"TPP liveaudio completed at clip {r}", flush=True)
                                break
                            if bool(stream_audio_is_always_on) and stream_audio_silence_clip is not None:
                                stream_clip_audio_input = stream_audio_silence_clip
                                stream_clip_is_silence = True
                                stream_clip_kind = "filler"
                                stream_clip_sample_rate = int(stream_audio_tail_sample_rate)
                                stream_clip_pcm16le = _fit_pcm16le_to_frames(
                                    b"",
                                    sample_rate=int(stream_clip_sample_rate),
                                    frame_count=int(infer_frames),
                                )
                                stream_clip_audible_samples = 0
                                stream_clip_visible_start_frames = 0
                                stream_clip_visible_frames = int(infer_frames)
                                if rank == 0:
                                    print(
                                        f"TPP liveaudio no chunk at clip {r}: running silent warm clip",
                                            flush=True,
                                    )
                            else:
                                time.sleep(float(stream_audio_poll_sec))
                                continue
                        if stream_clip_audio_input is None:
                            (
                                stream_clip_audio_input,
                                stream_clip_kind,
                                stream_clip_source_chunk_idx,
                                stream_clip_avatar_ref_path,
                                stream_clip_pcm16le,
                                stream_clip_sample_rate,
                                stream_clip_audible_samples,
                                stream_clip_visible_start_frames,
                                stream_clip_visible_frames,
                            ) = _stream_pop_audio_clip()
                            _stream_try_reply_boundary_prefill(min_required=1)
                        if stream_clip_audio_input is None:
                            if stream_audio_producer_error:
                                raise RuntimeError(f"liveaudio producer failed: {stream_audio_producer_error}")
                            if stream_audio_done:
                                if bool(stream_audio_cancelled) and rank == 0:
                                    print(f"TPP liveaudio cancelled at clip {r} (empty-pop)", flush=True)
                                if rank == 0:
                                    print(f"TPP liveaudio completed at clip {r} (empty-pop)", flush=True)
                                break
                            if bool(stream_audio_is_always_on) and stream_audio_silence_clip is not None:
                                stream_clip_audio_input = stream_audio_silence_clip
                                stream_clip_is_silence = True
                                stream_clip_kind = "filler"
                                stream_clip_sample_rate = int(stream_audio_tail_sample_rate)
                                stream_clip_pcm16le = _fit_pcm16le_to_frames(
                                    b"",
                                    sample_rate=int(stream_clip_sample_rate),
                                    frame_count=int(infer_frames),
                                )
                                stream_clip_audible_samples = 0
                                stream_clip_visible_start_frames = 0
                                stream_clip_visible_frames = int(infer_frames)
                            else:
                                time.sleep(float(stream_audio_poll_sec))
                                continue
                        if (
                            (stream_clip_audio_input is not None)
                            and (not bool(stream_first_clip_pop_logged))
                        ):
                            stream_first_clip_pop_dt = float(time.perf_counter() - stream_trace_t0)
                            stream_first_clip_pop_logged = True
                            stream_live_trace.note_first_clip(
                                event="selected",
                                q_after=int(len(stream_audio_clips)),
                                is_silence=bool(stream_clip_is_silence),
                                seen_chunks=int(stream_audio_seen_chunks),
                            )
                    if (
                        stream_audio_mode
                        and int(rank) == int(decode_rank)
                        and not bool(_remote_edge_latent_decode_delegated())
                    ):
                        _remote_edge_send_audio_until_for_frames(
                            int(remote_edge_video_frames_sent) + int(_remote_edge_audio_lead_frames())
                        )
                #----------------------------------------------Step 2.1: clip-level init------------------------------------------------------ 

                if bool(stream_audio_mode) and bool(stream_ref_latents_sync) and int(r) > 0 and int(active_nr) != 1:
                    ref_latents = ref_latents.contiguous()
                    if int(rank) != int(decode_rank):
                        ref_latents = torch.empty_like(ref_latents)
                    dist.broadcast(ref_latents, src=int(decode_rank))
                    ref_latents = ref_latents.contiguous()
                    if rank == 0 and ((not stream_audio_mode) or stream_timing_log):
                        print(
                            f"TPP ref_latents synchronized at clip {int(r)+1}",
                            flush=True,
                        )
                if (
                    bool(stateful_motion_latents)
                    and bool(stream_motion_latents_sync)
                    and int(r) > 0
                    and int(active_nr) != 1
                    and dist.is_initialized()
                ):
                    motion_latents = motion_latents.contiguous()
                    if int(rank) != int(decode_rank):
                        motion_latents = torch.empty_like(motion_latents)
                    dist.broadcast(motion_latents, src=int(decode_rank))
                    motion_latents = motion_latents.contiguous()
                    if rank == 0 and stream_timing_log:
                        print(
                            f"TPP motion_latents synchronized at clip {int(r)+1}",
                            flush=True,
                        )

                clip_infer_frames = int(infer_frames)
                if stream_audio_mode and (stream_clip_audio_input is not None):
                    try:
                        clip_t = int(stream_clip_audio_input.shape[-1])
                        if clip_t > 0:
                            clip_infer_frames = int(clip_t)
                    except Exception:
                        clip_infer_frames = int(infer_frames)
                if stream_audio_mode:
                    try:
                        if int(stream_clip_visible_frames) <= 0:
                            stream_clip_visible_frames = int(clip_infer_frames)
                    except Exception:
                        stream_clip_visible_frames = int(clip_infer_frames)
                    try:
                        stream_clip_visible_start_frames = int(
                            max(0, min(int(stream_clip_visible_start_frames), int(clip_infer_frames)))
                        )
                    except Exception:
                        stream_clip_visible_start_frames = 0
                    stream_clip_visible_frames = int(
                        max(
                            0,
                            min(
                                int(stream_clip_visible_frames),
                                int(clip_infer_frames) - int(stream_clip_visible_start_frames),
                            ),
                        )
                    )
                    stream_clip_pcm16le = _fit_pcm16le_to_frames(
                        stream_clip_pcm16le,
                        sample_rate=int(stream_clip_sample_rate),
                        frame_count=int(clip_infer_frames),
                    )
                stream_clip_segment_id = (
                    f"{str(job_id or 'job')}:{int(r)}:{int(stream_clip_source_chunk_idx)}"
                    if bool(stream_audio_mode)
                    else ""
                )
                clip_ref_switched = False
                if bool(stream_audio_mode) and pose_video is None:
                    next_ref_path = str(stream_clip_avatar_ref_path or "").strip()
                    if next_ref_path and os.path.exists(next_ref_path):
                        next_ref_path = os.path.abspath(next_ref_path)
                    else:
                        next_ref_path = ""
                    if next_ref_path and next_ref_path != str(stream_current_ref_image_path or ""):
                        ref_switch_t0 = time.perf_counter()
                        static_cond, static_cond_hit = self._get_static_reply_condition(
                            ref_image_path=next_ref_path,
                            size=size,
                            infer_frames=infer_frames,
                            drop_motion_noisy=bool(drop_motion_noisy),
                        )
                        ref_latents = static_cond["ref_latents"]
                        motion_latents = static_cond["motion_latents"]
                        videos_last_frames = static_cond.get("motion_frames_pixels", motion_latents).detach()
                        if bool(drop_motion_noisy):
                            zero_motion_latents = static_cond["zero_motion_latents"]
                        COND = [static_cond["cond_zero"]]
                        stream_current_ref_image_path = str(next_ref_path)
                        request_cache_prepared = False
                        stream_audio_last_prompt_mode = None
                        clip_ref_switched = True
                        if rank == 0:
                            print(
                                f"TPP liveaudio avatar ref switch clip={int(r)+1}/{int(active_nr)} "
                                f"chunk={int(stream_clip_source_chunk_idx)} "
                                f"ref={os.path.basename(str(next_ref_path))} "
                                f"cache={'hit' if static_cond_hit else 'miss'} "
                                f"dt={float(time.perf_counter() - ref_switch_t0):.3f}s",
                                flush=True,
                            )

                if bool(stream_audio_mode) and bool(clip_ref_switched):
                    stream_global_block_offset = 0

                if r == 0 or in_dit_device:
                    active_context = context
                    active_context_null = context_null
                    clip_prompt_mode = "speech"
                    clip_prefers_idle = False
                    clip_visual_prompt = ""
                    clip_visual_negative = ""
                    if stream_audio_mode and int(stream_clip_source_chunk_idx) > 0:
                        clip_visual_prompt, clip_visual_negative = _stream_chunk_visual_prompts(
                            int(stream_clip_source_chunk_idx)
                        )
                    if stream_audio_mode and idle_context is not None:
                        clip_prefers_idle = bool(stream_clip_prefers_idle_prompt(
                            clip_kind=str(stream_clip_kind),
                            is_silence=bool(stream_clip_is_silence),
                            enabled=bool(stream_audio_prompt_switch),
                        ))
                        if bool(clip_prefers_idle):
                            active_context = idle_context
                            active_context_null = idle_context_null
                            clip_prompt_mode = "idle"
                    if (
                        stream_audio_mode
                        and (not bool(clip_prefers_idle))
                        and (str(clip_visual_prompt or "").strip() or str(clip_visual_negative or "").strip())
                    ):
                        active_context, active_context_null, clip_prompt_mode = _stream_visual_context(
                            str(clip_visual_prompt or ""),
                            str(clip_visual_negative or ""),
                        )
                    clip_cfg_enabled = bool(tpp_cfg_enabled and active_context_null is not None)
                    cfg_batch_size = 2 if bool(clip_cfg_enabled) else 1
                    clip_prompt_mode_changed = str(stream_audio_last_prompt_mode or "") != str(clip_prompt_mode)
                    if bool(clip_prompt_mode_changed) and bool(request_cache_prepared):
                        self._reset_crossattn_cache()
                        if stream_timing_log and stream_audio_mode and rank == 0:
                            print(
                                f"TPP liveaudio prompt cache reset clip={int(r)+1}/{int(active_nr)} "
                                f"mode={str(clip_prompt_mode)}",
                                flush=True,
                            )
                    if stream_timing_log and stream_audio_mode and rank == 0 and str(stream_audio_last_prompt_mode or "") != str(clip_prompt_mode):
                        print(
                            f"TPP liveaudio prompt mode clip={int(r)+1}/{int(active_nr)} mode={str(clip_prompt_mode)} "
                            f"kind={str(stream_clip_kind)} silence={1 if stream_clip_is_silence else 0} "
                            f"cfg={1 if clip_cfg_enabled else 0} batch={int(cfg_batch_size)}",
                            flush=True,
                        )
                    stream_audio_last_prompt_mode = str(clip_prompt_mode)
                    seed_g = torch.Generator(device=self.device)
                    seed_offset = int(r)
                    if stream_audio_mode and bool(clip_prefers_idle):
                        idle_seed_source = int(stream_clip_source_chunk_idx) if int(stream_clip_source_chunk_idx) > 0 else int(r)
                        if int(stream_idle_seed_stride) > 0:
                            seed_offset = int(idle_seed_source) // int(stream_idle_seed_stride)
                        else:
                            seed_offset = int(idle_seed_source)
                    seed_g.manual_seed(seed + int(seed_offset))

                    lat_target_frames = (clip_infer_frames + 3 + self.motion_frames
                                        ) // 4 - lat_motion_frames
                    target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                    frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
                    cond_cache_size = self._estimate_cond_cache_size(HEIGHT, WIDTH)
                    clip_noise = [
                        torch.randn(
                            16,
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g)
                    ]
                    max_seq_len = np.prod(target_shape) // 4
                    kv_cache_size = int(max_seq_len)
                    kv_cap_frames = 0
                    kv_effective_latent_frames = int(target_shape[0])
                    try:
                        kv_cap_requested = int(os.getenv("LIVE_STREAM_KV_CACHE_FRAMES", "0") or 0)
                    except Exception:
                        kv_cap_requested = 0
                    if stream_audio_mode or int(kv_cap_requested) > 0:
                        kv_cache_size, kv_cap_frames, kv_effective_latent_frames = self._resolve_stream_kv_cache_size(
                            max_seq_len=int(max_seq_len),
                            frame_seq_length=int(frame_seq_length),
                            num_frames_per_block=int(self.num_frames_per_block),
                        )
                    kv_cache_shape_key = (
                        int(HEIGHT),
                        int(WIDTH),
                        int(target_shape[0]),
                        int(target_shape[1]),
                        int(target_shape[2]),
                        int(frame_seq_length),
                        int(cond_cache_size),
                        int(kv_effective_latent_frames),
                    )
                    if not bool(request_cache_prepared):
                        cache_t0 = time.perf_counter()
                        local_rank = torch.distributed.get_rank()
                        cache_need_reinit = False
                        step_ids = tuple()
                        if local_rank < num_gpus_dit:
                            num_steps = int(sampling_steps)
                            if not bool(getattr(self, "joint_sp_denoise", False)):
                                assert num_steps >= num_gpus_dit, (
                                    f"sampling_steps ({num_steps}) must be >= num_gpus_dit ({num_gpus_dit})"
                                )

                            step_start, step_end = _resolve_tpp_step_range_for_rank(
                                step_rank=int(local_rank),
                                num_steps=int(num_steps),
                                num_gpus_dit=int(num_gpus_dit),
                                joint_sp_denoise=bool(getattr(self, "joint_sp_denoise", False)),
                            )
                            step_ids = tuple(int(s) for s in range(step_start, step_end))

                            has_kv = isinstance(getattr(self, "kv_cache_by_step", None), dict)
                            same_steps = tuple(getattr(self, "_kv_cache_step_ids", tuple())) == step_ids
                            same_kv_size = int(getattr(self, "_kv_cache_size", 0) or 0) == int(kv_cache_size)
                            same_cond_size = int(getattr(self, "_kv_cache_cond_cache_size", 0) or 0) == int(cond_cache_size)
                            same_shape = tuple(getattr(self, "_kv_cache_shape_key", tuple())) == tuple(kv_cache_shape_key)
                            same_batch = int(getattr(self, "_kv_cache_batch_size", 0) or 0) == int(cfg_batch_size)

                            if not (has_kv and same_steps and same_kv_size and same_cond_size and same_shape and same_batch):
                                cache_need_reinit = True
                                previous_shape = tuple(getattr(self, "_kv_cache_shape_key", tuple()))
                                # Drop old step caches before allocating the replacement. A warmup with
                                # a different step count can otherwise keep both caches live briefly and
                                # OOM even when the new request is smaller.
                                self._release_stream_attention_caches(
                                    reason=(
                                        "kv_profile_changed "
                                        f"old_steps={list(getattr(self, '_kv_cache_step_ids', tuple()))} "
                                        f"new_steps={list(step_ids)} "
                                        f"old_kv={int(getattr(self, '_kv_cache_size', 0) or 0)} "
                                        f"new_kv={int(kv_cache_size)} "
                                        f"old_cond={int(getattr(self, '_kv_cache_cond_cache_size', 0) or 0)} "
                                        f"new_cond={int(cond_cache_size)} "
                                        f"old_batch={int(getattr(self, '_kv_cache_batch_size', 0) or 0)} "
                                        f"new_batch={int(cfg_batch_size)}"
                                    ),
                                    clear_model_precompute=bool(previous_shape and previous_shape != tuple(kv_cache_shape_key)),
                                )
                                self._initialize_kv_cache_by_steps(
                                    step_ids=step_ids,
                                    batch_size=int(cfg_batch_size),
                                    dtype=self.param_dtype,
                                    device=f"cuda:{local_rank}",
                                    kv_cache_size=kv_cache_size,
                                    cond_cache_size=cond_cache_size,
                                    shape_key=kv_cache_shape_key,
                                )
                            else:
                                self._reset_kv_cache_by_steps()

                        has_cross = isinstance(getattr(self, "crossattn_cache", None), list)
                        same_cross_batch = int(getattr(self, "_crossattn_cache_batch_size", 0) or 0) == int(cfg_batch_size)
                        if (not has_cross) or (not same_cross_batch):
                            cache_need_reinit = True
                            self._initialize_crossattn_cache(
                                batch_size=int(cfg_batch_size),
                                dtype=self.param_dtype,
                                device=self.device
                            )
                        else:
                            self._reset_crossattn_cache()

                        request_cache_prepared = True
                        if stream_timing_log:
                            print(
                                f"TPP cache prep rank={rank} mode={'init' if cache_need_reinit else 'reuse'} "
                                f"dt={float(time.perf_counter() - cache_t0):.3f}s "
                                f"kv_size={int(getattr(self, '_kv_cache_size', 0) or 0)} "
                                f"kv_max={int(max_seq_len)} kv_cap_frames={int(kv_cap_frames)} "
                                f"kv_latent_frames={int(kv_effective_latent_frames)} "
                                f"shape={int(HEIGHT)}x{int(WIDTH)} "
                                f"step_ids={list(step_ids) if len(step_ids) > 0 else '-'} "
                                f"clip={int(r)+1}/{int(active_nr)} batch={int(cfg_batch_size)} "
                                f"cfg={1 if clip_cfg_enabled else 0}",
                                flush=True,
                            )


                #----------------------------------------------Step 2.2: prepare clip-level cond---------------------------------
                if r==0 or in_dit_device:
                    clip_prepare_t0 = time.perf_counter()
                    clip_latents = [clip_noise[0].clone()]
                    with torch.no_grad():
                        left_idx = r * clip_infer_frames
                        right_idx = r * clip_infer_frames + clip_infer_frames
                        cond_latents = COND[r] if pose_video else COND[0] * 0
                        cond_latents = cond_latents.to(
                            dtype=self.param_dtype, device=self.device)
                        if stream_audio_mode:
                            # Static avatar conditioning is cached at the baseline
                            # infer length. B300 render one-pass can feed longer
                            # audio clips (for example 128/192 frames) to avoid
                            # tail mini-clips before avatar boundaries, so the
                            # conditioning tensor must be expanded to the actual
                            # clip latent length before block slicing. Otherwise
                            # later blocks slice an empty temporal window and
                            # cond_encoder fails with T=0.
                            try:
                                target_cond_t = int(target_shape[0])
                                cond_t = int(cond_latents.shape[2])
                            except Exception:
                                target_cond_t = 0
                                cond_t = 0
                            if int(target_cond_t) > 0 and int(cond_t) != int(target_cond_t):
                                if int(cond_t) > int(target_cond_t):
                                    cond_latents = cond_latents[:, :, : int(target_cond_t)].contiguous()
                                else:
                                    pad_t = int(target_cond_t) - int(cond_t)
                                    if int(cond_t) > 0 and pose_video:
                                        pad = cond_latents[:, :, -1:, :, :].expand(
                                            -1, -1, int(pad_t), -1, -1
                                        ).contiguous()
                                    else:
                                        pad_shape = list(cond_latents.shape)
                                        if len(pad_shape) >= 3:
                                            pad_shape[2] = int(pad_t)
                                        pad = torch.zeros(
                                            *pad_shape,
                                            device=cond_latents.device,
                                            dtype=cond_latents.dtype,
                                        )
                                    cond_latents = torch.cat([cond_latents, pad], dim=2).contiguous()
                                if rank == 0 and stream_timing_log:
                                    print(
                                        f"TPP liveaudio cond temporal adjusted clip={int(r)+1}/{int(active_nr)} "
                                        f"cond_t={int(cond_t)} target_t={int(target_cond_t)}",
                                        flush=True,
                                    )
                        if stream_audio_mode:
                            assert stream_clip_audio_input is not None
                            audio_input = stream_clip_audio_input.to(device=self.device, dtype=self.param_dtype)
                        else:
                            audio_input = audio_emb[..., left_idx:right_idx]
                    input_motion_latents = motion_latents.clone()
                    if stream_audio_mode and int(r) == 0:
                        stream_live_trace.note_first_clip_prepare(
                            prep_dt=float(time.perf_counter() - clip_prepare_t0),
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )

                    # if offload_model or self.init_on_cpu:
                    #     self.noise_model.to(self.device)
                    #     torch.cuda.empty_cache()

                #-----------------------------------------------Temporal denoising loop in single clip---------------------------------
                # 2.2.0 prefill cond caching
                # Prefill once at the first clip. Running it again at clip-2 adds a visible seam
                # after the first chunk without improving stability.
                if (r == 0 or bool(clip_ref_switched)) and in_dit_device: # prefill clean KV cache for all DiT ranks
                    prefill_t0 = time.perf_counter()
                    block_index = 0
                    block_latents = clip_latents[0][:, block_index *
                                    self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    left_idx = block_index * (self.num_frames_per_block * 4)
                    right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                    prefill_audio_input = audio_input[..., left_idx:right_idx]
                    block_arg_c = {
                        'context': _cfg_context_batch(active_context, active_context_null, bool(clip_cfg_enabled)),
                        'seq_len': None,
                        'cond_states': _cfg_batch_tensor(
                            cond_latents[:, :, block_index *
                                         self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block],
                            bool(clip_cfg_enabled),
                        ),
                        "motion_latents": _cfg_batch_tensor(input_motion_latents, bool(clip_cfg_enabled)),
                        'ref_latents': _cfg_batch_tensor(ref_latents, bool(clip_cfg_enabled)),
                        "audio_input": _cfg_batch_tensor(
                            prefill_audio_input,
                            bool(clip_cfg_enabled),
                            zero_uncond=True,
                        ),
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                        "sink_flag": True,
                    }
                    timestep = torch.ones(
                        [1, self.num_frames_per_block], device=self.device, dtype=self.param_dtype) * 0
                    if stream_audio_mode and stream_phase_sync_debug and self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    if stream_audio_mode:
                        stream_live_trace.note_first_prefill_phase(
                            phase="setup",
                            phase_dt=float(time.perf_counter() - prefill_t0),
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )
                    prefill_call_t0 = time.perf_counter()
                    self.noise_model( #update clean kv cache
                        _cfg_latent_inputs(block_latents, bool(clip_cfg_enabled)),
                        t=_cfg_timestep_batch(timestep * 0, bool(clip_cfg_enabled)),
                        **block_arg_c,
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frames_per_block * frame_seq_length,
                        current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length)
                    if stream_audio_mode and stream_phase_sync_debug and self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    if stream_audio_mode:
                        stream_live_trace.note_first_prefill_phase(
                            phase="noise_model",
                            phase_dt=float(time.perf_counter() - prefill_call_t0),
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )
                        stream_live_trace.note_first_prefill(
                            prefill_dt=float(time.perf_counter() - prefill_t0),
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )
                        


                num_blocks = target_shape[0] // self.num_frames_per_block
                profile_last_num_blocks = int(num_blocks)
                clip_global_block_offset = (
                    int(stream_global_block_offset)
                    if bool(stream_audio_mode)
                    else int(r) * int(num_blocks)
                )
                if stream_timing_log and stream_audio_mode and rank == 0:
                    print(
                        f"TPP liveaudio temporal offset clip={int(r)+1}/{int(active_nr)} "
                        f"blocks={int(num_blocks)} offset_blocks={int(clip_global_block_offset)} "
                        f"ref_switched={1 if bool(clip_ref_switched) else 0}",
                        flush=True,
                    )
                for block_index in range(num_blocks):
                    block_t0 = time.perf_counter()
                    first_block_phase_trace = bool(
                        stream_timing_log and stream_audio_mode and int(r) == 0 and int(block_index) == 0
                    )
                    if first_block_phase_trace:
                        stream_live_trace.note_first_block_start(
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )
                    recv_wait_dt = 0.0
                    denoise_total_dt = 0.0
                    send_wait_dt = 0.0
                    steps_executed = 0
                    vae_recv_dt = 0.0
                    decode_dt = 0.0
                    rgb_pack_dt = 0.0
                    cpu_pack_dt = 0.0
                    raw_enqueue_dt = 0.0
                    raw_write_dt = 0.0
                    raw_written_delta = 0
                    raw_backlog_before = int(raw_backlog_bytes) if rank == decode_rank else 0
                    raw_backlog_after = raw_backlog_before
                    raw_frames_enq_delta = 0
                    if stream_audio_mode and stream_audio_refill_during_denoise and (not stream_audio_async_producer):
                        # Optional background refill during denoise; keep throttled to avoid
                        # introducing periodic jitter in frame generation cadence.
                        if (int(block_index) % int(stream_audio_refill_block_interval)) == 0:
                            _stream_refill_audio_clips(min_required=1, block_until_ready=False)
                    if rank == 0 and stream_timing_log:
                        print(
                            f"TPP clip {r+1}/{active_nr} block {block_index+1}/{num_blocks} start",
                            flush=True,
                        )
                    # 2.2.1 prepare block-level cond
                    block_setup_t0 = time.perf_counter()
                    # Cache scheduler timesteps/sigmas, but recompute if sampling_steps
                    # or shift changes.
                    # Gradio sliders can change sampling_steps between requests; without this guard,
                    # the first request "locks" the scheduler length for the lifetime of the process.
                    cached_steps = getattr(self, "_sampler_num_steps", None)
                    cached_shift = getattr(self, "_sampler_shift", None)
                    shift_f = float(shift)
                    if (
                        (getattr(self, "_sampler_timesteps", None) is None)
                        or (cached_steps != int(sampling_steps))
                        or (cached_shift is None)
                        or (abs(float(cached_shift) - float(shift_f)) > 1e-6)
                    ):
                        sample_scheduler.set_timesteps(int(sampling_steps), device=self.device)
                        self._sampler_timesteps = sample_scheduler.timesteps
                        self._sampler_sigmas = sample_scheduler.sigmas
                        self._sampler_num_steps = int(sampling_steps)
                        self._sampler_shift = float(shift_f)

                    timesteps = self._sampler_timesteps
                    sample_scheduler.timesteps = timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._begin_index = 0
                    timestep_blocks_key = (
                        int(sampling_steps),
                        f"{float(shift):.6f}",
                        int(self.num_frames_per_block),
                        str(self.param_dtype),
                        str(self.device),
                    )
                    if (
                        getattr(self, "_sampler_timestep_blocks", None) is None
                        or getattr(self, "_sampler_timestep_blocks_key", None) != timestep_blocks_key
                    ):
                        self._sampler_timestep_blocks = tuple(
                            torch.as_tensor(timesteps[j], device=self.device, dtype=self.param_dtype)
                            .reshape(1, 1)
                            .expand(1, int(self.num_frames_per_block))
                            .contiguous()
                            for j in range(len(timesteps))
                        )
                        self._sampler_timestep_blocks_key = timestep_blocks_key
                    timestep_blocks = self._sampler_timestep_blocks

                    block_latents = clip_latents[0][:, block_index *
                                self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    if r==0 or in_dit_device:
                        left_idx = block_index * (self.num_frames_per_block * 4)
                        right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                        block_audio_input = audio_input[..., left_idx:right_idx]
                        block_arg_c = {
                            'context': _cfg_context_batch(active_context, active_context_null, bool(clip_cfg_enabled)),
                            'seq_len': None,
                            'cond_states': _cfg_batch_tensor(
                                cond_latents[:, :, block_index *
                                             self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block],
                                bool(clip_cfg_enabled),
                            ),
                            "motion_latents": _cfg_batch_tensor(input_motion_latents, bool(clip_cfg_enabled)),
                            'ref_latents': _cfg_batch_tensor(ref_latents, bool(clip_cfg_enabled)),
                            "audio_input": _cfg_batch_tensor(
                                block_audio_input,
                                bool(clip_cfg_enabled),
                                zero_uncond=True,
                            ),
                            "motion_frames": [self.motion_frames, lat_motion_frames],
                            "drop_motion_frames": drop_first_motion and r == 0,
                        }
                    if stream_timing_log and stream_audio_mode and int(r) == 0 and int(block_index) == 0:
                        stream_live_trace.note_first_block_setup(
                            setup_dt=float(time.perf_counter() - block_setup_t0),
                            q_depth=int(len(stream_audio_clips)),
                            done=bool(stream_audio_done),
                        )

                    num_steps = len(timesteps)
                    if not bool(getattr(self, "joint_sp_denoise", False)):
                        assert num_steps >= num_gpus_dit, (
                            f"sampling_steps ({num_steps}) must be >= num_gpus_dit ({num_gpus_dit})"
                        )

                    if in_dit_device:
                        step_start, step_end = _resolve_tpp_step_range_for_rank(
                            step_rank=int(rank),
                            num_steps=int(num_steps),
                            num_gpus_dit=int(num_gpus_dit),
                            joint_sp_denoise=bool(getattr(self, "joint_sp_denoise", False)),
                        )
                        assert step_end > step_start, (
                            f"Rank {rank} got empty step range for num_steps={num_steps}, num_gpus_dit={num_gpus_dit}"
                        )

                        # Receive the stage input once (for this block).
                        recv_wait_t0 = time.perf_counter()
                        if self.src_gpu is None:
                            latent_model_input = block_latents
                            recv_wait_dt = 0.0
                        else:
                            latent_model_input = torch.empty_like(block_latents)
                            dist.recv(latent_model_input, self.src_gpu)
                            if first_block_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            recv_wait_dt = float(time.perf_counter() - recv_wait_t0)
                        if first_block_phase_trace:
                            stream_live_trace.note_first_block_step_phase(
                                step_idx=int(step_start + 1),
                                total_steps=int(num_steps),
                                phase="recv",
                                phase_dt=float(recv_wait_dt),
                                q_depth=int(len(stream_audio_clips)),
                                done=bool(stream_audio_done),
                            )
                        if stream_timing_log and recv_wait_dt >= float(stream_block_log_slow_sec):
                            print(
                                f"TPP timing rank={rank} clip={r+1}/{active_nr} "
                                f"block={block_index+1}/{num_blocks} recv_wait={recv_wait_dt:.3f}s "
                                f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0}",
                                flush=True,
                            )

                        # Run this stage's steps sequentially.
                        for i in range(step_start, step_end):
                            denoise_t0 = time.perf_counter()
                            t = timesteps[i]
                            sample_scheduler._step_index = i
                            first_step_phase_trace = bool(
                                stream_timing_log
                                and stream_audio_mode
                                and int(r) == 0
                                and int(block_index) == 0
                                and int(i) == int(step_start)
                            )

                            timestep = timestep_blocks[i]

                            kv_cache_i = (
                                self.kv_cache_by_step.get(int(i), None)
                                if isinstance(getattr(self, "kv_cache_by_step", None), dict)
                                else None
                            )
                            if kv_cache_i is None:
                                raise RuntimeError(
                                    f"Missing per-step KV cache for step {int(i)}; "
                                    "strict mode disallows legacy single-cache path."
                                )

                            if stream_step_trace:
                                print(
                                    f"TPP step enter rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} step={i+1}/{num_steps} "
                                    f"phase=noise_model",
                                    flush=True,
                                )
                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            noise_model_t0 = time.perf_counter()
                            noise_pred_cond = self.noise_model(
                                _cfg_latent_inputs(latent_model_input, bool(clip_cfg_enabled)),
                                t=_cfg_timestep_batch(timestep, bool(clip_cfg_enabled)),
                                **block_arg_c,
                                kv_cache=kv_cache_i, crossattn_cache=self.crossattn_cache,
                                current_start=(int(clip_global_block_offset) + int(block_index)) * self.num_frames_per_block * frame_seq_length,
                                current_end=(int(clip_global_block_offset) + int(block_index) + 1) * self.num_frames_per_block * frame_seq_length,
                                mask=mask)
                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            noise_model_dt = float(time.perf_counter() - noise_model_t0)
                            if stream_step_trace:
                                print(
                                    f"TPP step exit  rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} step={i+1}/{num_steps} "
                                    f"phase=noise_model",
                                    flush=True,
                                )
                            if first_step_phase_trace:
                                stream_live_trace.note_first_block_step_phase(
                                    step_idx=int(i + 1),
                                    total_steps=int(num_steps),
                                    phase="noise_model",
                                    phase_dt=float(noise_model_dt),
                                    q_depth=int(len(stream_audio_clips)),
                                    done=bool(stream_audio_done),
                                )

                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            noise_cat_t0 = time.perf_counter()
                            noise_pred_batch = _noise_output_batch_tensor(noise_pred_cond)
                            if bool(clip_cfg_enabled):
                                if int(noise_pred_batch.shape[0]) < 2:
                                    raise RuntimeError("TPP CFG expected noise batch with cond/uncond entries")
                                noise_pred_uncond = noise_pred_batch[0]
                                noise_pred_guided_cond = noise_pred_batch[1]
                                noise_pred_tensor = noise_pred_uncond + guide_scale_value * (
                                    noise_pred_guided_cond - noise_pred_uncond
                                )
                            else:
                                noise_pred_tensor = noise_pred_batch[0]
                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            noise_cat_dt = float(time.perf_counter() - noise_cat_t0)
                            noise_pred = [noise_pred_tensor]
                            if first_step_phase_trace:
                                print(
                                    f"TPP first-block-step-meta rank={rank} step={i+1}/{num_steps} "
                                    f"noise_parts={int(noise_pred_batch.shape[0])} cfg={1 if clip_cfg_enabled else 0} "
                                    f"shape={list(noise_pred_tensor.shape)}",
                                    flush=True,
                                )
                                stream_live_trace.note_first_block_step_phase(
                                    step_idx=int(i + 1),
                                    total_steps=int(num_steps),
                                    phase="noise_cat",
                                    phase_dt=float(noise_cat_dt),
                                    q_depth=int(len(stream_audio_clips)),
                                    done=bool(stream_audio_done),
                                )

                            if stream_step_trace:
                                print(
                                    f"TPP step enter rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} step={i+1}/{num_steps} "
                                    f"phase=scheduler_step",
                                    flush=True,
                                )
                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            scheduler_step_t0 = time.perf_counter()
                            temp_x0 = sample_scheduler.step(
                                noise_pred[0].unsqueeze(0),
                                t,
                                latent_model_input.unsqueeze(0),
                                return_dict=False,
                                generator=seed_g)[0]
                            if first_step_phase_trace and stream_phase_sync_debug and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            scheduler_step_dt = float(time.perf_counter() - scheduler_step_t0)
                            if stream_step_trace:
                                print(
                                    f"TPP step exit  rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} step={i+1}/{num_steps} "
                                    f"phase=scheduler_step",
                                    flush=True,
                                )
                            if first_step_phase_trace:
                                stream_live_trace.note_first_block_step_phase(
                                    step_idx=int(i + 1),
                                    total_steps=int(num_steps),
                                    phase="scheduler_step",
                                    phase_dt=float(scheduler_step_dt),
                                    q_depth=int(len(stream_audio_clips)),
                                    done=bool(stream_audio_done),
                                )
                            latent_model_input = temp_x0.squeeze(0)
                            denoise_dt = float(time.perf_counter() - denoise_t0)
                            denoise_total_dt += float(denoise_dt)
                            steps_executed += 1
                            if stream_timing_log and stream_audio_mode and int(r) == 0 and int(block_index) == 0:
                                print(
                                    f"TPP first-block-step rank={rank} step={i+1}/{num_steps} "
                                    f"denoise={denoise_dt:.3f}s recv={float(recv_wait_dt):.3f}s "
                                    f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0}",
                                    flush=True,
                                )
                            if stream_timing_log and denoise_dt >= float(stream_block_log_slow_sec):
                                print(
                                    f"TPP timing rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} step={i+1}/{num_steps} "
                                    f"denoise={denoise_dt:.3f}s q={int(len(stream_audio_clips))} "
                                    f"done={1 if stream_audio_done else 0}",
                                    flush=True,
                                )

                        block_latents = latent_model_input
                        if self.tgt_gpu is not None:
                            send_t0 = time.perf_counter()
                            dist.send(block_latents.contiguous(), self.tgt_gpu)
                            send_dt = float(time.perf_counter() - send_t0)
                            send_wait_dt = float(send_dt)
                            if stream_timing_log and send_dt >= float(stream_block_log_slow_sec):
                                print(
                                    f"TPP timing rank={rank} clip={r+1}/{active_nr} "
                                    f"block={block_index+1}/{num_blocks} send_wait={send_dt:.3f}s "
                                    f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0}",
                                    flush=True,
                                )
                        # Rank-local core timing (without decode/write path) for diagnosing
                        # first-block spikes on non-decode ranks as well.
                        core_dt = float(time.perf_counter() - block_t0)
                        if stream_timing_log or (
                            stream_audio_mode and core_dt >= float(stream_block_log_slow_sec)
                        ):
                            print(
                                f"TPP block core rank={rank} clip={r+1}/{active_nr} "
                                f"block={block_index+1}/{num_blocks} total={core_dt:.3f}s "
                                f"recv={float(recv_wait_dt):.3f}s denoise={float(denoise_total_dt):.3f}s "
                                f"steps={int(steps_executed)} send={float(send_wait_dt):.3f}s "
                                f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0}",
                                flush=True,
                            )
                        _tpp_log_stage_stats(
                            core_dt=float(core_dt),
                            recv_wait_dt=float(recv_wait_dt),
                            denoise_total_dt=float(denoise_total_dt),
                            send_wait_dt=float(send_wait_dt),
                            steps_executed=int(steps_executed),
                            step_start=int(step_start),
                            step_end=int(step_end),
                            num_steps=int(num_steps),
                            clip_idx=int(r),
                            active_clips=int(active_nr),
                            block_index=int(block_index),
                            num_blocks=int(num_blocks),
                        )
                        profile_dit_blocks += 1
                        profile_steps += int(steps_executed)
                        profile_core_s += float(core_dt)
                        profile_recv_s += float(recv_wait_dt)
                        profile_denoise_s += float(denoise_total_dt)
                        profile_send_s += float(send_wait_dt)
                        if (
                            stream_audio_mode
                            and bool(stream_audio_defer_async_start)
                            and stream_audio_is_encode_rank
                            and int(r) == 0
                            and int(block_index) == 0
                        ):
                            _stream_start_async_producer()

                    if enable_vae_parallel and rank == decode_rank:
                        vae_recv_t0 = time.perf_counter()
                        block_latents = torch.empty_like(block_latents)
                        dist.recv(block_latents, self.src_gpu)
                        torch.cuda.synchronize()
                        vae_recv_dt = float(time.perf_counter() - vae_recv_t0)
                        if vae_recv_dt < 0.01:
                            print(f"WARNING: VAE serves as a bottleneck!")

                    #----------------------------------------------Step 2.3: block-level postprocess for vae---------------------------------
                    if rank == decode_rank:
                        if offload_model:
                            # Only makes sense when VAE runs on a dedicated GPU.
                            if enable_vae_parallel:
                                print(f"offloading model to cpu")
                                self.noise_model.cpu()
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        if bool(stream_update_ref_latents) and active_nr != 1:
                            if block_index == 0:  # cache new ref (attention sink anchor)
                                # NOTE: Do NOT broadcast here. Collectives must be called by all ranks
                                # in the same order; broadcasting only on decode_rank can silently
                                # desync and cause later clips to pick up stale refs from a previous
                                # request.
                                #
                                # The updated ref_latents (if any) will be synchronized by all
                                # ranks at the start of the next clip.
                                ref_latents = block_latents.unsqueeze(0)[:, :, 0:1]
                                if stream_timing_log and stream_audio_mode:
                                    print(
                                        f"TPP ref_latents updated clip={int(r)+1}/{int(active_nr)} "
                                        f"block={int(block_index)+1}/{int(num_blocks)}",
                                        flush=True,
                                    )

                        # decode to rgb
                        decode_t0 = time.perf_counter()
                        remote_edge_only = False
                        motion_decoded_block = None
                        if (r == 0 or bool(clip_ref_switched)) and block_index == 0:
                            decode_latents = motion_latents[:,:,:7]
                            _remote_edge_send_latents(
                                decode_latents,
                                reset_vae=True,
                                prime_only=True,
                            )
                            remote_edge_only = _remote_edge_latent_decode_delegated()
                            if (not bool(remote_edge_only)) or bool(stream_update_motion_from_decoded):
                                self.vae.stream_decode(decode_latents)
                        decode_latents = block_latents.unsqueeze(0)
                        block_output_frames = max(1, int(clip_infer_frames) // max(1, int(num_blocks)))
                        block_start_frame = int(block_index) * int(block_output_frames)
                        block_visible_start_frame = 0
                        block_audio_start_frame = int(block_start_frame)
                        block_visible_frames = int(block_output_frames)
                        if bool(stream_audio_mode):
                            visible_global_start = int(stream_clip_visible_start_frames)
                            visible_global_end = int(visible_global_start) + int(stream_clip_visible_frames)
                            block_global_start = int(block_start_frame)
                            block_global_end = int(block_global_start) + int(block_output_frames)
                            block_visible_global_start = int(max(int(block_global_start), int(visible_global_start)))
                            block_visible_global_end = int(min(int(block_global_end), int(visible_global_end)))
                            block_visible_frames = int(
                                max(0, int(block_visible_global_end) - int(block_visible_global_start))
                            )
                            block_visible_start_frame = int(
                                max(0, int(block_visible_global_start) - int(block_global_start))
                            )
                            block_audio_start_frame = int(
                                max(0, int(block_visible_global_start) - int(visible_global_start))
                            )
                        block_segment_id = None
                        block_segment_pcm = None
                        block_segment_audible_samples = None
                        block_subtitle_text = None
                        block_subtitle_start_samples = None
                        block_subtitle_end_samples = None
                        block_subtitle_total_samples = None
                        block_subtitle_alignment = None
                        block_subtitle_normalized_alignment = None
                        block_subtitle_alignment_base_samples = None
                        block_segment_turn_done = False
                        if bool(stream_audio_mode):
                            block_segment_id = f"{str(stream_clip_segment_id)}:b{int(block_index)}"
                            block_segment_pcm = _slice_pcm16le_for_frames(
                                stream_clip_pcm16le,
                                sample_rate=int(stream_clip_sample_rate),
                                start_frame=int(block_audio_start_frame),
                                frame_count=int(block_visible_frames),
                            )
                            block_start_samples, block_end_samples = _stream_sample_range_for_frames(
                                int(block_audio_start_frame),
                                int(block_visible_frames),
                                int(stream_clip_sample_rate),
                            )
                            block_segment_audible_samples = int(
                                max(
                                    0,
                                    min(int(block_end_samples), int(stream_clip_audible_samples))
                                    - int(block_start_samples),
                                )
                            )
                            subtitle_meta = {}
                            if int(stream_clip_source_chunk_idx) > 0:
                                try:
                                    subtitle_meta = _stream_chunk_meta(int(stream_clip_source_chunk_idx))
                                except Exception:
                                    subtitle_meta = {}
                                try:
                                    block_segment_turn_done = bool((subtitle_meta or {}).get("turn_done", False)) and (
                                        int(block_visible_frames) > 0
                                        and int(block_audio_start_frame) + int(block_visible_frames) >= int(stream_clip_visible_frames)
                                    )
                                except Exception:
                                    block_segment_turn_done = False
                            if int(stream_clip_source_chunk_idx) > 0 and int(block_segment_audible_samples or 0) > 0:
                                subtitle_text = str((subtitle_meta or {}).get("subtitle_text") or "").strip()
                                if subtitle_text:
                                    try:
                                        subtitle_base = int((subtitle_meta or {}).get("subtitle_start_samples") or 0)
                                    except Exception:
                                        subtitle_base = 0
                                    try:
                                        subtitle_total = int(
                                            (subtitle_meta or {}).get("subtitle_total_samples")
                                            or (subtitle_meta or {}).get("subtitle_end_samples")
                                            or 0
                                        )
                                    except Exception:
                                        subtitle_total = 0
                                    block_subtitle_text = str(subtitle_text)
                                    block_subtitle_start_samples = int(max(0, int(subtitle_base) + int(block_start_samples)))
                                    block_subtitle_end_samples = int(
                                        max(
                                            int(block_subtitle_start_samples),
                                            int(block_subtitle_start_samples) + int(block_segment_audible_samples or 0),
                                        )
                                    )
                                    block_subtitle_total_samples = int(
                                        max(int(subtitle_total), int(block_subtitle_end_samples))
                                    )
                                    block_subtitle_alignment = (
                                        dict((subtitle_meta or {}).get("subtitle_alignment"))
                                        if isinstance((subtitle_meta or {}).get("subtitle_alignment"), dict)
                                        else None
                                    )
                                    block_subtitle_normalized_alignment = (
                                        dict((subtitle_meta or {}).get("subtitle_normalized_alignment"))
                                        if isinstance((subtitle_meta or {}).get("subtitle_normalized_alignment"), dict)
                                        else None
                                    )
                                    try:
                                        block_subtitle_alignment_base_samples = int(
                                            (subtitle_meta or {}).get("subtitle_alignment_base_samples")
                                            if (subtitle_meta or {}).get("subtitle_alignment_base_samples") is not None
                                            else subtitle_base
                                        )
                                    except Exception:
                                        block_subtitle_alignment_base_samples = int(subtitle_base)
                        if (not bool(stream_audio_mode)) or int(block_visible_frames) > 0:
                            _remote_edge_send_latents(
                                block_latents,
                                keep_last_frames=int(block_output_frames),
                                segment_id=block_segment_id,
                                segment_kind=str(stream_clip_kind),
                                segment_audio_pcm16le=block_segment_pcm,
                                segment_sample_rate=int(stream_clip_sample_rate),
                                segment_start_frame=int(block_visible_start_frame),
                                segment_frames=int(block_visible_frames if bool(stream_audio_mode) else block_output_frames),
                                segment_audible_samples=block_segment_audible_samples,
                                segment_subtitle_text=block_subtitle_text,
                                segment_subtitle_start_samples=block_subtitle_start_samples,
                                segment_subtitle_end_samples=block_subtitle_end_samples,
                                segment_subtitle_total_samples=block_subtitle_total_samples,
                                segment_subtitle_alignment=block_subtitle_alignment,
                                segment_subtitle_normalized_alignment=block_subtitle_normalized_alignment,
                                segment_subtitle_alignment_base_samples=block_subtitle_alignment_base_samples,
                                segment_turn_done=bool(block_segment_turn_done),
                                avatar_ref_path=str(stream_clip_avatar_ref_path or "") or None,
                            )
                        remote_edge_only = _remote_edge_latent_decode_delegated()

                        rgb_pack_t0 = time.perf_counter()
                        rgb_bytes = None
                        raw_chunk_entry = None
                        image_cpu = None
                        if bool(remote_edge_only):
                            # In split latent mode, the edge owns VAE/RGB/PostVAE/LiveKit.
                            # Avoid doing a second local decode/pack path that competes with DiT.
                            if bool(stream_update_motion_from_decoded):
                                image_for_motion = torch.stack(self.vae.stream_decode(decode_latents))
                                motion_decoded_block = image_for_motion[:, :, -int(block_output_frames):].contiguous()
                            decode_dt = float(time.perf_counter() - decode_t0)
                        else:
                            image = torch.stack(self.vae.stream_decode(decode_latents))
                            image = image[:, :, -(clip_infer_frames)//num_blocks:] # 3
                            motion_decoded_block = image
                            decode_dt = float(time.perf_counter() - decode_t0)
                            has_live_sink = bool(
                                (hls_proc is not None and hls_proc.stdin is not None) or (raw_pipe_path is not None)
                            )
                            block_has_visible_output = bool((not stream_audio_mode) or int(block_visible_frames) > 0)
                            need_rgb_bytes = bool(
                                block_has_visible_output
                                and (not (stream_audio_mode and stream_clip_is_silence))
                                and has_live_sink
                            )
                            need_stream_file_frames = bool(
                                block_has_visible_output
                                and stream_file_enabled
                                and not (stream_audio_mode and stream_clip_is_silence)
                            )
                            # In realtime live_raw/hls mode we don't need to keep full CPU frame history.
                            # That extra image.cpu() copy is costly per block and causes end-of-reply slow drift.
                            need_cpu_out = bool(
                                (not bool(stream_file_enabled))
                                and ((not stream_audio_mode) or (stream_audio_mode and (not has_live_sink)))
                            )

                            if need_rgb_bytes or need_stream_file_frames:
                                # Convert on GPU first. Stream-file output keeps the restored
                                # T,C,H,W tensor on GPU and only crosses to host at the ffmpeg
                                # encoder boundary.
                                post_vae_output_h = int(HEIGHT)
                                post_vae_output_w = int(WIDTH)
                                if need_stream_file_frames and int(stream_file_output_h or 0) > 0 and int(stream_file_output_w or 0) > 0:
                                    post_vae_output_h = int(stream_file_output_h)
                                    post_vae_output_w = int(stream_file_output_w)
                                enhanced_tchw = self._enhance_live_raw_frames(
                                    image[0].permute(1, 0, 2, 3).contiguous(),
                                    clip_kind=str(stream_clip_kind),
                                    output_height=int(post_vae_output_h),
                                    output_width=int(post_vae_output_w),
                                )
                                if enhanced_tchw is not None:
                                    frames_01_tchw = enhanced_tchw.contiguous()
                                else:
                                    frames_01_tchw = ((image[0].permute(1, 0, 2, 3).contiguous() + 1.0) / 2.0).clamp(0.0, 1.0)
                                if bool(stream_audio_mode) and int(block_visible_frames) < int(frames_01_tchw.shape[0]):
                                    frames_01_tchw = frames_01_tchw[: int(max(0, int(block_visible_frames)))].contiguous()
                                if need_stream_file_frames:
                                    _stream_file_enqueue_frames(frames_01_tchw)
                                if need_rgb_bytes:
                                    rgb_gpu = (frames_01_tchw * 255.0).clamp(0, 255).to(torch.uint8)
                                    rgb_gpu = rgb_gpu.permute(0, 2, 3, 1).contiguous()  # T, H, W, 3
                                    if (raw_pipe_path is not None or raw_shm is not None) and hls_proc is None:
                                        try:
                                            raw_chunk_entry = _raw_make_async_entry(rgb_gpu)
                                        except Exception:
                                            raw_chunk_entry = None
                                    if raw_chunk_entry is None:
                                        rgb_bytes = rgb_gpu.cpu().numpy().tobytes()

                            if need_cpu_out:
                                cpu_pack_t0 = time.perf_counter()
                                image_cpu = image.cpu()
                                cpu_pack_dt = float(time.perf_counter() - cpu_pack_t0)
                        rgb_pack_dt = float(time.perf_counter() - rgb_pack_t0)

                        if hls_proc is not None and hls_proc.stdin is not None and rgb_bytes is not None:
                            try:
                                hls_proc.stdin.write(rgb_bytes)
                            except BrokenPipeError:
                                print(
                                    f"Rank {rank}: live HLS ffmpeg pipe closed (broken pipe).",
                                    flush=True,
                                )
                                try:
                                    hls_proc.stdin.close()
                                except Exception:
                                    pass
                                hls_proc = None
                            except Exception as e:
                                print(
                                    f"Rank {rank}: live HLS write failed: {e}",
                                    flush=True,
                                )

                        if (raw_pipe_path is not None or raw_shm is not None) and ((rgb_bytes is not None) or (raw_chunk_entry is not None)):
                            raw_enqueue_t0 = time.perf_counter()
                            raw_enqueued_before = int(raw_frames_enqueued)
                            raw_written_before = int(raw_frames_streamed)
                            first_raw_enqueue = bool(int(raw_frames_enqueued) <= 0)
                            raw_entry = raw_chunk_entry if raw_chunk_entry is not None else _raw_make_bytes_entry(rgb_bytes)
                            with raw_backlog_cv:
                                next_prompt_mode = str(clip_prompt_mode or "speech")
                                if next_prompt_mode not in {"speech", "idle"}:
                                    next_prompt_mode = "speech"
                                if str(next_prompt_mode) != str(raw_prompt_mode):
                                    raw_prompt_mode = str(next_prompt_mode)
                                    raw_prompt_mode_seq = int(raw_prompt_mode_seq) + 1
                                    raw_prompt_mode_start_frame = int(raw_enqueued_before)
                                next_source_chunk_idx = int(max(0, int(stream_clip_source_chunk_idx)))
                                if int(next_source_chunk_idx) != int(raw_source_chunk_idx):
                                    raw_source_chunk_idx = int(next_source_chunk_idx)
                                    raw_source_chunk_start_frame = int(raw_enqueued_before)
                                raw_frames_enqueued += int(raw_entry["frame_count"])
                                raw_frames_enq_delta = int(raw_entry["frame_count"])
                                raw_backlog_chunks.append(raw_entry)
                                raw_backlog_bytes += int(raw_entry["nbytes"])
                                raw_backlog_after = int(raw_backlog_bytes)
                                raw_backlog_cv.notify()
                            _write_raw_progress_marker(done=False)
                            if first_raw_enqueue and int(raw_frames_enq_delta) > 0:
                                stream_live_trace.note_first_raw_enqueue(
                                    frames_enqueued=int(raw_frames_enq_delta),
                                    backlog_before=int(raw_backlog_before),
                                    backlog_after=int(raw_backlog_after),
                                    q_depth=int(len(stream_audio_clips)),
                                    done=bool(stream_audio_done),
                                )
                            if (
                                raw_backlog_max_bytes > 0
                                and raw_backlog_after > int(raw_backlog_max_bytes)
                                and (not raw_backlog_warned)
                            ):
                                print(
                                    f"Rank {rank}: live RAW backlog high: {raw_backlog_after}B "
                                    f"(max_hint={raw_backlog_max_bytes}B).",
                                    flush=True,
                                )
                                raw_backlog_warned = True
                            raw_enqueue_dt = float(time.perf_counter() - raw_enqueue_t0)
                            raw_written_delta = max(0, int(raw_frames_streamed) - int(raw_written_before))
                            raw_write_dt = 0.0
                        if (
                            (not (stream_audio_mode and stream_clip_is_silence))
                            and (not stream_audio_mode or int(block_visible_frames) > 0)
                            and (image_cpu is not None)
                        ):
                            if bool(stream_audio_mode) and int(block_visible_frames) < int(image_cpu.shape[2]):
                                image_cpu = image_cpu[:, :, : int(max(0, int(block_visible_frames)))].contiguous()
                            out.append(image_cpu)
                        if (
                            bool(stateful_motion_latents)
                            and int(rank) == int(decode_rank)
                            and not bool(stream_clip_is_silence)
                            and int(block_visible_frames) > 0
                        ):
                            try:
                                if bool(stream_update_motion_from_decoded) and motion_decoded_block is not None:
                                    motion_block = motion_decoded_block
                                    try:
                                        if int(block_visible_start_frame) > 0 or int(block_visible_frames) < int(motion_block.shape[2]):
                                            motion_block = motion_block[
                                                :,
                                                :,
                                                int(block_visible_start_frame) : int(block_visible_start_frame)
                                                + int(block_visible_frames),
                                            ].contiguous()
                                    except Exception:
                                        pass
                                    overlap_frames_num = int(min(int(self.motion_frames), int(motion_block.shape[2])))
                                    if int(overlap_frames_num) > 0:
                                        videos_last_frames = torch.cat(
                                            [
                                                videos_last_frames[:, :, int(overlap_frames_num) :],
                                                motion_block[:, :, -int(overlap_frames_num) :],
                                            ],
                                            dim=2,
                                        ).detach()
                                    clip_visible_done = (
                                        int(block_audio_start_frame) + int(block_visible_frames) >= int(stream_clip_visible_frames)
                                    )
                                    if bool(clip_visible_done):
                                        videos_last_frames = videos_last_frames.to(
                                            dtype=self.vae.dtype,
                                            device=self.vae.device,
                                        ).contiguous()
                                        motion_latents = torch.stack(
                                            self.vae.encode(videos_last_frames)
                                        ).type_as(block_latents)
                                        if stream_timing_log:
                                            print(
                                                f"TPP motion_latents decoded-update clip={int(r)+1}/{int(active_nr)} "
                                                f"block={int(block_index)+1}/{int(num_blocks)} "
                                                f"visible={int(block_visible_frames)} "
                                                f"pixel_t={int(videos_last_frames.shape[2])} "
                                                f"latent_t={int(motion_latents.shape[2])}",
                                                flush=True,
                                            )
                                else:
                                    block_motion = block_latents.detach()
                                    if block_motion.ndim == 4:
                                        block_motion = block_motion.unsqueeze(0)
                                    target_motion_t = int(motion_latents.shape[2])
                                    if int(target_motion_t) > 0 and block_motion.ndim == 5:
                                        motion_base = motion_latents.to(
                                            device=block_motion.device,
                                            dtype=block_motion.dtype,
                                        )
                                        motion_latents = torch.cat(
                                            [motion_base, block_motion],
                                            dim=2,
                                        )[:, :, -int(target_motion_t) :].detach().contiguous()
                                        if stream_timing_log:
                                            print(
                                                f"TPP motion_latents latent-update clip={int(r)+1}/{int(active_nr)} "
                                                f"block={int(block_index)+1}/{int(num_blocks)} "
                                                f"visible={int(block_visible_frames)} "
                                                f"latent_t={int(motion_latents.shape[2])}",
                                                flush=True,
                                            )
                            except Exception as e:
                                if stream_timing_log:
                                    print(
                                        f"TPP motion_latents update skipped clip={int(r)+1}/{int(active_nr)} "
                                        f"block={int(block_index)+1}/{int(num_blocks)} err={e}",
                                        flush=True,
                                    )
                        block_total_dt = float(time.perf_counter() - block_t0)
                        profile_vae_blocks += 1
                        profile_vae_recv_s += float(vae_recv_dt)
                        profile_vae_decode_s += float(decode_dt)
                        profile_rgb_pack_s += float(rgb_pack_dt)
                        profile_cpu_pack_s += float(cpu_pack_dt)
                        profile_raw_enqueue_s += float(raw_enqueue_dt)
                        profile_raw_write_s += float(raw_write_dt)
                        if stream_audio_mode and int(rank) == int(decode_rank) and str(remote_edge_mode) == "latents":
                            remote_edge_last_block_dt = float(block_total_dt)
                            remote_edge_last_denoise_dt = float(denoise_total_dt)
                            remote_edge_last_recv_dt = float(recv_wait_dt)
                            remote_edge_last_send_wait_dt = float(send_wait_dt)
                            remote_edge_last_clip_frames = int(clip_infer_frames)
                            remote_edge_last_num_blocks = int(num_blocks)
                            remote_edge_last_kv_cache_size = int(kv_cache_size)
                            remote_edge_last_max_seq_len = int(max_seq_len)
                            remote_edge_last_kv_cap_frames = int(kv_cap_frames)
                            _remote_edge_log_producer_stats()
                        is_first_reply_block = bool(int(r) == 0 and int(block_index) == 0)
                        if stream_timing_log and stream_audio_mode and is_first_reply_block:
                            first_block_now_dt = float(time.perf_counter() - stream_trace_t0)
                            first_pop_gap = (
                                float(first_block_now_dt - float(stream_first_clip_pop_dt))
                                if isinstance(stream_first_clip_pop_dt, float)
                                else None
                            )
                            print(
                                f"TPP first-block rank={rank} clip={r+1}/{active_nr} "
                                f"block={block_index+1}/{num_blocks} total={block_total_dt:.3f}s "
                                f"recv={float(recv_wait_dt):.3f}s denoise={float(denoise_total_dt):.3f}s "
                                f"steps={int(steps_executed)} send={float(send_wait_dt):.3f}s "
                                f"decode={float(decode_dt):.3f}s rgb={float(rgb_pack_dt):.3f}s "
                                f"raw_enq={float(raw_enqueue_dt):.3f}s raw_wr={float(raw_write_dt):.3f}s "
                                f"raw_enq_frames={int(raw_frames_enq_delta)} raw_wr_frames={int(raw_written_delta)} "
                                f"raw_backlog={int(raw_backlog_before)}->{int(raw_backlog_after)} "
                                f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0} "
                                f"trace_dt={first_block_now_dt:.3f}s "
                                f"from_first_clip={f'{float(first_pop_gap):.3f}s' if first_pop_gap is not None else '-'}",
                                flush=True,
                            )
                        if stream_timing_log or (
                            stream_audio_mode and block_total_dt >= float(stream_block_log_slow_sec)
                        ):
                            print(
                                f"TPP block timing rank={rank} clip={r+1}/{active_nr} "
                                f"block={block_index+1}/{num_blocks} total={block_total_dt:.3f}s "
                                f"recv={float(recv_wait_dt):.3f}s denoise={float(denoise_total_dt):.3f}s "
                                f"steps={int(steps_executed)} send={float(send_wait_dt):.3f}s "
                                f"decode={float(decode_dt):.3f}s rgb={float(rgb_pack_dt):.3f}s "
                                f"raw_enq={float(raw_enqueue_dt):.3f}s raw_wr={float(raw_write_dt):.3f}s "
                                f"raw_enq_frames={int(raw_frames_enq_delta)} raw_wr_frames={int(raw_written_delta)} "
                                f"raw_backlog={int(raw_backlog_before)}->{int(raw_backlog_after)} "
                                f"q={int(len(stream_audio_clips))} done={1 if stream_audio_done else 0}",
                                flush=True,
                            )

                if bool(stream_audio_mode):
                    stream_global_block_offset = int(clip_global_block_offset) + int(num_blocks)

        profile_loop_s = float(time.perf_counter() - profile_loop_t0) if "profile_loop_t0" in locals() else 0.0

        def _remote_edge_file_output_expected() -> bool:
            if not live_raw_dir:
                return False
            try:
                manifest = _remote_edge_load_manifest()
                output = str(
                    manifest.get("output")
                    or ("rtmp" if str(manifest.get("rtmp_url") or "").strip() else "livekit")
                ).strip().lower()
                return bool(_remote_edge_env_enabled() and output == "file")
            except Exception:
                return False

        def _broadcast_remote_edge_file_result(value: str) -> str:
            if not dist.is_initialized():
                return str(value or "")
            payload = [str(value or "") if int(rank) == int(decode_rank) else ""]
            dist.broadcast_object_list(payload, src=int(decode_rank))
            return str(payload[0] or "")

        def _print_file_profile_summary(kind: str) -> None:
            if bool(stream_audio_mode) and not bool(stream_timing_log):
                return
            try:
                role = "decode" if int(rank) == int(decode_rank) else "dit"
                wall_s = float(time.perf_counter() - profile_total_t0)
                avg_denoise = profile_denoise_s / max(1, int(profile_dit_blocks))
                avg_vae = profile_vae_decode_s / max(1, int(profile_vae_blocks))
                print(
                    f"TPP file profile job={str(job_id or '-')} rank={int(rank)} role={role} kind={kind} "
                    f"clips={int(profile_active_clips)} blocks_per_clip={int(profile_last_num_blocks)} "
                    f"dit_blocks={int(profile_dit_blocks)} vae_blocks={int(profile_vae_blocks)} "
                    f"steps={int(profile_steps)} wall={wall_s:.3f}s "
                    f"audio={float(profile_audio_s):.3f}s static_cond={float(profile_static_cond_s):.3f}s "
                    f"prompt={float(profile_prompt_s):.3f}s scheduler_comm={float(profile_scheduler_comm_s):.3f}s "
                    f"loop={float(profile_loop_s):.3f}s core={float(profile_core_s):.3f}s "
                    f"recv={float(profile_recv_s):.3f}s denoise={float(profile_denoise_s):.3f}s "
                    f"avg_denoise_block={float(avg_denoise):.3f}s send={float(profile_send_s):.3f}s "
                    f"vae_recv={float(profile_vae_recv_s):.3f}s vae_decode={float(profile_vae_decode_s):.3f}s "
                    f"avg_vae_block={float(avg_vae):.3f}s rgb_pack={float(profile_rgb_pack_s):.3f}s "
                    f"cpu_pack={float(profile_cpu_pack_s):.3f}s raw_enq={float(profile_raw_enqueue_s):.3f}s "
                    f"raw_wr={float(profile_raw_write_s):.3f}s post_barrier={float(profile_post_barrier_s):.3f}s "
                    f"concat={float(profile_concat_s):.3f}s "
                    f"stream_file={1 if bool(stream_file_enabled) else 0} "
                    f"sf_blocks={int(stream_file_blocks)} sf_frames={int(stream_file_frames_in)}->{int(stream_file_frames_out)} "
                    f"sf_enq={float(stream_file_enqueue_s):.3f}s sf_rife={float(stream_file_rife_s):.3f}s "
                    f"sf_resize={float(stream_file_resize_s):.3f}s sf_pack={float(stream_file_pack_s):.3f}s "
                    f"sf_write={float(stream_file_write_s):.3f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"TPP file profile summary failed rank={rank}: {e}", flush=True)

        #-------------------------------------- Step 3: full-video postprocess--------------------------------------
        remote_edge_file_expected = bool(_remote_edge_file_output_expected())
        post_barrier_t0 = time.perf_counter()
        if dist.is_initialized():
            self._safe_barrier()
        profile_post_barrier_s += float(time.perf_counter() - post_barrier_t0)
        if rank == decode_rank:
                videos = None
                if len(out) <= 0:
                    if not stream_audio_mode and not bool(remote_edge_file_expected) and not bool(stream_file_enabled):
                        print(f"Rank {rank}: no non-silent frames generated; returning empty output", flush=True)
                        _print_file_profile_summary("empty_return")
                        return None, dataset_info
                    # Realtime and remote edge file-output modes may intentionally skip
                    # CPU frame accumulation. Continue to finalize live RAW/HLS/edge
                    # outputs and return the streamed/uploaded result.
                else:
                    concat_t0 = time.perf_counter()
                    videos = torch.cat(out, dim=2)
                    profile_concat_s += float(time.perf_counter() - concat_t0)
                try:
                    del clip_noise
                except Exception:
                    pass
                try:
                    del clip_latents
                except Exception:
                    pass
                try:
                    del block_latents
                except Exception:
                    pass
                self._sampler_timesteps = None
                self._sampler_sigmas = None
                self._sampler_timestep_blocks = None
                self._sampler_timestep_blocks_key = None
                # del sample_scheduler
                if hls_proc is not None:
                    try:
                        if hls_proc.stdin is not None:
                            hls_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        hls_proc.wait()
                    except Exception:
                        try:
                            hls_proc.kill()
                        except Exception:
                            pass
                if hls_log_f is not None:
                    try:
                        hls_log_f.close()
                    except Exception:
                        pass
                if rank == decode_rank:
                    _raw_stop_writer(drain=True)
                if bool(stream_file_enabled):
                    _stream_file_stop_writer(drain=True)
                    if stream_file_error:
                        raise RuntimeError(f"stream-file output failed: {stream_file_error}")
                    try:
                        print(
                            f"Rank {rank}: stream-file done: output={stream_file_path} "
                            f"blocks={int(stream_file_blocks)} frames={int(stream_file_frames_in)}->{int(stream_file_frames_out)} "
                            f"enqueue={float(stream_file_enqueue_s):.3f}s rife={float(stream_file_rife_s):.3f}s "
                            f"resize={float(stream_file_resize_s):.3f}s pack={float(stream_file_pack_s):.3f}s "
                            f"write={float(stream_file_write_s):.3f}s "
                            f"wall={max(0.0, float(stream_file_finished_s - stream_file_started_s)):.3f}s",
                            flush=True,
                        )
                    except Exception:
                        pass
                if raw_writer_error:
                    print(f"Rank {rank}: live RAW writer error: {raw_writer_error}", flush=True)
                if live_raw_dir:
                    try:
                        _write_raw_progress_marker(done=True)
                    except Exception:
                        pass
                    if raw_shm is not None:
                        try:
                            raw_shm.close()
                        except Exception:
                            pass
                        try:
                            raw_shm.unlink()
                        except Exception:
                            pass
                        raw_shm = None
                        raw_shm_name = None
                    try:
                        print(
                            f"Rank {rank}: live RAW stats: enq_frames={int(raw_frames_enqueued)} "
                            f"written_frames={int(raw_frames_streamed)} backlog_bytes={int(raw_backlog_bytes)} "
                            f"backlog_chunks={int(len(raw_backlog_chunks))}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    try:
                        if raw_done_json_path:
                            payload = {
                                "written_frames": int(raw_frames_streamed),
                                "enqueued_frames": int(raw_frames_enqueued),
                                "backlog_bytes": int(raw_backlog_bytes),
                                "backlog_chunks": int(len(raw_backlog_chunks)),
                                "frame_width": int(WIDTH),
                                "frame_height": int(HEIGHT),
                                "fps": float(self.fps),
                                "done_at_ms": int(time.time() * 1000.0),
                            }
                            tmp_done_json = str(raw_done_json_path) + ".tmp"
                            with open(tmp_done_json, "w", encoding="utf-8") as f:
                                json.dump(payload, f, ensure_ascii=False, indent=2)
                                try:
                                    f.flush()
                                    os.fsync(f.fileno())
                                except Exception:
                                    pass
                            os.replace(tmp_done_json, raw_done_json_path)
                            print(
                                f"Rank {rank}: live RAW done.json written: {raw_done_json_path}",
                                flush=True,
                            )
                    except Exception:
                        print(
                            f"Rank {rank}: live RAW done.json write failed: {traceback.format_exc()}",
                            flush=True,
                        )
                    try:
                        if raw_done_path:
                            with open(raw_done_path, "w", encoding="utf-8") as f:
                                f.write(str(int(time.time() * 1000.0)))
                                try:
                                    f.flush()
                                    os.fsync(f.fileno())
                                except Exception:
                                    pass
                            print(
                                f"Rank {rank}: live RAW .done written: {raw_done_path}",
                                flush=True,
                            )
                    except Exception:
                        print(
                            f"Rank {rank}: live RAW .done write failed: {traceback.format_exc()}",
                            flush=True,
                        )
                if live_raw_dir and len(raw_backlog_chunks) > 0:
                    try:
                        print(
                            f"Rank {rank}: live RAW closed with unwritten backlog="
                            f"{int(raw_backlog_bytes)}B chunks={int(len(raw_backlog_chunks))} "
                            f"frames_streamed={int(raw_frames_streamed)}",
                            flush=True,
                        )
                    except Exception:
                        pass
                if len(raw_backlog_chunks) > 0:
                    try:
                        with raw_backlog_cv:
                            while len(raw_backlog_chunks) > 0:
                                leftover_entry = raw_backlog_chunks.popleft()
                                _raw_release_entry(leftover_entry)
                            raw_backlog_bytes = 0
                    except Exception:
                        pass
                if remote_edge_sender is not None:
                    try:
                        remote_edge_result = remote_edge_sender.close()
                        if isinstance(remote_edge_result, dict):
                            remote_edge_file_result = dict(remote_edge_result)
                        print(
                            f"Rank {rank}: remote edge closed: frames={int(remote_edge_video_frames_sent)} "
                            f"audio_chunks={int(len(remote_edge_audio_sent_chunks))}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"Rank {rank}: remote edge close failed: {e}", flush=True)
                        if bool(remote_edge_file_expected):
                            try:
                                _broadcast_remote_edge_file_result("")
                            except Exception as broadcast_e:
                                print(f"Rank {rank}: remote edge file result broadcast failed: {broadcast_e}", flush=True)
                        if str(remote_edge_output or "").strip().lower() == "file" and _remote_edge_fail_fatal():
                            raise InferenceCancelled(f"remote_edge_file_close_failed: {e}") from e
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()

                if stream_audio_mode:
                    _stream_stop_async_producer()
                remote_edge_uploaded_path = ""
                if (
                    isinstance(remote_edge_file_result, dict)
                    and bool(remote_edge_file_result.get("uploaded"))
                    and str(remote_edge_file_result.get("path") or "").strip()
                ):
                    remote_edge_uploaded_path = f"edge-uploaded://{str(remote_edge_file_result.get('path') or '').strip()}"
                if bool(remote_edge_file_expected):
                    _broadcast_remote_edge_file_result(str(remote_edge_uploaded_path or ""))
                if remote_edge_uploaded_path:
                    _print_file_profile_summary("edge_uploaded_return")
                    return str(remote_edge_uploaded_path), dataset_info
                if bool(stream_file_enabled):
                    _print_file_profile_summary("stream_file_return")
                    return str(stream_file_path), dataset_info
                if videos is None:
                    _print_file_profile_summary("none_return")
                    return None, dataset_info
                _print_file_profile_summary("video_return")
                return videos[0], dataset_info
        else:
            if bool(remote_edge_file_expected):
                remote_edge_uploaded_path = _broadcast_remote_edge_file_result("")
                if remote_edge_uploaded_path:
                    _print_file_profile_summary("edge_uploaded_return")
                    return str(remote_edge_uploaded_path), dataset_info
            if stream_audio_mode:
                _stream_stop_async_producer()
            _print_file_profile_summary("non_decode_return")
            return None,dataset_info
    

    def tts(self, tts_prompt_audio, tts_prompt_text, tts_text):
        if not hasattr(self, 'cosyvoice'):
            self.load_tts()
        speech_list = []
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        prompt_speech_16k = load_wav(tts_prompt_audio, 16000)
        if tts_prompt_text is not None:
            for i in self.cosyvoice.inference_zero_shot(tts_text, tts_prompt_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        else:
            for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        torchaudio.save('tts.wav', torch.concat(speech_list, dim=1), self.cosyvoice.sample_rate)
        return 'tts.wav'

    def load_tts(self):
        if not os.path.exists('CosyVoice'):
            from wan.utils.utils import download_cosyvoice_repo
            download_cosyvoice_repo('CosyVoice')
        if not os.path.exists('CosyVoice2-0.5B'):
            from wan.utils.utils import download_cosyvoice_model
            download_cosyvoice_model('CosyVoice2-0.5B', 'CosyVoice2-0.5B')
        sys.path.append('CosyVoice')
        sys.path.append('CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosyvoice = CosyVoice2('CosyVoice2-0.5B')
