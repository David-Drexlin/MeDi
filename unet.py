# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import math 

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


class SineCosinePositionEncoder(nn.Module):
    def __init__(self, embedding_dim=32, max_position=100.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_position = max_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size,) float values (e.g. age in [0, 100])
        Returns: (batch_size, embedding_dim) with sine/cosine encoding
        """
        # 1) Clamp to [0, max_position]
        x_clamped = x.clamp(0.0, self.max_position)

        # 2) Normalize to [0, 1]
        x_norm = x_clamped / self.max_position

        # 3) Standard sine/cosine positional encoding
        half_dim = self.embedding_dim // 2
        emb = torch.arange(half_dim, device=x.device, dtype=x.dtype)
        # frequencies ~ 1 / (10000^(2i/embedding_dim))
        div_term = torch.exp(-math.log(10000.0) * (2 * emb / self.embedding_dim))
        pos = x_norm.unsqueeze(1) * div_term.unsqueeze(0)  # (B, half_dim)
        sin_ = torch.sin(pos)
        cos_ = torch.cos(pos)
        out = torch.cat([sin_, cos_], dim=1)  # (B, embedding_dim)

        # If embedding_dim is odd, you can slice or zero-pad
        if self.embedding_dim % 2 == 1:
            out = out[:, : self.embedding_dim]

        return out


@dataclass
class UNet2DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    (Parameters omitted for brevity)
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        domain_embeds: Optional[dict] = None,
        positional_domains: Optional[list] = None,     
        pos_domain_ranges: Optional[dict] = None,      
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. "
                f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "linear": 
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "additive":  # <-- NEW: additive mode for elementwise summation
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        else:
            self.class_embedding = None

        ## Domain embeddings setup
        if domain_embeds is None:
            domain_embeds = {}
        if positional_domains is None:
            positional_domains = []

        self.positional_domains = positional_domains
        self.pos_domain_ranges = pos_domain_ranges if pos_domain_ranges is not None else {}

        # For concatenation, linear, or additive approaches, set up domain embeddings accordingly.
        if class_embed_type == "concat":  ## DD
            total_embeds = 1 + len(domain_embeds) + len(positional_domains)
            floor_dim, remainder = divmod(time_embed_dim, total_embeds)
            class_embedding_dim = floor_dim + remainder  # e.g. when dividing among domains
            domain_embedding_dim = floor_dim

            self.class_embedding = nn.Embedding(num_class_embeds, class_embedding_dim)
            
            self.domain_embeddings = nn.ModuleDict()
            self.positional_domain_embeddings = nn.ModuleDict()

            for domain_size in domain_embeds.keys():
                self.domain_embeddings[domain_size] = nn.Embedding(len(domain_embeds[domain_size]), domain_embedding_dim)

            for domain_name in positional_domains:
                self.positional_domain_embeddings[domain_name] = SineCosinePositionEncoder(
                    embedding_dim=domain_embedding_dim,  
                    max_position=100.0
                )
            
        elif class_embed_type == "linear":
            self.domain_embeddings = nn.ModuleDict()
            for domain_size in domain_embeds.keys():
                self.domain_embeddings[domain_size] = nn.Embedding(len(domain_embeds[domain_size]), time_embed_dim)
            total_embeds = 1 + len(domain_embeds)  # 1 for class_emb, rest for domain_emb
            self.linear_comb = nn.Linear(total_embeds * time_embed_dim, time_embed_dim)
            
        elif class_embed_type == "additive":  # <-- NEW: setup for additive mode
            self.domain_embeddings = nn.ModuleDict()
            for domain_name, domain_vals in domain_embeds.items():
                self.domain_embeddings[domain_name] = nn.Embedding(len(domain_vals), time_embed_dim)
            self.positional_domain_embeddings = nn.ModuleDict()
            for domain_name in positional_domains:
                self.positional_domain_embeddings[domain_name] = SineCosinePositionEncoder(
                    embedding_dim=time_embed_dim,  
                    max_position=100.0
                )

        self.cross_attention_dim = time_embed_dim

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
                cross_attention_dim=self.cross_attention_dim,  # DD 
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # Upsampling blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
                cross_attention_dim=self.cross_attention_dim,  # DD
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[dict] = None,
        return_dict: bool = True,
        return_embeddings: bool = False,  # Added parameter
    ) -> Union[UNet2DOutput, Tuple]:
        if return_embeddings:
            zs_tilde = []

        # 0. Center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. Time embeddings
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Broadcast to batch dimension
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Class and domain embeddings
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            # Compute the class embedding
            class_emb = self.class_embedding(class_labels)

            # NEW: Handle the "additive" case explicitly
            if self.config.class_embed_type == "additive":
                domain_emb_sum = torch.zeros_like(class_emb)
                if domain_labels is not None:
                    for domain_name, domain_val in domain_labels.items():
                        if hasattr(self, "domain_embeddings") and domain_name in self.domain_embeddings:
                            domain_emb_sum = domain_emb_sum + self.domain_embeddings[domain_name](domain_val)
                        elif hasattr(self, "positional_domain_embeddings") and domain_name in self.positional_domain_embeddings:
                            domain_emb_sum = domain_emb_sum + self.positional_domain_embeddings[domain_name](domain_val)
                emb = emb + class_emb + domain_emb_sum
            else:
                # Original behavior for non-additive modes
                domain_emb_list = []
                if domain_labels is not None:
                    for domain_name, domain_val in domain_labels.items():
                        if hasattr(self, "domain_embeddings") and domain_name in self.domain_embeddings:
                            domain_emb = self.domain_embeddings[domain_name](domain_val)
                            domain_emb_list.append(domain_emb)
                        elif hasattr(self, "positional_domain_embeddings") and domain_name in self.positional_domain_embeddings:
                            domain_emb = self.positional_domain_embeddings[domain_name](domain_val)
                            domain_emb_list.append(domain_emb)

                if len(domain_emb_list) == 0:
                    emb = emb + class_emb
                else:
                    domain_emb = torch.cat(domain_emb_list, dim=-1)
                    if self.config.class_embed_type == "concat":
                        comb = torch.cat((class_emb, domain_emb), dim=-1)
                        emb = emb + comb
                    elif self.config.class_embed_type == "linear":
                        combined_emb = torch.cat((class_emb, domain_emb), dim=-1)
                        comb = self.linear_comb(combined_emb)
                        emb = emb + comb
                    else:
                        emb = emb + class_emb + domain_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 3. Pre-process
        skip_sample = sample
        sample = self.conv_in(sample)
        if return_embeddings:
            zs_tilde.append(sample.clone())

        # 4. Downsampling
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
            if return_embeddings:
                zs_tilde.append(sample.clone())

        # 5. Middle block
        sample = self.mid_block(sample, emb)
        if return_embeddings:
            zs_tilde.append(sample.clone())

        # 6. Upsampling
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)
            if return_embeddings:
                zs_tilde.append(sample.clone())

        # 7. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if skip_sample is not None:
            sample += skip_sample
        if return_embeddings:
            zs_tilde.append(sample.clone())

        # 8. Output
        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if return_embeddings:
            if not return_dict:
                return (sample, zs_tilde)
            else:
                return UNet2DOutput(sample=sample), zs_tilde
        else:
            if not return_dict:
                return (sample,)
            else:
                return UNet2DOutput(sample=sample)

