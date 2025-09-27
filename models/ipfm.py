from collections import OrderedDict
from typing import Optional
import os
import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers import T5EncoderModel, T5Tokenizer
import json
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, time_embedding_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor):
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, time_embedding_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(embedding_dim, embedding_dim * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(embedding_dim * 4, embedding_dim)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(embedding_dim, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(embedding_dim, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(embedding_dim, time_embedding_dim)

    def attention(self, q, kv):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(self, x, latents, timestep_embedding):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(     # attn: batch*64*768   lantents: batch*64*768
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))   # batch*64*768
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, embedding_dim, layers, heads, num_latents, input_dim, time_embedding_dim):
        super().__init__()
        self.latents = nn.Parameter(embedding_dim**-0.5 * torch.randn(num_latents, embedding_dim))
        self.time_aware_linear = nn.Linear(time_embedding_dim, embedding_dim, bias=True)
        self.proj_in = nn.Linear(input_dim, embedding_dim)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    embedding_dim, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, timestep_embedding):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)      # batch*64*768
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )       # batch*64*768

        x = self.proj_in(x)          # batch*128*768----batch*128*768
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)    # batch*64*768

        return latents


class ModelConfig:
    def __init__(self, time_channel, time_embed_dim, embedding_dim, layers, heads, num_latents, input_dim):
        self.time_channel = time_channel
        self.time_embed_dim = time_embed_dim
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.heads = heads
        self.num_latents = num_latents
        self.input_dim = input_dim
        # self.cross_attention_dim = cross_attention_dim


class IPFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.time_channel = config.time_channel
        self.time_embed_dim = config.time_embed_dim

        self.embedding_dim = config.embedding_dim
        self.layers = config.layers
        self.heads = config.heads
        self.num_latents = config.num_latents
        self.input_dim = config.input_dim
        # self.cross_attention_dim = config.cross_attention_dim

        self.position = Timesteps(
            self.time_channel, flip_sin_to_cos=True, downscale_freq_shift=0  # 输出 batch*320
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.time_channel,
            time_embed_dim=self.time_embed_dim,
            act_fn='silu',
            out_dim=None,
        )

        self.connector = PerceiverResampler(
            embedding_dim=self.embedding_dim,
            layers=self.layers,
            heads=self.heads,
            num_latents=self.num_latents,
            input_dim=self.input_dim,
            time_embedding_dim=self.time_embed_dim,
        )
        # self.proj_out = nn.Linear(self.embedding_dim, self.cross_attention_dim)

    @property
    def dtype(self):
        for param in self.parameters():
            return param.dtype
        return None

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)     # batch*320
        time_feature = (                              # batch*1*320
            time_feature.unsqueeze(dim=1)
            if time_feature.ndim == 2
            else time_feature
        )
        time_feature = time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(time_feature)     # batch*1*768

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )    # batch*64*768

        # encoder_hidden_states = self.proj_out(encoder_hidden_states)

        return encoder_hidden_states

    @classmethod
    def from_pretrained(cls, model_path):
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict['model'])

        # Instantiate model
        model = cls(model_config)

        # Load weights
        weights_path = os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save config
        config = {
            "model": {
                "time_channel": self.time_channel,
                "time_embed_dim": self.time_embed_dim,
                "embedding_dim": self.embedding_dim,
                "layers": self.layers,
                "heads": self.heads,
                "num_latents": self.num_latents,
                "input_dim": self.input_dim,
                # "cross_attention_dim": self.cross_attention_dim,
            }
        }
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save model weights
        weights_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), weights_path)
    # def save_pretrained(self, save_directory):
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #     config_path = os.path.join(save_directory, 'config.json')
    #     with open(config_path, 'w') as f:
    #         json.dump(self.config, f)
    #     weights_path = os.path.join(save_directory, 'pytorch_model.bin')
    #     torch.save(self.state_dict(), weights_path)


# class T5TextEmbedder(nn.Module):
#     def __init__(self, config, pretrained_path):
#         super().__init__()
#         self.config = config
#         self.device = config.device
#
#         self.model = T5EncoderModel.from_pretrained(pretrained_path)
#         self.model.to(self.device)
#         self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
#
#         self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
#         self.max_length = config.data.max_length
#
#     def forward(self, caption):
#         max_length = self.max_length
#         text_inputs = self.tokenizer(caption,
#                                      return_tensors="pt",
#                                      add_special_tokens=True,
#                                      max_length=max_length,
#                                      padding="max_length",
#                                      truncation=True,
#                                      )
#
#         text_input_ids = text_inputs.input_ids.to(self.device)
#         attention_mask = text_inputs.attention_mask.to(self.device)
#
#         outputs = self.model(text_input_ids, attention_mask=attention_mask)
#         embeddings = outputs.last_hidden_state
#
#         return embeddings


class T5TextEmbedder(nn.Module):
    def __init__(self, config, pretrained_path):
        super().__init__()
        self.config = config
        self.device = config.device

        self.model = T5EncoderModel.from_pretrained(pretrained_path).to(self.device)

        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = config.data.max_length

    def forward(self, caption):
        max_length = self.max_length
        text_inputs = self.tokenizer(caption,
                                     return_tensors="pt",
                                     add_special_tokens=True,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     )

        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        outputs = self.model(text_input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        return embeddings