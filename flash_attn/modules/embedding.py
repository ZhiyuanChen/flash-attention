# Copyright (c) 2022, Tri Dao.

import torch
from einops import rearrange
from torch import Tensor, _assert
from torch import distributed as dist
from torch import nn
from torch.nn.modules.utils import _pair

from flash_attn.utils.distributed import all_reduce, reduce_scatter

try:
    from flash_attn.modules import FusedDense
except ImportError:
    FusedDense = None


from typing import Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from flash_attn.functional import apply_rotary_emb_qkv_


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
            the project up to embed_dim
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class BertEmbeddings(nn.Module):
    def __init__(
        self, embed_dim, vocab_size, max_position_embeddings, type_vocab_size, padding_idx=None, device=None, dtype=None
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, **factory_kwargs)
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        token_type_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings


class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, *args, process_group=None, padding_idx=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
            if num_embeddings % world_size != 0:
                raise ValueError(
                    f"num_embeddings ({num_embeddings}) must be divisible by " f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(num_embeddings // world_size, *args, padding_idx=padding_idx, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:
            return super().forward(input)
        else:
            rank = dist.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = super().forward(input)
            embeddings[input_ids_mask] = 0.0
            return embeddings


class ColumnParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, *args, process_group=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
            if embedding_dim % world_size != 0:
                raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by " f"world_size ({world_size})")
        else:
            world_size = 1
        super().__init__(num_embeddings, embedding_dim // world_size, *args, **kwargs)


class ParallelGPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        process_group,
        padding_idx=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, embed_dim, padding_idx=padding_idx, process_group=process_group, **factory_kwargs
        )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = ColumnParallelEmbedding(
                max_position_embeddings, embed_dim, process_group=process_group, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None, combine_batch_seqlen_dim=False):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        world_size = dist.get_world_size(self.process_group)
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            if world_size <= 1:
                embeddings = embeddings + position_embeddings
            else:
                partition_dim = self.position_embeddings.embedding_dim
                rank = dist.get_rank(self.process_group)
                embeddings[..., rank * partition_dim : (rank + 1) * partition_dim] += position_embeddings
        if combine_batch_seqlen_dim:
            embeddings = rearrange(embeddings, "b s d -> (b s) d")
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return embeddings if world_size <= 1 else reduce_fn(embeddings, self.process_group)


class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        fused_bias_fc=False,
    ):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")

        linear_cls = nn.Linear if not fused_bias_fc or not bias else FusedDense
        self.proj = linear_cls(in_chans * patch_size[0] * patch_size[1], embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(
            rearrange(x, "b c (h p1) (w p2) -> b h w (c p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1])
        )
        if self.flatten:
            x = rearrange(x, "b h w c -> b (h w) c")
        x = self.norm(x)
        return x


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, interleaved=False, scale_base=None, device=None):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, seqlen_offset=0):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        seqlen = x.shape[1] + seqlen_offset
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset)
        if self.scale is None:
            return apply_rotary_emb_qkv_(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:], None, None, self.interleaved
            )
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
                self.interleaved,
            )
