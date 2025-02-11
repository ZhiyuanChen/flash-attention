from typing import Optional

import torch
from torch import Tensor
from torch import distributed as dist
from torch import nn
from torch.autograd import Function

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(dist):
    dist.all_gather_into_tensor = dist._all_gather_base
if "reduce_scatter_tensor" not in dir(dist):
    dist.reduce_scatter_tensor = dist._reduce_scatter_base


# Raw operation, does not support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False):
    world_size = dist.get_world_size(process_group)
    output = torch.empty(world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device)
    handle = dist.all_gather_into_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    return output, handle


# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False):
    world_size = dist.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device)
    handle = dist.reduce_scatter_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    return output, handle


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = dist.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class AllGatherFunc(Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply


class ReduceScatterFunc(Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply


def sync_shared_params(model: nn.Module, process_group: dist.ProcessGroup):
    # We want to iterate over parameters with _shared_params=True in the same order,
    # as different ranks might have different number of parameters (e.g., only rank 0 has bias).
    pamams_shared = {name: p for name, p in model.named_parameters() if getattr(p, "_shared_params", False)}
    for _, p in sorted(pamams_shared.items()):
        with torch.no_grad():
            # Broadcast needs src to be global rank, not group rank
            dist.broadcast(p, src=dist.get_global_rank(process_group, 0), group=process_group)


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/optimizer/optimizer.py#L256
def allreduce_sequence_parallel_grad(model: nn.Module, process_group: dist.ProcessGroup):
    # We want to iterate over parameters with _sequence_parallel=True in the same order,
    # as different ranks might have different number of parameters (e.g., only rank 0 has bias).
    params_seqparallel = {name: p for name, p in model.named_parameters() if getattr(p, "_sequence_parallel", False)}
    grads = [p.grad for _, p in sorted(params_seqparallel.items())]
    if grads:
        with torch.no_grad():
            coalesced = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
