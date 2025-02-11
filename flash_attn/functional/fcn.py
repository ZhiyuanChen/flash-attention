# Copyright (c) 2023, Tri Dao.
# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# We make it work with pytorch amp and with bfloat16.
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py

from functools import partial
from typing import Optional

# import fused_dense_cuda  # from apex
import torch
from torch import Tensor
from torch import distributed as dist
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F

from flash_attn.utils.distributed import all_gather_raw, all_reduce_raw, reduce_scatter_raw

from .activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd

try:
    import fused_dense_lib as fused_dense_cuda
except ImportError:
    fused_dense_cuda = None


class FusedDenseFunc(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, return_residual=False, process_group=None, sequence_parallel=True):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_output, weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel:
                handle_x.wait()
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_dense(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(x, weight, bias, return_residual, process_group, sequence_parallel)
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)


class FusedMLPFunc(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight1,
        bias1,
        weight2,
        bias2,
        activation="gelu_approx",
        save_pre_act=True,
        return_residual=False,
        checkpoint_lvl=0,
        heuristic=0,
        process_group=None,
        sequence_parallel=True,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather of x before doing the matmul.
        If sequence_parallel=False, then the input is already gathered.

        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out / relu_out in the bwd
        2: recompute pre_act and gelu_out / relu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        if activation == "sqrelu":
            assert heuristic == -1
        if not save_pre_act:
            checkpoint_lvl = 2
        assert checkpoint_lvl in [0, 1, 2]
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.activation = activation
        ctx.heuristic = heuristic

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2 = [a.to(dtype=dtype) for a in [weight1, weight2]]
            bias1 = bias1.to(dtype=dtype) if bias1 is not None else None
            bias2 = bias2.to(dtype=dtype) if bias2 is not None else None
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous() if bias1 is not None else None
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous() if bias2 is not None else None
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight1.shape, *weight2.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        if heuristic == -1:
            pre_act = F.linear(total_x, weight1, bias1)
            activation_fn = (
                partial(F.gelu, approximate="tanh")
                if activation == "gelu_approx"
                else (sqrelu_fwd if activation == "sqrelu" else F.relu)
            )
            with torch.jit.fuser("fuser2"):
                output1 = activation_fn(pre_act)
            # This is before adding bias1
            # pre_act = F.linear(total_x.reshape(batch_dim, n), weight1)
            # with torch.jit.fuser('fuser2'):
            #     output1 = bias_gelu(pre_act, bias1)
        else:
            is_gelu = activation == "gelu_approx"
            output1, *rest = fused_dense_cuda.linear_act_forward(
                total_x.reshape(batch_dim, n), weight1, bias1, is_gelu, save_pre_act, heuristic
            )
            if save_pre_act:
                pre_act = rest[0]
        output2 = F.linear(output1, weight2, bias2)
        if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == "relu"):
            # For RELU the pre_act is very small (just a bit-mask) so we just save it
            ctx.save_for_backward(x, weight1, weight2, pre_act, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, weight2, pre_act)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        output2 = output2.reshape(*batch_shape, output2.shape[-1])
        return output2 if not return_residual else (output2, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        activation = ctx.activation
        activation_fn = (
            partial(F.gelu, approximate="tanh")
            if activation == "gelu_approx"
            else (sqrelu_fwd if activation == "sqrelu" else F.relu)
        )
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        x, weight1, weight2, *rest = ctx.saved_tensors
        if process_group is None or not sequence_parallel:
            total_x = x
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl in [0, 1]:
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            if checkpoint_lvl == 0 or (checkpoint_lvl == 1 and activation == "relu"):
                pre_act, output1 = rest
            elif checkpoint_lvl == 1:
                (pre_act,) = rest
                with torch.jit.fuser("fuser2"):
                    output1 = activation_fn(pre_act)
        elif checkpoint_lvl == 2:
            (bias1,) = rest
            if process_group is not None and sequence_parallel:
                total_x, _ = all_gather_raw(x, process_group)
            if ctx.heuristic == -1:
                pre_act = F.linear(total_x, weight1, bias1)
                with torch.jit.fuser("fuser2"):
                    output1 = activation_fn(pre_act)
            else:
                output1, pre_act = fused_dense_cuda.linear_act_forward(
                    total_x.reshape(batch_dim, total_x.shape[-1]),
                    weight1,
                    bias1,
                    activation == "gelu_approx",
                    True,
                    ctx.heuristic,
                )

        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        output1 = output1.reshape(batch_dim, output1.shape[-1])
        pre_act = pre_act.reshape(batch_dim, pre_act.shape[-1])
        if ctx.needs_input_grad[3]:
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output, ctx.needs_input_grad[4])
        else:
            grad_weight2 = None
            grad_bias2 = grad_output if ctx.needs_input_grad[4] else None
        if ctx.heuristic == -1:
            # grad_pre_act = matmul_dgelu(grad_output, weight2, pre_act)
            grad_output1 = F.linear(grad_output, weight2.t())
            activation_grad_fn = (
                gelu_bwd if activation == "gelu_approx" else (sqrelu_bwd if activation == "sqrelu" else relu_bwd)
            )
            with torch.jit.fuser("fuser2"):
                grad_pre_act = activation_grad_fn(grad_output1, pre_act)
        else:
            # The cublasLt epilogue has to compute both gelu/relu grad and bias grad, we can't
            # just compute gelu/relu grad
            grad_pre_act, grad_bias1 = fused_dense_cuda.bias_act_linear_dgrad_bgrad(
                weight2, grad_output, pre_act, activation == "gelu_approx", ctx.heuristic
            )
            if not ctx.needs_input_grad[2]:
                grad_bias1 = None
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_pre_act, weight1.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]), grad_pre_act, weight1)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.heuristic == -1:
            if ctx.needs_input_grad[1]:
                if process_group is not None and sequence_parallel:
                    handle_x.wait()
                grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_wgrad(
                    total_x.reshape(batch_dim, total_x.shape[-1]), grad_pre_act, ctx.needs_input_grad[2]
                )
            else:
                grad_weight1 = None
                grad_bias1 = grad_pre_act if ctx.needs_input_grad[2] else None
        else:
            if ctx.needs_input_grad[1]:
                if process_group is not None and sequence_parallel:
                    handle_x.wait()
                grad_weight1 = F.linear(grad_pre_act.t(), total_x.reshape(batch_dim, total_x.shape[-1]).t())
            else:
                grad_weight1 = None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return (
            grad_input,
            grad_weight1,
            grad_bias1,
            grad_weight2,
            grad_bias2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_mlp(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    activation: str = "gelu_approx",
    save_pre_act: bool = True,
    return_residual: bool = False,
    checkpoint_lvl: int = 0,
    heuristic: int = 0,
    process_group: Optional[dist.ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    assert activation in ["gelu_approx", "relu", "sqrelu"]
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    # If we save pre-activation, dimension must be divisible by 128 (relu) or 8 (gelu)
    dim_eligible = not save_pre_act or (x.shape[-1] % (128 if activation == "relu" else 8) == 0)
    if (
        x.is_cuda
        and weight1.is_cuda
        and weight2.is_cuda
        and (bias1 is None or bias1.is_cuda)
        and (bias2 is None or bias2.is_cuda)
        and dtype_eligible
        and dim_eligible
    ):
        return FusedMLPFunc.apply(
            x,
            weight1,
            bias1,
            weight2,
            bias2,
            activation,
            save_pre_act,
            return_residual,
            checkpoint_lvl,
            heuristic,
            process_group,
            sequence_parallel,
        )
    else:
        assert process_group is None
        pre_act = F.linear(x, weight1, bias1)
        activation_fn = (
            partial(F.gelu, approximate="tanh") if activation == "gelu_approx" else partial(F.relu, inplace=True)
        )
        output1 = activation_fn(pre_act)
        output2 = F.linear(output1, weight2, bias2)
        return output2 if not return_residual else (output2, x)
