# Copyright (c) 2022, Tri Dao.

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from flash_attn.functional import fused_dense, fused_mlp
from flash_attn.functional.triton import fused_dense_sqrelu_dense_function
from flash_attn.utils.distributed import all_reduce, reduce_scatter


class FusedDense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_residual: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.
        """
        return fused_dense(x, self.weight, self.bias, return_residual=self.return_residual, process_group=process_group)


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: dist.ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        world_size = dist.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return fused_dense(
            x, self.weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel
        )


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: dist.ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if in_features % world_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by " f"world_size ({world_size})")
        # Only rank 0 will have bias
        super().__init__(in_features // world_size, out_features, bias=bias and rank == 0, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class FusedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        bias1=True,
        bias2=True,
        activation="gelu_approx",
        return_residual=False,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    ):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
                For H100, we set heuristic=-1 for both fp16 and bf16 as the fused cuBlasLt implementation
                is slower than the unfused version.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.activation = activation
        self.return_residual = return_residual
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic if activation != "sqrelu" else -1
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x, process_group=None):
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        if self.heuristic == "auto":
            if self.activation == "gelu_approx":
                if torch.cuda.get_device_capability("cuda") == (9, 0):
                    heuristic = -1
                else:
                    cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                    heuristic = 0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        out = fused_mlp(
            x,
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            activation=self.activation,
            save_pre_act=self.training,
            return_residual=self.return_residual,
            checkpoint_lvl=self.checkpoint_lvl,
            heuristic=heuristic,
            process_group=process_group,
        )
        if self.return_residual:
            out, x = out
        if process_group is not None:
            out = reduce_scatter(out, process_group)
        return out if not self.return_residual else (out, x)


class ParallelFusedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        activation="gelu_approx",
        process_group: dist.ProcessGroup = None,
        bias1=True,
        bias2=True,
        sequence_parallel=True,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    ):
        """
        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ["gelu_approx", "relu", "sqrelu"]
        assert process_group is not None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.activation = activation
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic if activation != "sqrelu" else -1
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, process_group, bias=bias1, **factory_kwargs)
        self.fc2 = RowParallelLinear(hidden_features, out_features, process_group, bias=bias2, **factory_kwargs)

    def forward(self, x):
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        if self.heuristic == "auto":
            if self.activation == "gelu_approx":
                cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                heuristic = 0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        out = fused_mlp(
            x,
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            activation=self.activation,
            save_pre_act=self.training,
            checkpoint_lvl=self.checkpoint_lvl,
            heuristic=heuristic,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
        )
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class FusedDenseSqreluDense(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, bias=True, checkpoint_lvl=0, device=None, dtype=None
    ):
        """
        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert bias == True, "DenseSqreluDense module without bias is currently not supported"
        self.checkpoint_lvl = checkpoint_lvl
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        assert x.is_cuda
        return fused_dense_sqrelu_dense_function(
            x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.checkpoint_lvl
        )
