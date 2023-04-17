# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
# But we make it much faster: we compute the local loss and the LSE, and by exchanging the LSE and
# the losses we can get the global loss. There's no need to do it step by step
# (compute local max, exchange, compute exp, compute local sum, exchange, etc.)
# The original xentropy interface is here: https://github.com/NVIDIA/apex/blob/master/apex/contrib/xentropy/softmax_xentropy.py
from torch import nn

from flash_attn.functional import softmax_cross_entropy_loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self, ignore_index=-100, reduction="mean", label_smoothing=0.0, inplace_backward=False, process_group=None
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = softmax_cross_entropy_loss(
            input, target, self.label_smoothing, self.ignore_index, self.inplace_backward, self.process_group
        )
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
