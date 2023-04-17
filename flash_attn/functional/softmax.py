import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from fused_softmax_lib import (scaled_masked_softmax_backward, scaled_masked_softmax_forward,
                               scaled_upper_triang_masked_softmax_backward, scaled_upper_triang_masked_softmax_forward)
from torch.autograd import Function


class ScaledUpperTriangMaskedSoftmax(Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None


def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, "causal mask is only for self attention"
    # Reshaping input to 3D tensor (attn_batches, sq, sk)
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.cuda.amp.autocast(enabled=False):
        probs = ScaledUpperTriangMaskedSoftmax.apply(*args)
    return probs.view(b, np, sq, sk)


# NOTE (mkozuki): `ScaledMaskedSoftmax` somehow doesn't work well with `torch.cuda.amp.custom_fwd`.
# Without `cast_inputs` kwarg, somehow inputs are not cast to dtype used in the autocast context.
# So I needed to manually write two `Function` inheritances.
# Fused operation which performs following three operations in sequence
# 1. Scale the tensor.
# 2. Apply the mask.
# 3. Perform softmax.
class ScaledMaskedSoftmax(Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return ScaledMaskedSoftmax.apply(*args)
