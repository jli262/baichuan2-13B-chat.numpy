import math
import numpy as np

def _get_interleave(n):

    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(x):
    """FP16-compatible function that fills a numpy array with -inf."""
    x[:] = np.NINF
    return x


def _buffered_future_mask(x, maxpos, alibi, attn_heads):
    # Create an upper triangular matrix
    _future_mask = np.triu(_fill_with_neg_inf(np.zeros([maxpos, maxpos])), 1)

    # Adjusting the shape and adding alibi
    _future_mask = np.expand_dims(_future_mask, axis=0) + alibi

    # Reshape and slice the mask to match the input tensor's shape and attention heads
    return _future_mask[: x.shape[0] * attn_heads, :maxpos, :maxpos]


def _gen_alibi_mask(x, n_head, max_pos):
    print("alibi_mask.x", x.ndim)
    # Assume _get_interleave is a function that returns a list or numpy array
    slopes = np.array(_get_interleave(n_head))  # Replace torch.Tensor with np.array
    position_point = np.arange(max_pos) - max_pos + 1
    position_point = np.expand_dims(np.expand_dims(position_point, axis=0), axis=0)
    position_point = np.broadcast_to(position_point, (n_head, 1, max_pos))
    print("alibi_mask.position_point", position_point.ndim, position_point.shape)
    diag = np.diag(position_point[0])
    print("alibi_mask.diag", diag.ndim, diag.shape)
    diag = np.expand_dims(np.expand_dims(diag, axis=0), axis=0)
    print("alibi_mask.diag", diag.ndim, diag.shape)
    position_point = position_point - np.transpose(diag, axes=(-1, -2) + tuple(range(diag.ndim - 2)))
    print("alibi_mask.position_point", position_point.ndim, position_point.shape)
    print("alibi_mask.slopes", slopes.shape)
    alibi = np.expand_dims(np.expand_dims(slopes, axis=1), axis=1) * position_point
    print("alibi_mask.alibi", alibi.ndim, alibi.shape)
    alibi = np.reshape(alibi, (n_head, 1, max_pos))
    print("alibi_mask.alibi", alibi.ndim, alibi.shape)
    alibi_mask = np.triu(_fill_with_neg_inf(np.zeros([max_pos, max_pos])), 1)
    print("alibi_mask.alibi_mask", alibi_mask.ndim, alibi_mask.shape)
    alibi_mask = np.expand_dims(alibi_mask, axis=0) + alibi
    print("alibi_mask.alibi_mask", alibi_mask.ndim, alibi_mask.shape)
    return alibi_mask

