import pytest
import torch
import numpy as np
import math
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F

try:
    import torch_npu

    BACKEND = "npu"
except ImportError:
    BACKEND = "cuda" if torch.cuda.is_available() else "cpu"

if BACKEND == "cuda":
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    from torch_npu import npu_fusion_attention

#copied from flash_attention.tests.test_flash_attn
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply
def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask
def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    dq_pad_fn = output_pad_fn
    if key_padding_mask is not None:
        dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
    else:
        dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
    return (
        q_unpad.detach().requires_grad_(),
        k_unpad.detach().requires_grad_(),
        v_unpad.detach().requires_grad_(),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q.detach().requires_grad_(),
        k.detach().requires_grad_(),
        v.detach().requires_grad_(),
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    )
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1), 
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=None,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og)

@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len_q", [4096])
@pytest.mark.parametrize("seq_len_k", [4096])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_dim", [4])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
def test_flash_self_attention(batch_size, seq_len_q, seq_len_k, num_heads, head_dim, causal, varlen):
    q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float16).to(BACKEND)
    k = torch.randn(batch_size, seq_len_k, num_heads, head_dim, dtype=torch.float16).to(BACKEND)
    v = torch.randn(batch_size, seq_len_k, num_heads, head_dim, dtype=torch.float16).to(BACKEND)
    if not varlen:
        q_ref = q.detach().clone().to("cpu")
        k_ref = k.detach().clone().to("cpu")
        v_ref = v.detach().clone().to("cpu")
        out_ref = attention_ref(q, k, v, causal=causal)
        if BACKEND == "npu":
            if causal:
                atten_mask_npu = (
                    torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
                    .bool()
                    .to(torch.device("npu"))
                )
                out = npu_fusion_attention(
                    q, 
                    k, 
                    v, 
                    num_heads, 
                    "BSND", 
                    scale = 1.0 / math.sqrt(q.shape[-1]),
                    atten_mask=atten_mask_npu, 
                    sparse_mode=3,
                )[0]
            else:
                out = npu_fusion_attention(
                    q, 
                    k, 
                    v, 
                    num_heads, 
                    "BSND", 
                    scale = 1.0 / math.sqrt(q.shape[-1]),
                    sparse_mode=0,
                )[0]
        elif BACKEND == "cuda":
            out = flash_attn_func(q, k, v, causal=causal)
        print(f"Output max diff: {(out - out_ref.to(BACKEND)).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref.to(BACKEND)).abs().mean().item()}")
        print(out - out_ref.to(BACKEND))
        assert torch.allclose(out, out_ref.to(BACKEND), atol=1e-2), "Outputs do not match!"
    else:
        query_padding_mask = generate_random_padding_mask(seq_len_q, batch_size, device=BACKEND, mode="random")
        key_padding_mask = generate_random_padding_mask(seq_len_k, batch_size, device=BACKEND, mode="random")
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v, output_pad_fn, _, _ = generate_qkv(
            q, k, v, key_padding_mask, key_padding_mask
        )
        if BACKEND == "npu":
            if causal:
                atten_mask_npu = (
                    torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
                    .bool()
                    .to(torch.device("npu"))
                )
                print(q_unpad, k_unpad, cu_seqlens_q, cu_seqlens_k)
                out_unpad = npu_fusion_attention(
                    q_unpad, 
                    k_unpad, 
                    v_unpad, 
                    num_heads, 
                    "TND", 
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                    scale = 1.0 / math.sqrt(q_unpad.shape[-1]),
                    atten_mask=atten_mask_npu, 
                    sparse_mode=3,
                )[0]
            else:
                out_unpad = npu_fusion_attention(
                    q_unpad, 
                    k_unpad, 
                    v_unpad, 
                    num_heads, 
                    "TND", 
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                    scale = 1.0 / math.sqrt(q_unpad.shape[-1]),
                    sparse_mode=0,
                )[0]
        else:
            out_unpad = flash_attn_varlen_func(
                q_unpad, 
                k_unpad, 
                v_unpad, 
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal
            )
        out = output_pad_fn(out_unpad)
        out_ref = attention_ref(q, k, v, key_padding_mask, key_padding_mask, causal=causal)
        print(f"Output max diff: {(out - out_ref.to(BACKEND)).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref.to(BACKEND)).abs().mean().item()}")
        assert torch.allclose(out, out_ref.to(BACKEND), atol=1e-2), "Outputs do not match!"

if __name__ == "__main__":
    pytest.main([__file__])