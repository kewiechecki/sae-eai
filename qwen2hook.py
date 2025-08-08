import torch as t
from contextlib import contextmanager

def _coerce_hidden_like(new_hidden: t.Tensor, ref: t.Tensor) -> t.Tensor:
    """
    Make new_hidden match dtype/device/shape of ref (the original hidden state).
    Accepts (seq_len, hidden) and will expand to (batch, seq_len, hidden) if needed.
    """
    nh = new_hidden.to(dtype=ref.dtype, device=ref.device)
    if nh.dim() == 2 and ref.dim() == 3:
        # Assume single-batch injection
        nh = nh.unsqueeze(0).expand(ref.shape[0], -1, -1)
    if nh.shape != ref.shape:
        raise ValueError(f"new_hidden shape {nh.shape} != expected {ref.shape}")
    return nh

def replace_layer_hidden_state(model, layer_idx: int, new_hidden: t.Tensor, once: bool = True):
    """
    Replaces the OUTPUT hidden state of model.model.layers[layer_idx].
    Returns the hook handle (you can .remove() it). If once=True, auto-removes after first use.
    """
    layer = model.model.layers[layer_idx]
    handle = None  # will be closed over

    def hook(mod, inputs, output):
        nonlocal handle
        # Qwen2DecoderLayer returns either Tensor or tuple (hidden, *extras)
        if isinstance(output, (tuple, list)):
            ref = output[0]
            nh = _coerce_hidden_like(new_hidden, ref)
            out_list = list(output)
            out_list[0] = nh
            new_out = type(output)(out_list) if isinstance(output, tuple) else out_list
        else:
            ref = output
            nh = _coerce_hidden_like(new_hidden, ref)
            new_out = nh
        if once and handle is not None:
            handle.remove()
        return new_out

    # Some older PyTorch versions ignore the return value of forward hooks.
    # We try forward hook first; if the output isn’t replaced you can use the pre-hook fallback below.
    handle = layer.register_forward_hook(hook, with_kwargs=True)
    return handle

def replace_next_layer_input(model, layer_idx: int, new_hidden: t.Tensor, once: bool = True):
    """
    Fallback: replaces the *INPUT* to the NEXT layer (layer_idx+1) via a forward_pre_hook.
    Useful if your torch version doesn't support modifying outputs in forward hooks.
    """
    assert layer_idx + 1 < len(model.model.layers), "No next layer to patch"
    next_layer = model.model.layers[layer_idx + 1]
    handle = None

    def prehook(mod, args, kwargs):
        nonlocal handle
        # Qwen2DecoderLayer forward signature: (hidden_states, ..., position_ids=..., ...)
        # Inputs can be passed positionally or via kwargs; we replace the first positional tensor or kwargs['hidden_states'].
        if 'hidden_states' in kwargs:
            ref = kwargs['hidden_states']
            nh = _coerce_hidden_like(new_hidden, ref)
            kwargs = dict(kwargs)
            kwargs['hidden_states'] = nh
            if once and handle is not None:
                handle.remove()
            return args, kwargs

        # positional path
        if len(args) >= 1 and isinstance(args[0], t.Tensor):
            ref = args[0]
            nh = _coerce_hidden_like(new_hidden, ref)
            args = (nh,) + tuple(args[1:])
            if once and handle is not None:
                handle.remove()
            return args, kwargs

        raise RuntimeError("Could not locate hidden_states in next layer's inputs")

    handle = next_layer.register_forward_pre_hook(prehook, with_kwargs=True)
    return handle

@contextmanager
def patch_layer_hidden_state(model, layer_idx: int, new_hidden: t.Tensor, mode: str = "output"):
    """
    Context manager for one-call patching.
    mode: "output" replaces this layer's output; "input" replaces next layer's input.
    """
    if mode == "output":
        h = replace_layer_hidden_state(model, layer_idx, new_hidden, once=True)
    else:
        h = replace_next_layer_input(model, layer_idx, new_hidden, once=True)
    try:
        yield
    finally:
        # If torch removed it on first use, .remove() will be a no-op
        h.remove()

