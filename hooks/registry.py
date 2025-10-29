from typing import Callable, Literal

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock


def locate_block(pipe, position: str) -> nn.Module:
    """
    Locate the block at the specified position in the pipeline.
    """
    block = pipe
    for step in position.split("."):
        if step.isdigit():
            block = block[int(step)]
        else:
            block = getattr(block, step)
    return block


def register_general_hook(pipe, position, hook, with_kwargs=False, is_pre_hook=False):
    """
    Register a forward hook in a module of the pipeline specified with 'position'.
    """
    block: nn.Module = locate_block(pipe, position)

    if is_pre_hook:
        return block.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
    return block.register_forward_hook(hook, with_kwargs=with_kwargs)


def _safe_clip(x: torch.Tensor):
    if x.dtype == torch.float16:
        x[torch.isposinf(x)] = 65504
        x[torch.isneginf(x)] = -65504
    return x


@torch.no_grad()
def fix_inf_values_hook(*args):
    if len(args) == 3:
        module, input, output = args
        kwinput = {}
    elif len(args) == 4:
        module, input, kwinput, output = args
    else:
        raise AssertionError(f"Unexpected args length: {len(args)}")

    if isinstance(module, FluxTransformerBlock):
        return _safe_clip(output[0]), _safe_clip(output[1])

    if isinstance(module, FluxSingleTransformerBlock):
        return _safe_clip(output)

    return output


@torch.no_grad()
def edit_streams_hook(*args,
                      recompute_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      stream: Literal["text", "image", "both"]):
    """
    Apply recompute_fn to the chosen stream of a Flux transformer block.
    """
    if len(args) == 3:
        module, input, output = args
        kwinput = {}
    elif len(args) == 4:
        module, input, kwinput, output = args
    else:
        raise AssertionError(f"Unexpected args length: {len(args)}")

    if isinstance(module, FluxTransformerBlock):
        if stream == "text":
            output_text = recompute_fn(kwinput["encoder_hidden_states"], output[0])
            output_image = output[1]
        elif stream == "image":
            output_image = recompute_fn(kwinput["hidden_states"], output[1])
            output_text = output[0]
        else:
            raise AssertionError("Stream 'both' not supported for FluxTransformerBlock.")

        return _safe_clip(output_text), _safe_clip(output_image)

    if isinstance(module, FluxSingleTransformerBlock):
        if stream == "text":
            output[:, :512] = recompute_fn(kwinput["hidden_states"][:, :512], output[:, :512])
        elif stream == "image":
            output[:, 512:] = recompute_fn(kwinput["hidden_states"][:, 512:], output[:, 512:])
        else:
            output = recompute_fn(kwinput["hidden_states"], output)
        return _safe_clip(output)

    return output
