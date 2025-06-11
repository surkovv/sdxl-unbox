from typing import Callable, Literal
import torch
import torch.nn as nn    
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock


def register_general_hook(pipe, position, hook, with_kwargs=False, is_pre_hook=False):
    """Registers a forward hook in a module of the pipeline specified with 'position'

    Args:
        pipe (_type_): _description_
        position (_type_): _description_
        hook (_type_): _description_
        with_kwargs (bool, optional): _description_. Defaults to False.
        is_pre_hook (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    block: nn.Module = locate_block(pipe, position)

    if is_pre_hook:
        return block.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
    else:
        return block.register_forward_hook(hook, with_kwargs=with_kwargs)


def locate_block(pipe, position: str) -> nn.Module:
    '''
    Locate the block at the specified position in the pipeline.
    '''
    block = pipe
    for step in position.split('.'):
        if step.isdigit():
            step = int(step)
            block = block[step]
        else:
            block = getattr(block, step)
    return block


def _safe_clip(x: torch.Tensor):
    if x.dtype == torch.float16:
        x[torch.isposinf(x)] = 65504
        x[torch.isneginf(x)] = -65504
    return x
    

@torch.no_grad()
def fix_inf_values_hook(*args):

    # Case 1: no kwards are passed to the module
    if len(args) == 3:
        module, input, output = args
    # Case 2: when kwargs are passed to the model as input
    elif len(args) == 4:
        module, input, kwinput, output = args

    if isinstance(module, FluxTransformerBlock):
        return _safe_clip(output[0]), _safe_clip(output[1])

    elif isinstance(module, FluxSingleTransformerBlock):
        return _safe_clip(output)
    

@torch.no_grad()
def edit_streams_hook(*args, 
                      recompute_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                      stream: Literal["text", "image", "both"]):
    """ 
        recompute_fn will get as input the input tensor and the output tensor for such stream
        and returns what should be the new modified output
    """

    # Case 1: no kwards are passed to the module
    if len(args) == 3:
        module, input, output = args
    # Case 2: when kwargs are passed to the model as input
    elif len(args) == 4:
        module, input, kwinput, output = args
    else: 
        raise AssertionError(f'Weird len(args):{len(args)}')

    if isinstance(module, FluxTransformerBlock):

        if stream == 'text':
            output_text = recompute_fn(kwinput["encoder_hidden_states"], output[0])
            output_image = output[1]
        elif stream == 'image':
            output_image = recompute_fn(kwinput["hidden_states"], output[1])
            output_text = output[0]
        else:
            raise AssertionError("Branch not supported for this layer.")

        return _safe_clip(output_text), _safe_clip(output_image)

    elif isinstance(module, FluxSingleTransformerBlock):
        
        if stream == 'text':
            output[:, :512] = recompute_fn(kwinput["hidden_states"][:, :512], output[:, :512])
        elif stream == 'image':
            output[:, 512:] = recompute_fn(kwinput["hidden_states"][:, 512:], output[:, 512:])
        else:
            output = recompute_fn(kwinput["hidden_states"], output)
        
        return _safe_clip(output)
    