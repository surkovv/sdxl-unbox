from typing import Callable, List, Optional
import torch

class TimedHook:
    def __init__(self, hook_fn, total_steps, apply_at_steps=None):
        self.hook_fn = hook_fn
        self.total_steps = total_steps
        self.apply_at_steps = apply_at_steps
        self.current_step = 0

    def identity(self, module, input, output):
        return output

    def __call__(self, module, input, output):
        if self.apply_at_steps is not None:
            if self.current_step in self.apply_at_steps:
                self.__increment()
                return self.hook_fn(module, input, output)
            else:
                self.__increment()
                return self.identity(module, input, output)

        return self.identity(module, input, output)

    def __increment(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            self.current_step = 0

@torch.no_grad()
def add_feature(sae, feature_idx, value, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    mask[..., feature_idx] = value
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)

@torch.no_grad()
def add_feature_on_area_base(sae, feature_idx, activation_map, module, input, output):
    return add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output)

@torch.no_grad()
def add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output):
    # add the feature to cond and subtract from uncond
    # this assumes diff.shape[0] == 2
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    if len(activation_map) == 2:
        activation_map = activation_map.unsqueeze(0)
    mask[..., feature_idx] = mask[..., feature_idx] = activation_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    to_add = to_add.chunk(2)
    output[0][0] -= to_add[0].permute(0, 3, 1, 2).to(output[0].device)[0]
    output[0][1] += to_add[1].permute(0, 3, 1, 2).to(output[0].device)[0]
    return output


@torch.no_grad()
def add_feature_on_area_base_cond(sae, feature_idx, activation_map, module, input, output):
    # add the feature to cond
    # this assumes diff.shape[0] == 2
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    diff_uncond, diff_cond = diff.chunk(2)
    activated = sae.encode(diff_cond)
    mask = torch.zeros_like(activated, device=diff_cond.device)
    if len(activation_map) == 2:
        activation_map = activation_map.unsqueeze(0)
    mask[..., feature_idx] = mask[..., feature_idx] = activation_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    output[0][1] += to_add.permute(0, 3, 1, 2).to(output[0].device)[0]
    return output


@torch.no_grad()
def replace_with_feature_base(sae, feature_idx, value, module, input, output):
    # this assumes diff.shape[0] == 2
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    diff_uncond, diff_cond = diff.chunk(2)
    activated = sae.encode(diff_cond)
    mask = torch.zeros_like(activated, device=diff_cond.device)
    mask[..., feature_idx] = value
    to_add = mask @ sae.decoder.weight.T
    input[0][1] += to_add.permute(0, 3, 1, 2).to(output[0].device)[0]
    return input


@torch.no_grad()
def add_feature_on_area_turbo(sae, feature_idx, activation_map, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    if len(activation_map) == 2:
        activation_map = activation_map.unsqueeze(0)
    mask[..., feature_idx] = mask[..., feature_idx] = activation_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)

@torch.no_grad
def add_feature_on_area_flux(
    sae,
    feature_idx,
    activation_map,
    module,
    input: torch.Tensor, 
    output: torch.Tensor,
    ):

    diff = (output - input).to(sae.device)
    activated = sae.encode(diff)

    # TODO: check
    if len(activation_map) == 2:
        activation_map = activation_map.unsqueeze(0)
    mask = torch.zeros_like(activated, device=diff.device)
    activation_map = activation_map.flatten()
    mask[..., feature_idx] = activation_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    return output + to_add.to(output.device, output.dtype)

            

@torch.no_grad()
def replace_with_feature_turbo(sae, feature_idx, value, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    mask[..., feature_idx] = value
    to_add = mask @ sae.decoder.weight.T
    return (input[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def reconstruct_sae_hook(sae, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    reconstructed = sae.decoder(activated) + sae.pre_bias
    return (input[0] + reconstructed.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def ablate_block(module, input, output):
    return input
