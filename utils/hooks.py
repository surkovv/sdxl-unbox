import torch

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
