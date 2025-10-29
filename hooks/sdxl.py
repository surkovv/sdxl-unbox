import torch


def _ensure_activation_map(activation_map: torch.Tensor) -> torch.Tensor:
    if activation_map.dim() == 2:
        activation_map = activation_map.unsqueeze(0)
    return activation_map


def _sdxl_diff(output, input, sae) -> torch.Tensor:
    return (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)


def _decode_to_latent(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 3, 1, 2).to(reference.device, reference.dtype)


def _sdxl_mask_from_map(encoded: torch.Tensor, feature_idx: int, activation_map: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(encoded, device=encoded.device)
    mask[..., feature_idx] = activation_map.to(mask.device)
    return mask


def _sdxl_mask_from_value(encoded: torch.Tensor, feature_idx: int, value: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(encoded, device=encoded.device)
    mask[..., feature_idx] = value.to(mask.device)
    return mask


@torch.no_grad()
def add_feature(sae, feature_idx, value, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    encoded = sae.encode(diff)
    mask = _sdxl_mask_from_value(encoded, feature_idx, value)
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + _decode_to_latent(to_add, output[0]),)


@torch.no_grad()
def add_feature_on_area_base(sae, feature_idx, activation_map, module, input, output):
    return add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output)


@torch.no_grad()
def add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    encoded = sae.encode(diff)
    activation_map = _ensure_activation_map(activation_map)
    mask = _sdxl_mask_from_map(encoded, feature_idx, activation_map)
    to_add = mask @ sae.decoder.weight.T
    uncond_delta, cond_delta = to_add.chunk(2)
    output[0][0] -= _decode_to_latent(uncond_delta, output[0])[0]
    output[0][1] += _decode_to_latent(cond_delta, output[0])[0]
    return output


@torch.no_grad()
def add_feature_on_area_base_cond(sae, feature_idx, activation_map, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    _, diff_cond = diff.chunk(2)
    encoded = sae.encode(diff_cond)
    activation_map = _ensure_activation_map(activation_map)
    mask = _sdxl_mask_from_map(encoded, feature_idx, activation_map)
    to_add = mask @ sae.decoder.weight.T
    output[0][1] += _decode_to_latent(to_add, output[0])[0]
    return output


@torch.no_grad()
def replace_with_feature_base(sae, feature_idx, value, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    _, diff_cond = diff.chunk(2)
    encoded = sae.encode(diff_cond)
    mask = _sdxl_mask_from_value(encoded, feature_idx, value)
    to_add = mask @ sae.decoder.weight.T
    input[0][1] += _decode_to_latent(to_add, output[0])[0]
    return input


@torch.no_grad()
def add_feature_on_area_turbo(sae, feature_idx, activation_map, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    encoded = sae.encode(diff)
    activation_map = _ensure_activation_map(activation_map)
    mask = _sdxl_mask_from_map(encoded, feature_idx, activation_map)
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + _decode_to_latent(to_add, output[0]),)


@torch.no_grad()
def replace_with_feature_turbo(sae, feature_idx, value, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    encoded = sae.encode(diff)
    mask = _sdxl_mask_from_value(encoded, feature_idx, value)
    to_add = mask @ sae.decoder.weight.T
    return (input[0] + _decode_to_latent(to_add, output[0]),)


@torch.no_grad()
def reconstruct_sae_hook(sae, module, input, output):
    diff = _sdxl_diff(output, input, sae)
    activated = sae.encode(diff)
    reconstructed = sae.decoder(activated) + sae.pre_bias
    return (input[0] + _decode_to_latent(reconstructed, output[0]),)


@torch.no_grad()
def ablate_block(module, input, output):
    return input
