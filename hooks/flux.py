import torch


@torch.no_grad()
def add_feature_on_area_flux(
    sae,
    feature_idx,
    activation_map,
    module,
    input: torch.Tensor,
    output: torch.Tensor,
):
    diff = (output - input).to(sae.device)
    encoded = sae.encode(diff)
    flattened_map = activation_map.flatten()
    mask = torch.zeros_like(encoded, device=encoded.device)
    mask[..., feature_idx] = flattened_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    return output + to_add.to(output.device, output.dtype)
