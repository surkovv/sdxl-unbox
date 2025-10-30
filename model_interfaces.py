from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from einops import rearrange

from hooks import (
    add_feature_on_area_base,
    add_feature_on_area_flux,
    add_feature_on_area_turbo,
    replace_with_feature_base,
    replace_with_feature_turbo,
)


class ModelAdapter(ABC):
    """Base class that exposes a homogeneous interface for model interaction."""

    def __init__(self, pipe: Any, spec: "ModelSpec"):
        self.pipe = pipe
        self.spec = spec

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def heatmap_scale(self) -> int:
        return self.spec.heatmap_scale

    @property
    def feature_icon_hook(self) -> Optional[Callable]:
        return self.spec.feature_icon_hook

    def get_block_path(self, block_code: str) -> str:
        return self.spec.code_to_block[block_code]

    @abstractmethod
    def generate_with_cache(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        positions_to_cache: List[str],
        seed: int = 42,
    ) -> Tuple[Any, Any]:
        """Run a cached forward pass and return (output, cache)."""

    @abstractmethod
    def process_cache(
        self,
        cache: Any,
        saes_dict: Dict[str, Any],
        timestep: Optional[int],
    ) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        """Convert a raw cache to (top_features, sparse_maps)."""

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        """Plain generation without caching or edits."""

    @abstractmethod
    def apply_edit(
        self,
        prompt: str,
        block_code: str,
        hook: Callable,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        """Run the pipeline while applying a hook/edit."""

    def top_image_url(self, block_code: str, feature_index: int) -> str:
        if not self.spec.top_image_repo:
            raise ValueError(f"No top image repository configured for {self.model_id}")
        return self.spec.top_image_repo.format(block=block_code, index=feature_index)


class SDXLAdapter(ModelAdapter):
    """Adapter for SDXL Base and Turbo pipelines."""

    def _generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    def generate_with_cache(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        positions_to_cache: List[str],
        seed: int = 42,
    ) -> Tuple[Any, Any]:
        images, cache = self.pipe.run_with_cache(
            prompt,
            positions_to_cache=positions_to_cache,
            num_inference_steps=int(num_steps),
            generator=self._generator(seed),
            guidance_scale=guidance_scale,
            save_input=True,
            save_output=True,
        )
        return images.images[0], cache

    def process_cache(
        self,
        cache: Dict[str, Dict[str, torch.Tensor]],
        saes_dict: Dict[str, Any],
        timestep: Optional[int],
    ) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        top_features_dict: Dict[str, List[int]] = {}
        sparse_maps_dict: Dict[str, Any] = {}

        for code, block in self.spec.code_to_block.items():
            sae = saes_dict[code]
            diff = cache["output"][block] - cache["input"][block]
            if diff.shape[0] == 2:
                diff = diff[1].unsqueeze(0)

            if timestep is not None and timestep < diff.shape[1]:
                diff = diff[:, timestep : timestep + 1]

            diff = diff.permute(0, 1, 3, 4, 2).squeeze(0)
            with torch.no_grad():
                sparse_maps = sae.encode(diff)

            flat_maps = sparse_maps.reshape(-1, sparse_maps.shape[-1])
            averages = torch.mean(flat_maps, dim=0)
            top_features = torch.topk(averages, 10).indices

            top_features_dict[code] = top_features.cpu().tolist()
            sparse_maps_dict[code] = sparse_maps.cpu().numpy()

        return top_features_dict, sparse_maps_dict

    def generate_image(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        result = self.pipe.run_with_hooks(
            prompt,
            position_hook_dict={},
            num_inference_steps=int(num_steps),
            generator=self._generator(seed),
            guidance_scale=guidance_scale,
        )
        return result.images[0]

    def apply_edit(
        self,
        prompt: str,
        block_code: str,
        hook: Callable,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        position = self.get_block_path(block_code)
        result = self.pipe.run_with_hooks(
            prompt,
            position_hook_dict={position: hook},
            num_inference_steps=int(num_steps),
            generator=self._generator(seed),
            guidance_scale=guidance_scale,
        )
        return result.images[0]


class FluxAdapter(ModelAdapter):
    """Adapter for Flux pipelines wrapped by CachedPipeline."""

    def generate_with_cache(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        positions_to_cache: List[str],
        seed: int = 42,
    ) -> Tuple[Any, Any]:
        output = self.pipe.run(
            prompt,
            num_inference_steps=int(num_steps),
            width=1024,
            height=1024,
            cache_activations=True,
            guidance_scale=guidance_scale,
            positions_to_cache=positions_to_cache,
            inverse=False,
            seed=seed,
        )
        return output.images[0], self.pipe.activation_cache

    def process_cache(
        self,
        cache: Any,
        saes_dict: Dict[str, Any],
        timestep: Optional[int],
    ) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
        top_features_dict: Dict[str, List[int]] = {}
        sparse_maps_dict: Dict[str, Any] = {}

        for code in self.spec.code_to_block.keys():
            sae = saes_dict[code]
            with torch.no_grad():
                activations = torch.stack(cache.image_activation)
                features = sae.encode(activations)

            if self.spec.exclude_list:
                features[..., self.spec.exclude_list] = 0

            if timestep is not None and timestep < features.shape[0]:
                features = features[timestep : timestep + 1]

            sparse_maps = rearrange(features, "t b (w h) n -> b t w h n", w=64, h=64).squeeze(0)
            flat_maps = sparse_maps.reshape(-1, sparse_maps.shape[-1])
            averages = torch.mean(flat_maps, dim=0)
            top_features = torch.topk(averages, 10).indices

            top_features_dict[code] = top_features.cpu().tolist()
            sparse_maps_dict[code] = sparse_maps.cpu().numpy()

        return top_features_dict, sparse_maps_dict

    def generate_image(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        output = self.pipe.run(
            prompt,
            num_inference_steps=int(num_steps),
            width=1024,
            height=1024,
            cache_activations=False,
            guidance_scale=guidance_scale,
            inverse=False,
            seed=seed,
        )
        return output.images[0]

    def apply_edit(
        self,
        prompt: str,
        block_code: str,
        hook: Callable,
        num_steps: int,
        guidance_scale: float,
        *,
        seed: int = 42,
    ) -> Any:
        output = self.pipe.run_with_edit(
            prompt,
            seed=seed,
            num_inference_steps=int(num_steps),
            edit_fn=lambda input, output: hook(None, input, output),
            layers_for_edit_fn=self.spec.edit_layers or list(range(18, 57)),
            stream="image",
            guidance_scale=guidance_scale,
        )
        return output.images[0]

    def top_image_url(self, block_code: str, feature_index: int) -> str:
        if not self.spec.top_image_repo:
            raise ValueError(f"No top image repository configured for {self.model_id}")
        part_threshold = self.spec.top_image_part_threshold or 7000
        part = 1 if feature_index <= part_threshold else 2
        folder = block_code if block_code.startswith("flux_") else f"flux_{block_code}"
        return self.spec.top_image_repo.format(block=folder, part=part, index=feature_index)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    adapter_cls: Type[ModelAdapter]
    steps: int
    max_steps: int
    guidance_scale: float
    choices: List[str]
    default_choice: str
    code_to_block: Dict[str, str]
    downsample_factor: int
    num_features: int
    add_feature_on_area: Callable
    heatmap_scale: int
    feature_icon_hook: Optional[Callable] = None
    exclude_list: Optional[List[int]] = None
    top_image_repo: Optional[str] = None
    top_image_part_threshold: Optional[int] = None
    edit_layers: Optional[List[int]] = None


def _flux_edit_layers() -> List[int]:
    return [i for i in range(18, 57)]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "stabilityai/stable-diffusion-xl-base-1.0": ModelSpec(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        adapter_cls=SDXLAdapter,
        steps=25,
        max_steps=50,
        guidance_scale=8.0,
        choices=["up.0.1 (style)", "down.2.1 (composition)", "up.0.0 (details)", "mid.0"],
        default_choice="down.2.1 (composition)",
        code_to_block={
            "down.2.1": "unet.down_blocks.2.attentions.1",
            "mid.0": "unet.mid_block.attentions.0",
            "up.0.1": "unet.up_blocks.0.attentions.1",
            "up.0.0": "unet.up_blocks.0.attentions.0",
        },
        downsample_factor=16,
        num_features=5120,
        add_feature_on_area=add_feature_on_area_base,
        heatmap_scale=32,
        feature_icon_hook=replace_with_feature_base,
        top_image_repo="https://huggingface.co/surokpro2/sdxl_sae_images/resolve/main/{block}/{index}.jpg",
    ),
    "stabilityai/sdxl-turbo": ModelSpec(
        model_id="stabilityai/sdxl-turbo",
        adapter_cls=SDXLAdapter,
        steps=1,
        max_steps=4,
        guidance_scale=0.0,
        choices=["up.0.1 (style)", "down.2.1 (composition)", "up.0.0 (details)", "mid.0"],
        default_choice="down.2.1 (composition)",
        code_to_block={
            "down.2.1": "unet.down_blocks.2.attentions.1",
            "mid.0": "unet.mid_block.attentions.0",
            "up.0.1": "unet.up_blocks.0.attentions.1",
            "up.0.0": "unet.up_blocks.0.attentions.0",
        },
        downsample_factor=32,
        num_features=5120,
        add_feature_on_area=add_feature_on_area_turbo,
        heatmap_scale=32,
        feature_icon_hook=replace_with_feature_turbo,
        top_image_repo="https://huggingface.co/surokpro2/sdxl_sae_images/resolve/main/{block}/{index}.jpg",
    ),
    "black-forest-labs/FLUX.1-schnell": ModelSpec(
        model_id="black-forest-labs/FLUX.1-schnell",
        adapter_cls=FluxAdapter,
        steps=1,
        max_steps=4,
        guidance_scale=0.0,
        choices=["18"],
        default_choice="18",
        code_to_block={"18": "transformer.transformer_blocks.18"},
        downsample_factor=8,
        num_features=12288,
        add_feature_on_area=add_feature_on_area_flux,
        heatmap_scale=16,
        exclude_list=[2462, 2974, 1577, 786, 3188, 9986, 4693, 8472, 8248, 325, 9596, 2813, 10803, 11773, 11410, 1067, 2965, 10488, 4537, 2102],
        top_image_repo="https://huggingface.co/datasets/antoniomari/flux_sae_images/resolve/main/{block}/part{part}/{index}.jpg",
        top_image_part_threshold=7000,
        edit_layers=_flux_edit_layers(),
    ),
    "black-forest-labs/FLUX.1-dev": ModelSpec(
        model_id="black-forest-labs/FLUX.1-dev",
        adapter_cls=FluxAdapter,
        steps=25,
        max_steps=50,
        guidance_scale=0.0,
        choices=["18"],
        default_choice="18",
        code_to_block={"18": "transformer.transformer_blocks.18"},
        downsample_factor=8,
        num_features=12288,
        add_feature_on_area=add_feature_on_area_flux,
        heatmap_scale=16,
        exclude_list=[2462, 2974, 1577, 786, 3188, 9986, 4693, 8472, 8248, 325, 9596, 2813, 10803, 11773, 11410, 1067, 2965, 10488, 4537, 2102],
        top_image_repo="https://huggingface.co/datasets/antoniomari/flux_sae_images/resolve/main/{block}/part{part}/{index}.jpg",
        top_image_part_threshold=7000,
        edit_layers=_flux_edit_layers(),
    ),
}


def create_model_adapter(pipe: Any) -> ModelAdapter:
    model_id = getattr(getattr(pipe, "pipe", pipe), "name_or_path", None)
    if model_id is None:
        raise ValueError("Unable to determine model identifier from pipeline.")

    if model_id not in MODEL_SPECS:
        raise KeyError(f"Model '{model_id}' is not configured.")

    spec = MODEL_SPECS[model_id]
    return spec.adapter_cls(pipe, spec)
