from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from huggingface_hub import hf_hub_download

from SDLens import HookedStableDiffusionXLPipeline, CachedPipeline as CachedFluxPipeline
from SDLens.cache_and_edit.flux_pipeline import EditedFluxPipeline
from SAE import SparseAutoencoder
from model_interfaces import MODEL_SPECS


DEVICE = "cuda"
CHECKPOINT_ROOT = Path("checkpoints")
FLUX_REPO_ID = "antoniomari/SAE_flux_18"
FLUX_BRANCH = "main"


def resolve_dtype(model_id: str) -> torch.dtype:
    """Select the appropriate torch dtype for the given model."""
    return torch.float16 if model_id.startswith("black-forest-labs/FLUX") else torch.float32


def load_pipeline(model_id: str, torch_dtype: torch.dtype):
    """Instantiate and configure the diffusion pipeline for the requested model."""
    if model_id.startswith("black-forest-labs/FLUX"):
        base_pipe = EditedFluxPipeline.from_pretrained(
            model_id,
            device_map="balanced",
            torch_dtype=torch_dtype,
        )
        base_pipe.set_progress_bar_config(disable=True)
        return CachedFluxPipeline(base_pipe)

    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="balanced",
        variant=("fp16" if torch_dtype == torch.float16 else None),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_flux_sae_bundle(model_id: str, torch_dtype: torch.dtype):
    spec = MODEL_SPECS[model_id]
    saes_dict: Dict[str, SparseAutoencoder] = {}
    means_dict: Dict[str, torch.Tensor] = {}

    block_code = next(iter(spec.code_to_block.keys()))
    state_dict_path = hf_hub_download(
        repo_id=FLUX_REPO_ID,
        filename="state_dict.pth",
        revision=FLUX_BRANCH,
    )
    config_path = hf_hub_download(
        repo_id=FLUX_REPO_ID,
        filename="config.json",
        revision=FLUX_BRANCH,
    )
    mean_path = hf_hub_download(
        repo_id=FLUX_REPO_ID,
        filename="mean.pt",
        revision=FLUX_BRANCH,
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    sae = SparseAutoencoder(**config)
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    sae.load_state_dict(state_dict)
    sae = sae.to(DEVICE, dtype=torch_dtype).eval()

    means = torch.load(mean_path, map_location="cpu")
    means = means.to(torch_dtype).to(DEVICE)

    saes_dict[block_code] = sae
    means_dict[block_code] = means

    return saes_dict, means_dict


def _load_sdxl_sae_bundle(model_id: str, torch_dtype: torch.dtype):
    spec = MODEL_SPECS[model_id]
    saes_dict: Dict[str, SparseAutoencoder] = {}
    means_dict: Dict[str, torch.Tensor] = {}

    for code, block in spec.code_to_block.items():
        checkpoint_dir = CHECKPOINT_ROOT / f"{block}_k10_hidden5120_auxk256_bs4096_lr0.0001" / "final"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

        sae = SparseAutoencoder.load_from_disk(str(checkpoint_dir))
        sae = sae.to(DEVICE, dtype=torch_dtype).eval()

        mean_tensor = torch.load(checkpoint_dir / "mean.pt", map_location="cpu")
        mean_tensor = mean_tensor.to(torch_dtype).to(DEVICE)

        saes_dict[code] = sae
        means_dict[code] = mean_tensor

    return saes_dict, means_dict


def load_sae_bundle(model_id: str, torch_dtype: torch.dtype) -> Tuple[Dict[str, SparseAutoencoder], Dict[str, torch.Tensor]]:
    """Load SAEs and corresponding mean tensors for the chosen model."""
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Model {model_id} is not configured.")

    if model_id.startswith("black-forest-labs/FLUX"):
        return _load_flux_sae_bundle(model_id, torch_dtype)

    return _load_sdxl_sae_bundle(model_id, torch_dtype)

