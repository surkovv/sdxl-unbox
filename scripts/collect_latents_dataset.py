import datetime
import io
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
import webdataset as wds
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_interfaces import MODEL_SPECS, create_model_adapter  # noqa: E402
from utils.loaders import load_pipeline, resolve_dtype  # noqa: E402

import fire


def _sanitize_block_name(block_code: str) -> str:
    return block_code.replace(".", "_").replace("/", "_")


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    buffer.seek(0)
    return buffer.read()


def _serialize_numpy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return buffer.read()


def _serialize_json(data: Dict) -> bytes:
    return json.dumps(data).encode("utf-8")


def _collect_sdxl_tensors(cache: Dict[str, Dict[str, torch.Tensor]], module_path: str) -> Dict[str, bytes]:
    output_tensor = cache["output"][module_path]
    diff_tensor = cache["output"][module_path] - cache["input"][module_path]
    return {
        "output.pth": _serialize_tensor(output_tensor),
        "diff.pth": _serialize_tensor(diff_tensor),
    }


def _collect_flux_tensors(cache, include_text_stream: bool = True) -> Dict[str, bytes]:
    tensors = {}
    if getattr(cache, "image_activation", None):
        image_activation = torch.stack(cache.image_activation)
        tensors["image_activation.pth"] = _serialize_tensor(image_activation)
    if include_text_stream and getattr(cache, "text_activation", None):
        text_activation = torch.stack(cache.text_activation)
        tensors["text_activation.pth"] = _serialize_tensor(text_activation)
    return tensors


def main(
    save_path,
    model_id: str = "stabilityai/sdxl-turbo",
    start_at: int = 0,
    finish_at: int = 30000,
    dataset_batch_size: int = 50,
    num_inference_steps: int = None,
    guidance_scale: float = None,
):
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Model {model_id} is not configured.")

    dtype = resolve_dtype(model_id)
    pipe = load_pipeline(model_id, dtype)
    adapter = create_model_adapter(pipe)
    spec = adapter.spec

    steps = spec.steps if num_inference_steps is None else int(num_inference_steps)
    guidance = spec.guidance_scale if guidance_scale is None else float(guidance_scale)
    block_map = spec.code_to_block
    positions_to_cache = list(block_map.values())

    dataset = load_dataset(
        "guangyil/laion-coco-aesthetic",
        split="train",
        columns=["caption"],
        streaming=True,
    ).shuffle(seed=42)
    dataloader = DataLoader(dataset, batch_size=dataset_batch_size)

    ct = datetime.datetime.now()
    save_path = os.path.join(save_path, str(ct))
    os.makedirs(save_path, exist_ok=True)

    block_writers = {
        code: wds.TarWriter(f"{save_path}/{_sanitize_block_name(code)}.tar")
        for code in block_map.keys()
    }
    image_writer = wds.TarWriter(f"{save_path}/images.tar")

    for batch_index, batch in tqdm(enumerate(dataloader), unit="batch"):
        if batch_index < start_at:
            continue
        if batch_index >= finish_at:
            break

        prompts = batch["caption"]
        batch_size = len(prompts)
        if batch_size == 0:
            continue

        kwargs_to_save = {
            "prompt": prompts,
            "positions_to_cache": positions_to_cache,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": batch_index,
            "model_id": model_id,
        }

        if model_id.startswith("black-forest-labs/FLUX"):
            output = pipe.run(
                prompts,
                num_inference_steps=steps,
                width=1024,
                height=1024,
                cache_activations=True,
                guidance_scale=guidance,
                positions_to_cache=positions_to_cache,
                inverse=False,
                seed=batch_index,
            )
            cache = pipe.activation_cache
        else:
            generators = [
                torch.Generator(device="cpu").manual_seed(batch_index * dataset_batch_size + i)
                for i in range(batch_size)
            ]
            output, cache = pipe.run_with_cache(
                prompts,
                positions_to_cache=positions_to_cache,
                num_inference_steps=steps,
                generator=generators,
                guidance_scale=guidance,
                save_input=True,
                save_output=True,
            )

        sample_key = f"sample_{batch_index}"
        image_array = np.stack([np.array(image) for image in output.images])
        image_writer.write(
            {
                "__key__": sample_key,
                "images.npy": _serialize_numpy(image_array),
                "gen_args.json": _serialize_json(kwargs_to_save),
            }
        )

        for block_code, module_path in block_map.items():
            if model_id.startswith("black-forest-labs/FLUX"):
                tensor_payloads = _collect_flux_tensors(cache)
            else:
                tensor_payloads = _collect_sdxl_tensors(cache, module_path)

            block_sample = {"__key__": sample_key, "gen_args.json": _serialize_json(kwargs_to_save)}
            block_sample.update(tensor_payloads)
            block_writers[block_code].write(block_sample)

    image_writer.close()
    for writer in block_writers.values():
        writer.close()


if __name__ == "__main__":
    fire.Fire(main)
