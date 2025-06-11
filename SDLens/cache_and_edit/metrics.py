import numpy as np
from PIL import Image
from typing import Union
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor



def masked_mse_tiled_mask(
    image1_pil: Image.Image, 
    image2_pil: Image.Image, 
    tile_mask: Union[np.ndarray, torch.Tensor], 
    tile_size: int = 16
) -> float:
    # Convert images to float32 numpy arrays, normalized [0, 1]
    img1 = np.asarray(image1_pil).astype(np.float32) / 255.0
    img2 = np.asarray(image2_pil).astype(np.float32) / 255.0

    # Convert mask to numpy if it's a torch tensor
    if isinstance(tile_mask, torch.Tensor):
        tile_mask = tile_mask.detach().cpu().numpy()
    
    tile_mask = tile_mask.astype(np.float32)

    # Upsample mask using np.kron to match image resolution
    upsampled_mask = np.expand_dims(np.kron(tile_mask, np.ones((tile_size, tile_size), dtype=np.float32)), axis=-1)

    # Invert mask: 1 = exclude → 0; 0 = include → 1
    include_mask = 1.0 - upsampled_mask

    # Compute squared difference
    diff_squared = (img1 - img2) ** 2
    masked_diff = diff_squared * include_mask

    # Sum and normalize by valid (included) pixels
    valid_pixel_count = np.sum(include_mask)
    if valid_pixel_count == 0:
        raise ValueError("All pixels are masked out. Cannot compute MSE.")

    mse = np.sum(masked_diff) / valid_pixel_count
    return float(mse)


def compute_clip_similarity(image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP similarity between a PIL image and a text prompt.
    Loads CLIP model only once and caches it.
    
    Args:
        image (PIL.Image.Image): Input image.
        prompt (str): Text prompt.
    
    Returns:
        float: Cosine similarity between image and text.
    """
    if not hasattr(compute_clip_similarity, "model"):
        compute_clip_similarity.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        compute_clip_similarity.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        compute_clip_similarity.model.eval()

    model = compute_clip_similarity.model
    processor = compute_clip_similarity.processor

    image = image.convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=[prompt], return_tensors="pt")


    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (image_features @ text_features.T).item()

    return similarity


def compute_dinov2_similarity(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute perceptual similarity between two images using DINOv2 embeddings.
    
    Args:
        image1 (PIL.Image.Image): First image.
        image2 (PIL.Image.Image): Second image.
        
    Returns:
        float: Cosine similarity between DINOv2 embeddings of the images.
    """
    # Load model and processor only once
    if not hasattr(compute_dinov2_similarity, "model"):
        compute_dinov2_similarity.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        compute_dinov2_similarity.model = AutoModel.from_pretrained("facebook/dinov2-base")
        compute_dinov2_similarity.model.eval()

    processor = compute_dinov2_similarity.processor
    model = compute_dinov2_similarity.model

    # Preprocess both images
    inputs = processor(images=[image1.convert("RGB"), image2.convert("RGB")], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)  # [CLS] or mean-pooled features

        # Normalize
        features = F.normalize(features, p=2, dim=-1)

        # Cosine similarity
        similarity = (features[0] @ features[1].T).item()

    return similarity
