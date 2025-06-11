from typing import Optional, Tuple
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from cache_and_edit import CachedPipeline
import numpy as np
from IPython.display import display

from cache_and_edit.flux_pipeline import EditedFluxPipeline

def image2latent(pipe, image, latent_nudging_scalar = 1.15):
    image = pipe.image_processor.preprocess(image).type(pipe.vae.dtype).to("cuda")
    latents = pipe.vae.encode(image)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * latent_nudging_scalar

    latents = pipe._pack_latents(
        latents=latents,
        batch_size=1,
        num_channels_latents=16,
        height=image.size(2) // 8,
        width= image.size(3) // 8
    )

    return latents


def get_inverted_input_noise(pipe: CachedPipeline, 
                             image, 
                             prompt: str = "",
                             num_steps: int = 28,
                             latent_nudging_scalar: int = 1.15):
    """_summary_

    Args:
        pipe (CachedPipeline): _description_
        image (_type_): _description_
        num_steps (int, optional): _description_. Defaults to 28.

    Returns:
        _type_: _description_
    """

    width, height = image.size 
    inverted_latents_list = []

    if isinstance(pipe.pipe, EditedFluxPipeline):

        _ = pipe.run(
            prompt,
            num_inference_steps=num_steps,
            seed=42,
            guidance_scale=1,
            output_type="latent",
            latents=image2latent(pipe.pipe, image, latent_nudging_scalar=latent_nudging_scalar),
            empty_clip_embeddings=False,
            inverse=True,
            width=width,
            height=height,
            is_inverted_generation=True,
            inverted_latents_list=inverted_latents_list
        ).images[0]

        return inverted_latents_list

    
    else:
        noise = pipe.run(
            prompt,
            num_inference_steps=num_steps,
            seed=42,
            guidance_scale=1,
            output_type="latent",
            latents=image2latent(pipe.pipe, image, latent_nudging_scalar=latent_nudging_scalar),
            empty_clip_embeddings=False,
            inverse=True,
            width=width,
            height=height
        ).images[0]

        return noise
    



def resize_bounding_box(
    bb_mask: torch.Tensor,
    target_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """
    Given a bounding box mask, patches it into a mask with the target size.
    The mask is a 2D tensor of shape (H, W) where each element is either 0 or 1.
    Any patch that contains at least one 1 in the original mask will be set to 1 in the output mask.

    Args:
        bb_mask (torch.Tensor): The bounding box mask as a boolean tensor of shape (H, W).
        target_size (Tuple[int, int]): The size of the target mask as a tuple (H, W).

    Returns:
        torch.Tensor: The resized bounding box mask as a boolean tensor of shape (H, W).
    """
    
    w_mask, h_mask = bb_mask.shape[-2:]
    w_target, h_target = target_size

    # Make sure the sizes are compatible
    if w_mask % w_target != 0 or h_mask % h_target != 0:
        raise ValueError(
            f"Mask size {bb_mask.shape[-2:]} is not compatible with target size {target_size}"
        )
    
    # Compute the size of a patch
    patch_size = (w_mask // w_target, h_mask // h_target)

    # Iterate over the mask, one patch at a time, and save a 0 patch if the patch is empty or a 1 patch if the patch is not empty
    out_mask = torch.zeros((w_target, h_target), dtype=bb_mask.dtype, device=bb_mask.device)
    for i in range(w_target):
        for j in range(h_target):
            patch = bb_mask[
                i * patch_size[0] : (i + 1) * patch_size[0],
                j * patch_size[1] : (j + 1) * patch_size[1],
            ]
            if torch.sum(patch) > 0:
                out_mask[i, j] = 1
            else:
                out_mask[i, j] = 0

    return out_mask


def place_image_in_bounding_box(
    image_tensor_whc: torch.Tensor, 
    mask_tensor_wh: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resizes an input image to fit within a bounding box (from a mask)
    preserving aspect ratio, and places it centered on a new canvas.

    Args:
        image_tensor_whc: Input image tensor, shape [width, height, channels].
        mask_tensor_wh: Bounding box mask, shape [width, height]. Defines canvas size
                          and contains a rectangle of 1s for the BB.

    Returns:
        A tuple:
        - output_image_whc (torch.Tensor): Canvas with the resized image placed.
                                           Shape [canvas_width, canvas_height, channels].
        - new_mask_wh (torch.Tensor): Mask showing the actual placement of the image.
                                      Shape [canvas_width, canvas_height].
    """
    
    # Validate input image dimensions
    if not (image_tensor_whc.ndim == 3 and image_tensor_whc.shape[0] > 0 and image_tensor_whc.shape[1] > 0):
        raise ValueError(
            "Input image_tensor_whc must be a 3D tensor [width, height, channels] "
            "with width > 0 and height > 0."
        )
    img_orig_w, img_orig_h, num_channels = image_tensor_whc.shape

    # Validate mask tensor dimensions
    if not (mask_tensor_wh.ndim == 2):
        raise ValueError("Input mask_tensor_wh must be a 2D tensor [width, height].")
    canvas_w, canvas_h = mask_tensor_wh.shape

    # Prepare default empty outputs for early exit scenarios
    empty_output_image = torch.zeros(
        canvas_w, canvas_h, num_channels, 
        dtype=image_tensor_whc.dtype, device=image_tensor_whc.device
    )
    empty_new_mask = torch.zeros(
        canvas_w, canvas_h, 
        dtype=mask_tensor_wh.dtype, device=mask_tensor_wh.device
    )

    # 1. Find Bounding Box (BB) coordinates from the input mask_tensor_wh
    #    fg_coords shape: [N, 2], where N is num_nonzero. Each row: [x_coord, y_coord].
    fg_coords = torch.nonzero(mask_tensor_wh, as_tuple=False) 
    
    if fg_coords.numel() == 0: # No bounding box found in mask
        return empty_output_image, empty_new_mask

    # Determine min/max extents of the bounding box
    x_min_bb, y_min_bb = fg_coords[:, 0].min(), fg_coords[:, 1].min()
    x_max_bb, y_max_bb = fg_coords[:, 0].max(), fg_coords[:, 1].max()

    bb_target_w = x_max_bb - x_min_bb + 1
    bb_target_h = y_max_bb - y_min_bb + 1

    if bb_target_w <= 0 or bb_target_h <= 0: # Should not happen if fg_coords not empty
        return empty_output_image, empty_new_mask

    # 2. Prepare image for resizing: TF.resize expects [C, H, W]
    #    Input image_tensor_whc is [W, H, C]. Permute to [C, H_orig, W_orig].
    image_tensor_chw = image_tensor_whc.permute(2, 1, 0) 

    # 3. Calculate new dimensions for the image to fit in BB, preserving aspect ratio
    scale_factor_w = bb_target_w / img_orig_w
    scale_factor_h = bb_target_h / img_orig_h
    scale = min(scale_factor_w, scale_factor_h) # Fit entirely within BB

    resized_img_w = int(img_orig_w * scale)
    resized_img_h = int(img_orig_h * scale)
    
    if resized_img_w == 0 or resized_img_h == 0: # Image scaled to nothing
        return empty_output_image, empty_new_mask
        
    # 4. Resize the image. TF.resize expects size as [H, W].
    try:
        # antialias=True for better quality (requires torchvision >= 0.8.0 approx)
        resized_image_chw = TF.resize(image_tensor_chw, [resized_img_h, resized_img_w], antialias=True)
    except TypeError: # Fallback for older torchvision versions
        resized_image_chw = TF.resize(image_tensor_chw, [resized_img_h, resized_img_w])

    # Permute resized image back to [W, H, C] format
    resized_image_whc = resized_image_chw.permute(2, 1, 0)

    # 5. Create the output canvas image (initialized to zeros)
    output_image_whc = torch.zeros(
        canvas_w, canvas_h, num_channels, 
        dtype=image_tensor_whc.dtype, device=image_tensor_whc.device
    )

    # 6. Calculate pasting coordinates to center the resized image within the original BB
    offset_x = (bb_target_w - resized_img_w) // 2
    offset_y = (bb_target_h - resized_img_h) // 2

    paste_x_start = x_min_bb + offset_x
    paste_y_start = y_min_bb + offset_y

    paste_x_end = paste_x_start + resized_img_w
    paste_y_end = paste_y_start + resized_img_h
    
    # Place the resized image onto the canvas
    output_image_whc[paste_x_start:paste_x_end, paste_y_start:paste_y_end, :] = resized_image_whc

    # 7. Create the new mask representing where the image was actually placed
    new_mask_wh = torch.zeros(
        canvas_w, canvas_h, 
        dtype=mask_tensor_wh.dtype, device=mask_tensor_wh.device
    )
    new_mask_wh[paste_x_start:paste_x_end, paste_y_start:paste_y_end] = 1

    return output_image_whc, new_mask_wh



### Function to cut image and put it in bounding box (either cut or not cut)
def compose_noise_masks(cached_pipe,
                  foreground_image: Image, 
                  background_image: Image, 
                  target_mask: torch.Tensor,
                  foreground_mask: torch.Tensor,
                  option: str = "bg", # bg, bg_fg, segmentation1, tf_icon
                  photoshop_fg_noise: bool = False,
                  num_inversion_steps: int = 100,
                  ):
    
    """
    Composes noise masks for image generation using different strategies.
    This function composes noise masks for stable diffusion inversion, with several composition strategies:
    - "bg": Uses only background noise
    - "bg_fg": Combines background and foreground noise using a target mask
    - "segmentation1": Uses segmentation mask to compose foreground and background noise
    - "segmentation2": Implements advanced composition with additional boundary noise
    Parameters:
    ----------
    cached_pipe : object
        The cached stable diffusion pipeline used for noise inversion
    foreground_image : PIL.Image
        The foreground image to be placed in the background
    background_image : PIL.Image
        The background image
    target_mask : torch.Tensor
        Target mask indicating the position where the foreground should be placed
    foreground_mask : torch.Tensor
        Segmentation mask of the foreground object
    option : str, default="bg"
        Composition strategy: "bg", "bg_fg", "segmentation1", or "segmentation2"
    photoshop_fg_noise : bool, default=False
        Whether to generate noise from a photoshopped composition of foreground and background
    num_inversion_steps : int, default=100
        Number of steps for the inversion process
    Returns:
    -------
    dict
        A dictionary containing:
        - "noise": Dictionary of generated noises (composed_noise, foreground_noise, background_noise)
        - "latent_masks": Dictionary of latent masks used for composition
    """
    
    # assert options
    assert option in ["bg", "bg_fg", "segmentation1", "segmentation2"], f"Invalid option: {option}"
    
    # calculate size of latent noise for mask resizing
    PATCH_SIZE = 16
    latent_size = background_image.size[0] // PATCH_SIZE
    latents = (latent_size, latent_size)

    # process the options
    if option == "bg":
        # only background noise
        bg_noise = get_inverted_input_noise(cached_pipe, background_image, num_steps=num_inversion_steps)
        composed_noise = bg_noise

        all_noise = {
                "composed_noise": composed_noise,
                "background_noise": bg_noise,
                }
        all_latent_masks = {}


    elif option == "bg_fg":

        # resize and scale the image to the bounding box
        reframed_fg_img, resized_mask = place_image_in_bounding_box(
        torch.from_numpy(np.array(foreground_image)),
        (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )

        #print("Placed Foreground Image")
        reframed_fg_img = Image.fromarray(reframed_fg_img.numpy())
        #display(reframed_fg_img)

        #print("Placed Mask")
        resized_mask_img = Image.fromarray((resized_mask.numpy() * 255).astype(np.uint8))
        #display(resized_mask_img)

        # invert resized & padded image
        if photoshop_fg_noise:
            #print("Photoshopping FG IMAGE")
            photoshop_img = Image.fromarray(
                (torch.tensor(np.array(background_image)) * ~resized_mask.cpu().unsqueeze(-1) + torch.tensor(np.array(reframed_fg_img)) * resized_mask.cpu().unsqueeze(-1)).numpy()
            )
            #display(photoshop_img)
            fg_noise = get_inverted_input_noise(cached_pipe, photoshop_img, num_steps=num_inversion_steps)
        else:
            fg_noise = get_inverted_input_noise(cached_pipe, reframed_fg_img, num_steps=num_inversion_steps)
        bg_noise = get_inverted_input_noise(cached_pipe, background_image, num_steps=num_inversion_steps)

        # overwrite get masked in latent space
        latent_mask = resize_bounding_box(
            resized_mask,
            target_size=latents,
                ).flatten().unsqueeze(-1).to("cuda")

        # compose the noise
        composed_noise = bg_noise * (~latent_mask) + fg_noise * latent_mask
        all_latent_masks = {
            "latent_mask": latent_mask,
                }
        all_noise = {
                "composed_noise": composed_noise,
                "foreground_noise": fg_noise,
                "background_noise": bg_noise,
                    }
        
    elif option == "segmentation1":
        # cut out the object and compose it with the background noise
        
        # segmented foreground image
        segmented_fg_image = torch.tensor(
        np.array(
        foreground_mask.resize(foreground_image.size)
        )).to(torch.bool).unsqueeze(-1) * torch.tensor(
            np.array(foreground_image)
            )
        
        # resize and scale the image to the bounding box
        reframed_fg_img, resized_mask = place_image_in_bounding_box(
        segmented_fg_image,
        (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )

        reframed_fg_img = Image.fromarray(reframed_fg_img.numpy())
        #display(reframed_fg_img)

        resized_mask_img = Image.fromarray((resized_mask.numpy() * 255).astype(np.uint8))

        # resize and scale the mask itself
        foreground_mask = foreground_mask.convert("RGB") # to avoid extraction of contours and make work with function
        reframed_segmentation_mask, resized_mask = place_image_in_bounding_box(
            torch.from_numpy(np.array(foreground_mask)),
            (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )

        reframed_segmentation_mask = reframed_segmentation_mask.numpy()
        reframed_segmentation_mask_img = Image.fromarray(reframed_segmentation_mask)
        #print("Placed Segmentation Mask")
        #display(reframed_segmentation_mask_img)

        # invert resized & padded image 
        # fg_noise = get_inverted_input_noise(cached_pipe, reframed_fg_img, num_steps=num_inversion_steps)

        if photoshop_fg_noise:
            # temporarily convert to apply mask
            #print("Photoshopping FG IMAGE")
            seg_mask_temp = torch.from_numpy(reframed_segmentation_mask).bool()
            bg_temp = torch.tensor(np.array(background_image))
            fg_temp = torch.tensor(np.array(reframed_fg_img))

            photoshop_img = Image.fromarray(
                (bg_temp * (~seg_mask_temp) + fg_temp * seg_mask_temp).numpy()
            ).convert("RGB")
            #display(photoshop_img)
            fg_noise = get_inverted_input_noise(cached_pipe, photoshop_img, num_steps=num_inversion_steps)
        else:
            fg_noise = get_inverted_input_noise(cached_pipe, reframed_fg_img, num_steps=num_inversion_steps)


        bg_noise = get_inverted_input_noise(cached_pipe, background_image, num_steps=num_inversion_steps)
        bg_noise_init = bg_noise[-1].squeeze(0) if isinstance(bg_noise, list) else bg_noise
        fg_noise_init = fg_noise[-1].squeeze(0) if isinstance(fg_noise, list) else fg_noise

        # overwrite background in resized mask
        # convert mask from 512x512x3 to 512x512 first
        reframed_segmentation_mask = reframed_segmentation_mask[:, :, 0]
        reframed_segmentation_mask = torch.from_numpy(reframed_segmentation_mask).to(dtype=bool)
        latent_mask = resize_bounding_box(
            reframed_segmentation_mask,
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")
        bb_mask = resize_bounding_box(
            resized_mask,
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")

        # compose noise
        composed_noise = bg_noise_init * (~latent_mask) + fg_noise_init * latent_mask

        all_latent_masks = {
            "latent_segmentation_mask": latent_mask,
            # FIXME: handle bounding box better (making sure shapes are correct, especially when bg and fg images have different sizes, e.g. test image 69)
            "bb_mask": bb_mask,
            }
        all_noise = {
                "composed_noise": composed_noise,
                "foreground_noise": fg_noise_init,
                "background_noise": bg_noise_init,
                "foreground_noise_list": fg_noise if isinstance(fg_noise, list) else None,
                "background_noise_list": bg_noise if isinstance(bg_noise, list) else None,
        }

        
    elif option == "segmentation2":
        # add random noise in the background

        # segmented foreground image
        segmented_fg_image = torch.tensor(
        np.array(
        foreground_mask.resize(foreground_image.size)
        )).to(torch.bool).unsqueeze(-1) * torch.tensor(
            np.array(foreground_image)
            )
        
        # resize and scale the image to the bounding box
        reframed_fg_img, resized_mask = place_image_in_bounding_box(
        segmented_fg_image,
        (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )

        #print("Segmented and Placed FG Image")
        reframed_fg_img = Image.fromarray(reframed_fg_img.numpy())
        #display(reframed_fg_img)

        # resize and scale the mask itself
        foreground_mask = foreground_mask.convert("RGB")
        reframed_segmentation_mask, resized_mask = place_image_in_bounding_box(
            torch.from_numpy(np.array(foreground_mask)),
            (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )

        reframed_segmentation_mask = reframed_segmentation_mask.numpy()
        reframed_segmentation_mask_img = Image.fromarray(reframed_segmentation_mask)
        #print("Reframed Segmentation Mask")
        #display(reframed_segmentation_mask_img)

        xor_mask = target_mask ^ np.array(reframed_segmentation_mask_img.convert("L"))
        #print("XOR Mask")
        #display(Image.fromarray(xor_mask))

        # invert resized & padded image 
        # fg_noise = get_inverted_input_noise(cached_pipe, reframed_fg_img, num_steps=num_inversion_steps)
        if photoshop_fg_noise:
            #print("Photoshopping FG IMAGE")
            # temporarily convert to apply mask
            seg_mask_temp = torch.from_numpy(reframed_segmentation_mask).bool()
            bg_temp = torch.tensor(np.array(background_image))
            fg_temp = torch.tensor(np.array(reframed_fg_img))

            photoshop_img = Image.fromarray(
                (bg_temp * (~seg_mask_temp) + fg_temp * seg_mask_temp).numpy()
            ).convert("RGB")
            #display(photoshop_img)
            fg_noise = get_inverted_input_noise(cached_pipe, photoshop_img, num_steps=num_inversion_steps)
        else:
            fg_noise = get_inverted_input_noise(cached_pipe, reframed_fg_img, num_steps=num_inversion_steps)
        bg_noise = get_inverted_input_noise(cached_pipe, background_image, num_steps=num_inversion_steps)

        # overwrite background in resized mask
        # convert mask from 512x512x3 to 512x512
        reframed_segmentation_mask = reframed_segmentation_mask[:, :, 0]
        reframed_segmentation_mask = torch.from_numpy(reframed_segmentation_mask).to(dtype=bool)

        # get all masks in latents and move to device
        latent_seg_mask = resize_bounding_box(
            reframed_segmentation_mask,
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")
        print(latent_seg_mask.shape)


        latent_xor_mask = resize_bounding_box(
            torch.from_numpy(xor_mask),
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")


        print(resized_mask.shape)
        latent_target_mask = resize_bounding_box(
            resized_mask,
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")

        # implement x∗T = xrT ⊙Mseg +xmT ⊙(1−Muser)+z⊙(Muser ⊕Mseg)
        bg_noise_init = bg_noise[-1].squeeze(0) if isinstance(bg_noise, list) else bg_noise
        fg_noise_init = fg_noise[-1].squeeze(0) if isinstance(fg_noise, list) else fg_noise

        bg = bg_noise_init[-1] * (~latent_target_mask)
        fg = fg_noise_init[-1] * latent_seg_mask
        boundary = latent_xor_mask * torch.randn(latent_xor_mask.shape).to("cuda")
        composed_noise = bg + fg + boundary

        all_latent_masks = {
            "latent_target_mask": latent_target_mask,
            "latent_segmentation_mask": latent_seg_mask,
            "latent_xor_mask": latent_xor_mask,
                            }
        all_noise = {
                "composed_noise": composed_noise,
                "foreground_noise": fg_noise_init,
                "background_noise": bg_noise_init,
                "foreground_noise_list": fg_noise if isinstance(fg_noise, list) else None,
                "background_noise_list": bg_noise if isinstance(bg_noise, list) else None,
                    }
    
    # always add latent bbox mask (for bg consistency or any other future application)
    latent_bbox_mask = resize_bounding_box(
        torch.from_numpy(np.array(target_mask.resize(background_image.size))), # reseize just to be sure
        target_size=latents,
    ).flatten().unsqueeze(-1).to("cuda")
    all_latent_masks["latent_bbox_mask"] = latent_bbox_mask
    
    # always add latent segmentation mkas
    reframed_fg_img, resized_mask = place_image_in_bounding_box(
        torch.from_numpy(np.array(foreground_image)),
        (torch.from_numpy(np.array(target_mask)) / 255.0).to(dtype=bool)
        )
    bb_mask = resize_bounding_box(
            resized_mask,
            target_size=latents,
        ).flatten().unsqueeze(-1).to("cuda")
    all_latent_masks["latent_segmentation_mask"] = bb_mask
    
    # output 
    return {
        "noise": all_noise,
        "latent_masks": all_latent_masks,
            }