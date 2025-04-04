import gradio as gr
import os
import torch
from PIL import Image
from SDLens import HookedStableDiffusionXLPipeline
from SAE import SparseAutoencoder
from utils import TimedHook, add_feature_on_area_base, replace_with_feature_base, add_feature_on_area_turbo, replace_with_feature_turbo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import threading
from retrieval import FeatureRetriever

code_to_block = {
    "down.2.1": "unet.down_blocks.2.attentions.1",
    "mid.0": "unet.mid_block.attentions.0",
    "up.0.1": "unet.up_blocks.0.attentions.1",
    "up.0.0": "unet.up_blocks.0.attentions.0"
}
lock = threading.Lock()

base_guidance_scale_default = 8.0
turbo_guidance_scale_default = 0.0

def process_cache(cache, saes_dict, timestep=None):

    top_features_dict = {}
    sparse_maps_dict = {}

    for code in code_to_block.keys():
        block = code_to_block[code]
        sae = saes_dict[code]

        diff = cache["output"][block] - cache["input"][block]
        if diff.shape[0] == 2: # guidance is on and we need to select the second output
            diff = diff[1].unsqueeze(0)

        # If a specific timestep is provided, select that timestep from the cached activations
        if timestep is not None and timestep < diff.shape[1]:
            diff = diff[:, timestep:timestep+1]
        
        diff = diff.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0)
        with torch.no_grad():
            sparse_maps = sae.encode(diff)
        averages = torch.mean(sparse_maps, dim=(0, 1))

        top_features = torch.topk(averages, 10).indices

        top_features_dict[code] = top_features.cpu().tolist()
        sparse_maps_dict[code] = sparse_maps.cpu().numpy()

    return top_features_dict, sparse_maps_dict


def plot_image_heatmap(cache, block_select, radio):
    code = block_select.split()[0]
    feature = int(radio)
    block = code_to_block[code]
    
    heatmap = cache["heatmaps"][code][:, :, feature]
    heatmap = np.kron(heatmap, np.ones((32, 32)))
    image = cache["image"].convert("RGBA")
    
    jet = plt.cm.jet
    cmap = jet(np.arange(jet.N))
    cmap[:1, -1] = 0
    cmap[1:, -1] = 0.6
    cmap = ListedColormap(cmap)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap_rgba = cmap(heatmap)
    heatmap_image = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    heatmap_with_transparency = Image.alpha_composite(image, heatmap_image)

    return heatmap_with_transparency


def create_prompt_part(pipe, saes_dict, demo):
    def image_gen(prompt, timestep=None, num_steps=None, guidance_scale=None):
        lock.acquire()
        try:
            # Default values
            is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
            default_n_steps = 25 if is_base_model else 1
            default_guidance = base_guidance_scale_default if is_base_model else turbo_guidance_scale_default
            
            # Use provided values if available, otherwise use defaults
            n_steps = default_n_steps if num_steps is None else int(num_steps)
            guidance = default_guidance if guidance_scale is None else float(guidance_scale)
            
            # Convert timestep to integer if it's not None
            timestep_int = None if timestep is None else int(timestep)
            
            images, cache = pipe.run_with_cache(
                prompt,
                positions_to_cache=list(code_to_block.values()),
                num_inference_steps=n_steps,
                generator=torch.Generator(device="cpu").manual_seed(42),
                guidance_scale=guidance,
                save_input=True,
                save_output=True
            )
        finally:
            lock.release()
        
        top_features_dict, top_sparse_maps_dict = process_cache(cache, saes_dict, timestep_int)
        return images.images[0], {
            "image": images.images[0],
            "heatmaps": top_sparse_maps_dict,
            "features": top_features_dict
        }

    def update_radio(cache, block_select):
        code = block_select.split()[0]
        return gr.update(choices=cache["features"][code])

    def update_img(cache, block_select, radio):
        new_img = plot_image_heatmap(cache, block_select, radio)
        return new_img
        
    def update_visibility():
        is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        return gr.update(visible=is_base_model), gr.update(visible=is_base_model)

    with gr.Tab("Explore", elem_classes="tabs") as explore_tab:
        cache = gr.State(value={
            "image": None,
            "heatmaps": None,
            "features": []
        })
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Row(equal_height=True):
                    prompt_field = gr.Textbox(lines=1, label="Enter prompt here", value="A cinematic shot of a professor sloth wearing a tuxedo at a BBQ party and eathing a dish with peas.")
                    button = gr.Button("Generate", elem_classes="generate_button1")

                with gr.Row():
                    image = gr.Image(width=512, height=512, image_mode="RGB", label="Generated image")
            
            with gr.Column(scale=4):
                block_select = gr.Dropdown(
                    choices=["up.0.1 (style)", "down.2.1 (composition)", "up.0.0 (details)", "mid.0"], 
                    value="down.2.1 (composition)",
                    label="Select block", 
                    elem_id="block_select",
                    interactive=True
                )
                
                # Add SDXL base specific controls
                is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
                
                with gr.Group() as sdxl_base_controls:
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=25 if is_base_model else 1,
                        step=1,
                        label="Number of steps",
                        elem_id="steps_slider",
                        interactive=True,
                        visible=is_base_model
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=15.0,
                        value=base_guidance_scale_default if is_base_model else turbo_guidance_scale_default,
                        step=0.1,
                        label="Guidance scale",
                        elem_id="guidance_slider",
                        interactive=True,
                        visible=is_base_model
                    )
                
                    # Add timestep selector
                    n_steps = 25 if is_base_model else 1
                    timestep_selector = gr.Slider(
                        minimum=0,
                        maximum=n_steps-1,
                        value=None,
                        step=1,
                        label="Timestep (leave empty for average across all steps)",
                        elem_id="timestep_selector",
                        interactive=True,
                        visible=is_base_model
                    )
                    recompute_button = gr.Button("Recompute", elem_id="recompute_button",
                                                 visible=is_base_model)
                
                # Update max timestep when steps change
                steps_slider.change(lambda s: gr.update(maximum=s-1), [steps_slider], [timestep_selector])
                
                radio = gr.Radio(choices=[], label="Select a feature", interactive=True)
        
        button.click(image_gen, [prompt_field, timestep_selector, steps_slider, guidance_slider], outputs=[image, cache])
        cache.change(update_radio, [cache, block_select], outputs=[radio])
        block_select.select(update_radio, [cache, block_select], outputs=[radio])
        radio.select(update_img, [cache, block_select, radio], outputs=[image])
        recompute_button.click(image_gen, [prompt_field, timestep_selector, steps_slider, guidance_slider], outputs=[image, cache])
        demo.load(image_gen, [prompt_field, timestep_selector, steps_slider, guidance_slider], outputs=[image, cache])

    return explore_tab

def downsample_mask(image, factor):
    downsampled = image.reshape(
        (image.shape[0] // factor, factor,
        image.shape[1] // factor, factor)
    )
    downsampled = downsampled.mean(axis=(1, 3))
    return downsampled

def create_intervene_part(pipe: HookedStableDiffusionXLPipeline, saes_dict, means_dict, demo):
    def image_gen(prompt, num_steps, guidance_scale=None):
        lock.acquire()
        is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        default_guidance = base_guidance_scale_default if is_base_model else turbo_guidance_scale_default
        guidance = default_guidance if guidance_scale is None else float(guidance_scale)
        try:
            images = pipe.run_with_hooks(
                prompt,
                position_hook_dict={},
                num_inference_steps=int(num_steps),
                generator=torch.Generator(device="cpu").manual_seed(42),
                guidance_scale=guidance,
            )
        finally:
            lock.release()
        if images.images[0].size == (1024, 1024):
            return images.images[0].resize((512, 512))
        else:
            return images.images[0]

    def image_mod(prompt, block_str, brush_index, strength, num_steps, input_image, guidance_scale=None, start_index=None, end_index=None):
        block = block_str.split(" ")[0]
        is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        mask = (input_image["layers"][0] > 0)[:, :, -1].astype(float)
        if is_base_model:
            mask = downsample_mask(mask, 16)
        else:
            mask = downsample_mask(mask, 32)
        mask = torch.tensor(mask, dtype=torch.float32, device="cuda")

        if mask.sum() == 0:
            gr.Info("No mask selected, please draw on the input image")
            
        if is_base_model:
            # Set default values for start_index and end_index if not provided
            if start_index is None:
                start_index = 0
            if end_index is None:
                end_index = int(num_steps)
                
            # Ensure start_index and end_index are within valid ranges
            start_index = max(0, min(int(start_index), int(num_steps)))
            end_index = max(0, min(int(end_index), int(num_steps)))
            
            # Ensure start_index is less than end_index
            if start_index >= end_index:
                start_index = max(0, end_index - 1)
            def myhook(module, input, output):
                return add_feature_on_area_base(
                    saes_dict[block],
                    brush_index,
                    mask * means_dict[block][brush_index] * strength,
                    module,
                    input, 
                    output)
            hook = TimedHook(myhook, int(num_steps), np.arange(start_index, end_index))
        else:
            def hook(module, input, output):
                return add_feature_on_area_turbo(
                    saes_dict[block],
                    brush_index,
                    mask * means_dict[block][brush_index] * strength,
                    module,
                    input, 
                    output)
        lock.acquire()
        is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        default_guidance = base_guidance_scale_default if is_base_model else turbo_guidance_scale_default
        guidance = default_guidance if guidance_scale is None else float(guidance_scale)
        
        try:
            image = pipe.run_with_hooks(
                prompt,
                position_hook_dict={code_to_block[block]: hook},
                num_inference_steps=int(num_steps),
                generator=torch.Generator(device="cpu").manual_seed(42),
                guidance_scale=guidance
            ).images[0]
        finally:
            lock.release()
        return image

    def feature_icon(block_str, brush_index, guidance_scale=None):
        block = block_str.split(" ")[0]
        if block in ["mid.0", "up.0.0"]:
            gr.Info("Note that Feature Icon works best with down.2.1 and up.0.1 blocks but feel free to explore", duration=3)

        def hook(module, input, output):
            if is_base_model:
                return replace_with_feature_base(
                    saes_dict[block],
                    brush_index,
                    means_dict[block][brush_index] * saes_dict[block].k,
                    module,
                    input,
                    output
                )
            else:
                return replace_with_feature_turbo(
                    saes_dict[block],
                    brush_index,
                    means_dict[block][brush_index] * saes_dict[block].k,
                    module,
                    input,
                    output)
        lock.acquire()
        is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        n_steps = 25 if is_base_model else 1
        default_guidance = base_guidance_scale_default if is_base_model else turbo_guidance_scale_default
        guidance = default_guidance if guidance_scale is None else float(guidance_scale)
        
        try:
            image = pipe.run_with_hooks(
                "",
                position_hook_dict={code_to_block[block]: hook},
                num_inference_steps=n_steps,
                generator=torch.Generator(device="cpu").manual_seed(42),
                guidance_scale=guidance,
            ).images[0]
        finally:
            lock.release()
        return image

    with gr.Tab("Paint!", elem_classes="tabs") as intervene_tab:
        image_state = gr.State(value=None)
        with gr.Row():
            with gr.Column(scale=3):
                # Generation column
                with gr.Row():
                    # prompt and num_steps
                    is_base_model = pipe.pipe.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
                    n_steps = 25 if is_base_model else 1
                    prompt_field = gr.Textbox(lines=1, label="Enter prompt here", value="A dog plays with a ball, cartoon", elem_id="prompt_input")                    
                    
                with gr.Row():
                    num_steps = gr.Number(value=n_steps, label="Number of steps", minimum=1, maximum=50, elem_id="num_steps", precision=0)
                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=15.0,
                        value=base_guidance_scale_default if is_base_model else turbo_guidance_scale_default,
                        step=0.1,
                        label="Guidance scale",
                        elem_id="paint_guidance_slider",
                        interactive=True,
                        visible=is_base_model
                    )
                    
                with gr.Row():
                    # Generate button
                    button_generate = gr.Button("Generate", elem_id="generate_button")
            with gr.Column(scale=3):
                # Intervention column
                with gr.Row():
                    # dropdowns and number inputs
                    with gr.Column(scale=7):
                        with gr.Row():
                            block_select = gr.Dropdown(
                                choices=["up.0.1 (style)", "down.2.1 (composition)", "up.0.0 (details)", "mid.0"], 
                                value="down.2.1 (composition)",
                                label="Select block", 
                                elem_id="block_select"
                            )
                            brush_index = gr.Number(value=0, label="Brush index", minimum=0, maximum=5119, elem_id="brush_index", precision=0)
                        with gr.Row():
                            button_icon = gr.Button('Feature Icon', elem_id="feature_icon_button")
                        with gr.Row():
                            gr.Markdown("**TimedHook Range** (which steps to apply the feature)", visible=is_base_model)
                        with gr.Row():
                            start_index = gr.Number(value=0, label="Start index", minimum=0, maximum=n_steps, elem_id="start_index", precision=0, visible=is_base_model)
                            end_index = gr.Number(value=n_steps, label="End index", minimum=0, maximum=n_steps, elem_id="end_index", precision=0, visible=is_base_model)
                    with gr.Column(scale=3):
                        with gr.Row():
                            strength = gr.Number(value=10, label="Strength", minimum=-40, maximum=40, elem_id="strength", precision=2)
                        with gr.Row():
                            button = gr.Button('Apply', elem_id="apply_button")

        with gr.Row():
            with gr.Column():
                # Input image
                i_image = gr.Sketchpad(
                    height=610,
                    layers=False, transforms=[], placeholder="Generate and paint!",
                    brush=gr.Brush(default_size=64, color_mode="fixed", colors=['black']),
                    container=False,
                    canvas_size=(512, 512),
                    label="Input Image")
                clear_button = gr.Button("Clear")
                clear_button.click(lambda x: x, [image_state], [i_image])
            # Output image
            o_image = gr.Image(width=512, height=512, label="Output Image")

        # Set up the click events
        button_generate.click(image_gen, inputs=[prompt_field, num_steps, guidance_slider], outputs=[image_state])
        image_state.change(lambda x: x, [image_state], [i_image])
        
        if is_base_model:
            # Update max values for start_index and end_index when num_steps changes
            def update_index_maxes(steps):
                return gr.update(maximum=steps), gr.update(maximum=steps)
        
            num_steps.change(update_index_maxes, [num_steps], [start_index, end_index])
        
        button.click(image_mod, 
                    inputs=[prompt_field, block_select, brush_index, strength, num_steps, i_image, guidance_slider, start_index, end_index], 
                    outputs=o_image)
        button_icon.click(feature_icon, inputs=[block_select, brush_index, guidance_slider], outputs=o_image)
        demo.load(image_gen, [prompt_field, num_steps, guidance_slider], outputs=[image_state])


    return intervene_tab


def create_top_images_plus_search_part(retriever, demo):
    def update_cache(block_select, search_by_text, search_by_index):
        if search_by_text == "":
            top_indices = []
            index = search_by_index
            block = block_select.split(" ")[0]
            url = f"https://huggingface.co/surokpro2/sdxl_sae_images/resolve/main/{block}/{index}.jpg"
            return url, {"image": url, "feature_idx": index, "features": top_indices}
        else:
            if retriever is None:
                raise ValueError("Feature retrieval is not enabled")
            lock.acquire()
            try: 
                top_indices = list(retriever.query_text(search_by_text, block_select.split(" ")[0]).keys())
            finally:
                lock.release()
            block = block_select.split(" ")[0]
            top_indices = list(map(int, top_indices))
            index = top_indices[0]
            url = f"https://huggingface.co/surokpro2/sdxl_sae_images/resolve/main/{block}/{index}.jpg"
            return url, {"image": url, "feature_idx": index, "features": top_indices[:20]}

    def update_radio(cache):
        return gr.update(choices=cache["features"], value=cache["feature_idx"])

    def update_img(cache, block_select, index):
        block = block_select.split(" ")[0]
        url = f"https://huggingface.co/surokpro2/sdxl_sae_images/resolve/main/{block}/{index}.jpg"
        return url

    with gr.Tab("Top Images", elem_classes="tabs") as explore_tab:
        cache = gr.State(value={
            "image": None,
            "feature_idx": None,
            "features": []
        })
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Row():
                    # top images
                    image = gr.Image(width=600, height=600, image_mode="RGB", label="Top images")
            
            with gr.Column(scale=4):
                block_select = gr.Dropdown(
                    choices=["up.0.1 (style)", "down.2.1 (composition)", "up.0.0 (details)", "mid.0"], 
                    value="down.2.1 (composition)",
                    label="Select block", 
                    elem_id="block_select",
                    interactive=True
                )
                search_by_index = gr.Number(value=0, label="Search by index", minimum=0, maximum=5119, precision=0)
                search_by_text = gr.Textbox(lines=1, label="Search by text", value="")
                radio = gr.Radio(choices=[], label="Select a feature", interactive=True)
        

        search_by_text.change(update_cache, 
                        [block_select, search_by_text, search_by_index], 
                        outputs=[image, cache])
        block_select.select(update_cache,
                        [block_select, search_by_text, search_by_index],  
                        outputs=[image, cache])
        cache.change(update_radio, [cache], outputs=[radio])
        radio.select(update_img, [cache, block_select, radio], outputs=[image])
        search_by_index.change(update_img, [cache, block_select, search_by_index], outputs=[image])
        demo.load(update_img, 
                  [cache, block_select, search_by_index], 
                  outputs=[image])

    return explore_tab


def create_intro_part():
    with gr.Tab("Instructions", elem_classes="tabs") as intro_tab:
        gr.Markdown(
            '''# Unpacking SDXL Turbo with Sparse Autoencoders
            ## Demo Overview
            This demo showcases the use of Sparse Autoencoders (SAEs) to understand the features learned by the Stable Diffusion XL Turbo model. 
            
            ## How to Use
            ### Explore
            * Enter a prompt in the text box and click on the "Generate" button to generate an image.
            * You can observe the active features in different blocks plot on top of the generated image.
            ### Top Images
            * For each feature, you can view the top images that activate the feature the most.
            ### Paint!
            * Generate an image using the prompt.
            * Paint on the generated image to apply interventions.
            * Use the "Feature Icon" button to understand how the selected brush functions.

            ### Remarks
            * Not all brushes mix well with all images. Experiment with different brushes and strengths.
            * Feature Icon works best with `down.2.1 (composition)` and `up.0.1 (style)` blocks.
            * This demo is provided for research purposes only. We do not take responsibility for the content generated by the demo.

            ### Interesting features to try
            To get started, try the following features:
            - down.2.1 (composition): 2301 (evil) 3747 (image frame) 4998 (cartoon)
            - up.0.1 (style): 4977 (tiger stripes) 90 (fur) 2615 (twilight blur)
            '''
        )
    
    return intro_tab


def create_demo(pipe, saes_dict, means_dict, use_retrieval=True):
    custom_css = """
    .tabs button {
        font-size: 20px !important; /* Adjust font size for tab text */
        padding: 10px !important;   /* Adjust padding to make the tabs bigger */
        font-weight: bold !important; /* Adjust font weight to make the text bold */
    }
    .generate_button1 {
        max-width: 160px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    """
    if use_retrieval:
        retriever = FeatureRetriever()
    else:
        retriever = None

    with gr.Blocks(css=custom_css) as demo:
        with create_intro_part():
            pass
        with create_prompt_part(pipe, saes_dict, demo):
            pass
        with create_top_images_plus_search_part(retriever, demo):
            pass
        with create_intervene_part(pipe, saes_dict, means_dict, demo):
            pass
        
    return demo
