from collections import defaultdict
from functools import partial
import gc
from typing import Callable, Dict, List, Literal, Union, Optional, Type, Union
import torch
from SDLens.cache_and_edit.activation_cache import FluxActivationCache, ModelActivationCache, PixartActivationCache, ActivationCacheHandler
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from SDLens.cache_and_edit.hooks import locate_block, register_general_hook, fix_inf_values_hook, edit_streams_hook
from SDLens.cache_and_edit.qkv_cache import QKVCacheFluxHandler, QKVCache, CachedFluxAttnProcessor3_0
from SDLens.cache_and_edit.scheduler_inversion import FlowMatchEulerDiscreteSchedulerForInversion
from SDLens.cache_and_edit.flux_pipeline import EditedFluxPipeline

from diffusers.pipelines import FluxPipeline



class CachedPipeline:
    
    def __init__(self, pipe: EditedFluxPipeline, text_seq_length: int = 512):

        assert isinstance(pipe, EditedFluxPipeline) or isinstance(pipe, FluxPipeline), "Use EditedFluxPipeline class in `cache_and_edit/flux_pipeline.py`"
        self.pipe = pipe
        self.text_seq_length = text_seq_length

        # Cache handlers
        self.activation_cache_handler = None
        self.qkv_cache_handler = None
        # keeps references to all registered hooks
        self.registered_hooks = []


    def setup_cache(self, use_activation_cache = True, 
                    use_qkv_cache = False, 
                    positions_to_cache: List[str] = None,
                    positions_to_cache_foreground: List[str] = None,
                    qkv_to_inject: QKVCache = None,
                    inject_kv_mode: Literal["image", "text", "both"] = None,
                    q_mask=None,
                    processor_class: Optional[Type] = CachedFluxAttnProcessor3_0
                    ) -> None:
        """
            Sets up activation_cache and/or qkv_cache, setting the required hooks.
            If positions_to_cache is None, then all modules will be cached.
            If inject_kv_mode is None, then qkv cache will be stored, otherwise qkv_to_inject will be injected.
        """

        if use_activation_cache:
            if isinstance(self.pipe, EditedFluxPipeline) or isinstance(self.pipe, FluxPipeline):
                activation_cache = FluxActivationCache()
            else:
                raise AssertionError(f"activation cache not implemented for {type(self.pipe)}")

            self.activation_cache_handler = ActivationCacheHandler(activation_cache, positions_to_cache)
            # register hooks crated by activation_cache
            self._set_hooks(position_hook_dict=self.activation_cache_handler.forward_hooks_dict,
                            with_kwargs=True)
        
        if use_qkv_cache:
            if isinstance(self.pipe, EditedFluxPipeline) or isinstance(self.pipe, FluxPipeline):
                self.qkv_cache_handler = QKVCacheFluxHandler(self.pipe, 
                                                             positions_to_cache, 
                                                             positions_to_cache_foreground,
                                                             inject_kv=inject_kv_mode, 
                                                             text_seq_length=self.text_seq_length,
                                                             q_mask=q_mask,
                                                             processor_class=processor_class,
                                                             )
            else:
                raise AssertionError(f"QKV cache not implemented for {type(self.pipe)}")
            
            # qkv_cache does not use hooks
                

    @property
    def activation_cache(self) -> ModelActivationCache:
        return self.activation_cache_handler.cache if hasattr(self, "activation_cache_handler") and self.activation_cache_handler else None
    

    @property
    def qkv_cache(self) -> QKVCache:
        return self.qkv_cache_handler.cache if hasattr(self, "qkv_cache_handler") and self.qkv_cache_handler else None
    

    @torch.no_grad
    def run(self, 
            prompt: Union[str, List[str]], 
            num_inference_steps: int = 1,
            seed: int = 42,
            width=1024,
            height=1024,
            cache_activations: bool = False,
            cache_qkv: bool = False,
            guidance_scale: float = 0.0,
            positions_to_cache: List[str] = None,
            empty_clip_embeddings: bool = True,
            inverse: bool = False,
            **kwargs):
        """run the pipeline, possibly cachine activations or QKV.

        Args:
            prompt (str): Prompt to run the pipeline (NOTE: for Flux, parameters passed are prompt='' and prompt2=prompt)
            num_inference_steps (int, optional): Num steps for inference. Defaults to 1.
            seed (int, optional): seed for generators. Defaults to 42.
            cache_activations (bool, optional): Whether to cache activations. Defaults to True.
            cache_qkv (bool, optional): Whether to cache queries, keys, values. Defaults to False.
            positions_to_cache (List[str], optional): list of blocks to cache. 
                    If None, all transformer blocks will be cached. Defaults to None.

        Returns:
            _type_: same output as wrapped pipeline.
        """
        
        # First, clear all registered hooks 
        self.clear_all_hooks()

        # Delete cache already present
        if self.activation_cache or self.qkv_cache:

            if self.activation_cache:
                del(self.activation_cache_handler.cache)
                del(self.activation_cache_handler)

            if self.qkv_cache:
                # Necessary to delete the old cache. 
                self.qkv_cache_handler.clear_cache()
                del(self.qkv_cache_handler)

            gc.collect()  # force Python to clean up unreachable objects            
            torch.cuda.empty_cache()  # tell PyTorch to release unused GPU memory from its cache

        # Setup cache again for the current inference pass
        self.setup_cache(cache_activations, cache_qkv, positions_to_cache, inject_kv_mode=None)

        assert isinstance(seed, int)

        if isinstance(prompt, str):
            empty_prompt = [""]
            prompt = [prompt]
        else:
            empty_prompt = [""] * len(prompt)
        
        gen = [torch.Generator(device="cpu").manual_seed(seed) for _ in range(len(prompt))]

        if inverse:
            # maybe create scheduler for inversion
            if not hasattr(self, "inversion_scheduler"):
                self.inversion_scheduler = FlowMatchEulerDiscreteSchedulerForInversion.from_config(
                    self.pipe.scheduler.config, 
                    inverse=True
                )
                self.og_scheduler = self.pipe.scheduler
            
            self.pipe.scheduler = self.inversion_scheduler

        output = self.pipe(
                prompt=empty_prompt if empty_clip_embeddings else prompt,
                prompt_2=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                width=width,
                height=height,
                **kwargs
            )
        
        # Restore original scheduler
        if inverse: 
            self.pipe.scheduler = self.og_scheduler

        return output
    
    @torch.no_grad
    def run_inject_qkv(self, 
            prompt: Union[str, List[str]], 
            positions_to_inject: List[str] = None,
            positions_to_inject_foreground: List[str] = None,
            inject_kv_mode: Literal["image", "text", "both"] = "image",
            num_inference_steps: int = 1,
            guidance_scale: float = 0.0,
            seed: int = 42,
            empty_clip_embeddings: bool = True,
            q_mask=None,
            width: int = 1024,
            height: int = 1024,
            processor_class: Optional[Type] = CachedFluxAttnProcessor3_0,
            **kwargs):
        """run the pipeline, possibly cachine activations or QKV.

        Args:
            prompt (str): Prompt to run the pipeline (NOTE: for Flux, parameters passed are prompt='' and prompt2=prompt)
            num_inference_steps (int, optional): Num steps for inference. Defaults to 1.
            seed (int, optional): seed for generators. Defaults to 42.
            cache_activations (bool, optional): Whether to cache activations. Defaults to True.
            cache_qkv (bool, optional): Whether to cache queries, keys, values. Defaults to False.
            positions_to_cache (List[str], optional): list of blocks to cache. 
                    If None, all transformer blocks will be cached. Defaults to None.

        Returns:
            _type_: same output as wrapped pipeline.
        """
        
        # First, clear all registered hooks 
        self.clear_all_hooks()

        # Delete previous QKVCache
        if hasattr(self, "qkv_cache_handler") and self.qkv_cache_handler is not None:
            self.qkv_cache_handler.clear_cache()
            del(self.qkv_cache_handler)
            gc.collect()  # force Python to clean up unreachable objects            
            torch.cuda.empty_cache()  # tell PyTorch to release unused GPU memory from its cache

        # Will setup existing QKV cache to be injected
        self.setup_cache(use_activation_cache=False, 
                         use_qkv_cache=True, 
                         positions_to_cache=positions_to_inject,
                         positions_to_cache_foreground=positions_to_inject_foreground,
                         inject_kv_mode=inject_kv_mode,
                         q_mask=q_mask,
                         processor_class=processor_class,
                         )
        
        self.qkv_cache_handler

        assert isinstance(seed, int)

        if isinstance(prompt, str):
            empty_prompt = [""] 
            prompt = [prompt] 
        else:
            empty_prompt = [""] * len(prompt)
        
        gen = [torch.Generator(device="cpu").manual_seed(seed) for _ in range(len(prompt))]

        output = self.pipe(
                prompt=empty_prompt if empty_clip_embeddings else prompt,
                prompt_2=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                width=width,
                height=height,
                **kwargs
            )
        


        return output


    def clear_all_hooks(self):

        # 1. Clear all registered hooks
        for hook in self.registered_hooks:
                hook.remove()
        self.registered_hooks = []

        # 2. Eventually clear other hooks registered in the pipeline but not present here
        # TODO: make it general for other models
        for i in range(len(locate_block(self.pipe, "transformer.transformer_blocks"))):
            locate_block(self.pipe, f"transformer.transformer_blocks.{i}")._forward_hooks.clear()
            
        for i in range(len(locate_block(self.pipe, "transformer.single_transformer_blocks"))):
            locate_block(self.pipe, f"transformer.single_transformer_blocks.{i}")._forward_hooks.clear()


    def _set_hooks(self, 
                   position_hook_dict: Dict[str, List[Callable]] = {}, 
                   position_pre_hook_dict: Dict[str, List[Callable]] = {},
                   with_kwargs=False
    ):
        '''
        Set hooks at specified positions and register them.
        Args:
            position_hook_dict: A dictionary mapping positions to hooks.
                The keys are positions in the pipeline where the hooks should be registered.
                The values are either a single hook or a list of hooks to be registered at the specified position.
                Each hook should be a callable that takes three arguments: (module, input, output).
            **kwargs: Keyword arguments to pass to the pipeline.
        '''

        # Register hooks
        for is_pre_hook, hook_dict in [(True, position_pre_hook_dict), (False, position_hook_dict)]:
            for position, hook in hook_dict.items():
                assert isinstance(hook, list)
                for h in hook:
                    self.registered_hooks.append(register_general_hook(self.pipe, position, h, with_kwargs, is_pre_hook))
        

    def run_with_edit(self, 
                      prompt: str,
                      edit_fn: callable,
                      layers_for_edit_fn: List[int],
                      stream: Literal['text', 'image', 'both'],
                      guidance_scale: float = 0.0,
                      seed=42,
                      num_inference_steps=1,
                      empty_clip_embeddings: bool = True,
                      width: int = 1024,
                      height: int = 1024,
                      **kwargs,
                    ):

        assert isinstance(seed, int)

        self.clear_all_hooks()
    

        # Setup hooks for edit_fn at the specified layers
        # NOTE: edit_fn_hooks has to be Dict[str, List[Callable]]
        edit_fn_hooks = {f"transformer.transformer_blocks.{layer}": [lambda *args: edit_streams_hook(*args, recompute_fn=edit_fn, stream=stream)]
                            for layer in layers_for_edit_fn if layer < 19}
        edit_fn_hooks.update({f"transformer.single_transformer_blocks.{layer - 19}": [lambda *args: edit_streams_hook(*args, recompute_fn=edit_fn, stream=stream)]
                                for layer in layers_for_edit_fn if layer >= 19})

        
        # register hooks in the pipe
        self._set_hooks(position_hook_dict=edit_fn_hooks, with_kwargs=True)

        # Create generators

        if isinstance(prompt, str):
            empty_prompt = [""]
            prompt = [prompt]
        else:
            empty_prompt = [""] * len(prompt)

        gen = [torch.Generator(device="cpu").manual_seed(seed) for _ in range(len(prompt))]

        with torch.no_grad():
            output = self.pipe(
                prompt=empty_prompt if empty_clip_embeddings else prompt,
                prompt_2=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                width=width,
                height=height,
                **kwargs
            )
        
        return output
    