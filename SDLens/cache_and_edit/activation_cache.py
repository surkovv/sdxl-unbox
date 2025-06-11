from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from SDLens.cache_and_edit.hooks import fix_inf_values_hook, register_general_hook
import torch

class ModelActivationCache(ABC):
    """
    Cache for inference pass of a Diffusion Transformer.
    Used to cache residual-streams and activations.
    """
    def __init__(self):
    
        # Initialize caches for "double transformer" blocks using the subclass-defined NUM_TRANSFORMER_BLOCKS
        if hasattr(self, 'NUM_TRANSFORMER_BLOCKS'):
            self.image_residual = []
            self.image_activation = []
            self.text_residual = []
            self.text_activation = []

        # Initialize caches for "single transformer" blocks if defined (using NUM_SINGLE_TRANSFORMER_BLOCKS)
        if hasattr(self, 'NUM_SINGLE_TRANSFORMER_BLOCKS'):
            self.text_image_residual = []
            self.text_image_activation = []

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for attr_name, value in self.__dict__.items():
            if isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value):
                shapes = value[0].shape
                lines.append(f"  {attr_name}: len={len(value)}, shapes={shapes}")
            else:
                lines.append(f"  {attr_name}: {type(value)}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @abstractmethod
    def get_cache_info(self):
        """
        Return details about the cache configuration.
        Subclasses must implement this to provide info on their transformer block counts.
        """
        pass


class FluxActivationCache(ModelActivationCache):
    # Define number of blocks for double and single transformer caches
    NUM_TRANSFORMER_BLOCKS = 19
    NUM_SINGLE_TRANSFORMER_BLOCKS = 38

    def __init__(self):
        super().__init__()

    def get_cache_info(self):
        return {
            "transformer_blocks": self.NUM_TRANSFORMER_BLOCKS,
            "single_transformer_blocks": self.NUM_SINGLE_TRANSFORMER_BLOCKS,
        }
    
    def __getitem__(self, key):
        return getattr(self, key)


class PixartActivationCache(ModelActivationCache):
    # Define number of blocks for the double transformer cache only
    NUM_TRANSFORMER_BLOCKS = 28

    def __init__(self):
        super().__init__()

    def get_cache_info(self):
        return {
            "double_transformer_blocks": self.NUM_TRANSFORMER_BLOCKS,
        }


class ActivationCacheHandler:
    """ Used to manage ModelActivationCache of a Diffusion Transformer.
    """

    def __init__(self, cache: ModelActivationCache, positions_to_cache: List[str] = None):
        """Constructor.

        Args:
            cache (ModelActivationCache): cache to be used to store tensors.
            positions_to_cache (List[str], optional): name of modules to cached. 
                If None, all modules as specified in `cache.get_cache_info()` will be cached. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        self.cache = cache
        self.positions_to_cache = positions_to_cache

    @torch.no_grad()
    def cache_residual_and_activation_hook(self, *args):
        """ 
            To be used as a forward hook on a Transformer Block.
            It caches both residual_stream and activation (defined as output - residual_stream).
        """

        if len(args) == 3:
            module, input, output = args
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            encoder_hidden_states = output[0]            
            hidden_states = output[1]

            self.cache.image_activation.append(hidden_states - kwinput["hidden_states"])
            self.cache.text_activation.append(encoder_hidden_states - kwinput["encoder_hidden_states"])
            self.cache.image_residual.append(kwinput["hidden_states"])
            self.cache.text_residual.append(kwinput["encoder_hidden_states"])

        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache.text_image_activation.append(output - kwinput["hidden_states"])
            self.cache.text_image_residual.append(kwinput["hidden_states"])
        else:
            raise NotImplementedError(f"Caching not implemented for {type(module)}")


    @property
    def forward_hooks_dict(self):
        
        # insert cache storing in dict
        hooks = defaultdict(list)

        if self.positions_to_cache is None:
            for block_type, num_layers in self.cache.get_cache_info().items():
                for i in range(num_layers):
                    module_name: str = f"transformer.{block_type}.{i}"
                    hooks[module_name].append(fix_inf_values_hook)
                    hooks[module_name].append(self.cache_residual_and_activation_hook)
        else:
            for module_name in self.positions_to_cache:
                hooks[module_name].append(fix_inf_values_hook)
                hooks[module_name].append(self.cache_residual_and_activation_hook)

        return hooks
        