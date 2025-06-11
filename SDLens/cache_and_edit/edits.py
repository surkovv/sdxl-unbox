
class Edit:

    def __init__(self, ablator, vanilla_pre_forward_dict: Callable[[str, int], dict],
                                vanilla_forward_dict: Callable[[str, int], dict],
                                ablated_pre_forward_dict: Callable[[str, int], dict],
                                ablated_forward_dict: Callable[[str, int], dict],):
        self.ablator=ablator
        self.vanilla_seed = 42
        self.vanilla_pre_forward_dict = vanilla_pre_forward_dict
        self.vanilla_forward_dict = vanilla_forward_dict

        self.ablated_seed = 42
        self.ablated_pre_forward_dict = ablated_pre_forward_dict
        self.ablated_forward_dict = ablated_forward_dict

    
    def get_edit(name: str):
        
        if name == "edit_streams":
            ablator = TransformerActivationCache()
            stream: str = kwargs["stream"]
            layers = kwargs["layers"]
            edit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kwargs["edit_fn"]

            interventions = {f"transformer.transformer_blocks.{layer}": lambda *args: ablator.edit_streams(*args, recompute_fn=partial(edit_fn, layer=layer), stream=stream) for layer in layers if layer < 19}
            interventions.update({f"transformer.single_transformer_blocks.{layer - 19}": lambda *args: ablator.edit_streams(*args, recompute_fn=partial(edit_fn, layer=layer), stream=stream) for layer in layers if layer >= 19})

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num:  {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: interventions,
                        )
     
    
"""
    def get_ablation(name: str, **kwargs):

        if name == "intermediate_text_stream_to_input":

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.cache_attention_activation(*args, full_output=True)},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, stream="text")},
                            ablated_forward_dict=lambda block_type, layer_num: {})
        elif name == "input_to_intermediate_text_stream":
            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": lambda *args: ablator.cache_attention_activation(*args, full_output=True)},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.replace_stream_input(*args, stream="text")},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        elif name == "set_input_text":

            tensor: torch.Tensor = kwargs["tensor"]

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.replace_stream_input(*args, use_tensor=tensor, stream="text")},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.clamp_output(*args)})

        elif name == "replace_text_stream_activation":
            ablator = AttentionAblationCacheHook()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0


            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.set_ablated_attention(*args, weight=weight)})
        
        elif name == "replace_text_stream":
            ablator = TransformerActivationCache()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {})
 
        
        elif name == "input=output":
            return Ablation(None,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablate_block(*args)})
        
        elif name == "reweight_text_stream": 
            ablator = TransformerActivationCache()

            residual_w=kwargs["residual_w"]
            activation_w=kwargs["activation_w"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.reweight_text_stream(*args, residual_w=residual_w, activation_w=activation_w)})
        
        elif name == "add_input_text":

            tensor: torch.Tensor = kwargs["tensor"]

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.add_text_stream_input(*args, use_tensor=tensor)},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.clamp_output(*args)})

        elif name == "nothing":
            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {})
        
        elif name == "reweight_image_stream": 
            ablator = TransformerActivationCache()
            residual_w=kwargs["residual_w"]
            activation_w=kwargs["activation_w"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.reweight_image_stream(*args, residual_w=residual_w, activation_w=activation_w)})
        
        if name == "intermediate_image_stream_to_input":

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.cache_attention_activation(*args, full_output=True)},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, stream='image')},
                            ablated_forward_dict=lambda block_type, layer_num: {})


        elif name == "replace_text_stream_one_layer":
            ablator = AttentionAblationCacheHook()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0


            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.restore_text_stream})

        elif name == "replace_intermediate_representation":
            ablator = TransformerActivationCache()
            tensor: torch.Tensor = kwargs["tensor"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, use_tensor=tensor, stream='text_image')},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        elif name == "destroy_registers":
            ablator = TransformerActivationCache()
            layers: List[int] = kwargs['layers']
            k: float = kwargs["k"]
            stream: str = kwargs['stream']
            random: bool = kwargs["random"] if "random" in kwargs else False
            lowest_norm: bool = kwargs["lowest_norm"] if "lowest_norm" in kwargs else False

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.destroy_registers(*args,  k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            ablated_forward_dict=lambda block_type, layer_num: {})
        
        elif name == "patch_registers":
            ablator = TransformerActivationCache()
            layers: List[int] = kwargs['layers']
            k: float = kwargs["k"]
            stream: str = kwargs['stream']
            random: bool = kwargs["random"] if "random" in kwargs else False
            lowest_norm: bool = kwargs["lowest_norm"] if "lowest_norm" in kwargs else False

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num:  {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.destroy_registers(*args, k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.set_cached_registers(*args, k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        elif name == "add_registers":
            ablator = TransformerActivationCache()
            num_registers: int = kwargs["num_registers"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num:  {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer": lambda *args: insert_extra_registers(*args, num_registers=num_registers)},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer": lambda *args: discard_extra_registers(*args, num_registers=num_registers)},)
        

        elif name == "edit_streams":
            ablator = TransformerActivationCache()
            stream: str = kwargs["stream"]
            layers = kwargs["layers"]
            edit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kwargs["edit_fn"]

            interventions = {f"transformer.transformer_blocks.{layer}": lambda *args: ablator.edit_streams(*args, recompute_fn=partial(edit_fn, layer=layer), stream=stream) for layer in layers if layer < 19}
            interventions.update({f"transformer.single_transformer_blocks.{layer - 19}": lambda *args: ablator.edit_streams(*args, recompute_fn=partial(edit_fn, layer=layer), stream=stream) for layer in layers if layer >= 19})

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num:  {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: interventions,
                        )
"""