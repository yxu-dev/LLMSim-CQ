# interface.py
# common imports
from types import MethodType
from typing import Callable

import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module

# framefusion methods
from lm_eval.similarity.main import Similarity
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from lm_eval.similarity.models.llama3_model import (
    llama_simi_mlp_forward,
    llama_simi_attention_forward,
    llama_simi_decoderlayer_forward,
    llama_simi_model_forward,
)

from lm_eval.similarity.utils import IGNORE_TOKEN, TEXT_TOKEN, get_attr_by_name

# model types
from transformers import LlavaNextVideoForConditionalGeneration, PreTrainedModel, MllamaForConditionalGeneration


def apply_similarity(model, cli_args=None):
    """
    Apply FrameFusion to the model

    Args:
        model: the model to apply FrameFusion to
        cli_args: the command line arguments
    """
    # Llama3  Model
    if isinstance(model, LlamaForCausalLM):
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"
        mlp_key = "mlp"

        llm_forward = llama_simi_model_forward
        decoder_forward = llama_simi_decoderlayer_forward
        attention_forward = llama_simi_attention_forward
        mlp_forward = llama_simi_mlp_forward

    else:
        raise NotImplementedError


    # if cli_args.simi:
    #     for name, module in model.named_modules():
    #         if name.startswith(f"{llm_key}.{decoder_key}"):
    #             if isinstance(module, torch.nn.Linear):
    #                 weights = module.weight.data.detach()
    #                 L1_norm = torch.mean(torch.abs(weights), dim=0)
    #                 L1_norm = L1_norm / torch.max(L1_norm)
    #                 module.L1_norm = L1_norm

    replace_similarity_forward(
        model,
        llm_forward,
        decoder_forward,
        attention_forward,
        mlp_forward,
        llm_key=llm_key,
        decoder_key=decoder_key,
        attention_key=attention_key,
        mlp_key=mlp_key,
        cli_args=cli_args,
    )


def replace_similarity_forward(
    module: torch.nn.Module,
    llm_forward: Callable,
    decoder_forward: Callable,
    attention_forward: Callable,
    mlp_forward: Callable,
    llm_key: str = "model",
    decoder_key: str = "layers",
    decoder_keys: list = None,
    attention_key: str = "self_attn",
    mlp_key: str = "mlp",
    cli_args=None,
):
    """
    Replace the forward method of the model with the framefusion forward method.
    Make framefusion a property of the model.

    The keys are accessed in an hierarchical manner: llm_key -> decoder_key -> attention_key. Each key can have multiple hierarchies, e.g. "llm.model", which will be accessed by module.llm.model
    
    For models with multiple transformers (like Mllama), decoder_keys can be a list of decoder key paths to process multiple transformers in a single call.
    """
    # if cli_args.simi:
    #     if cli_args.threshold_list != "":
    #         threshold_list = [float(threshold) for threshold in cli_args.threshold_list.split(",")]
    #     else:
    #         threshold_list = None
    #     simi = Similarity(
    #         sparse_mode=cli_args.sparse_mode,
    #         spatial_temporal_mode=cli_args.spatial_temporal_mode,
    #         fast_mode=cli_args.fast,
    #         threshold=cli_args.threshold,
    #         zero_multiplier=cli_args.zero_multiplier,
    #         threshold_list=threshold_list,
    #         record_process=cli_args.record_process,
    #         use_L1_norm=cli_args.use_L1_norm,
    #         match_range=cli_args.match_range,
    #         alpha_list=cli_args.alpha_list,
    #         selected_layers=cli_args.selected_layers,
    #         export_simi=cli_args.export_simi,
    #         partition_size=cli_args.partition_size,
    #         mask_positions=cli_args.mask_positions,
    #         token_wise_only=cli_args.token_wise_only,
    #         simi_merge=cli_args.simi_merge,
    #         simi_merge_threshold=cli_args.simi_merge_threshold,
    #         tile_size=cli_args.tile_size,
    #         model_name=cli_args.model,
    #         dataset_name=cli_args.tasks,
    #     )
    # elif cli_args.CMC:
    #     cmc = CMC(interval_size=8, 
    #               threshold=cli_args.CMC_threshold, 
    #               threshold_query=cli_args.CMC_query_threshold, 
    #               threshold_score=cli_args.CMC_attn_threshold, 
    #               record_process=cli_args.record_process,
    #               simplified=cli_args.CMC_simple,
    #               )
    # elif cli_args.adaptiv:
    #     adaptiv = Adaptiv(threshold=cli_args.adaptiv_threshold)



    # if cli_args.simi:
    #     module.simi = simi
    # elif cli_args.CMC:
    #     module.cmc = cmc
    # elif cli_args.adaptiv:
    #     module.adaptiv = adaptiv

    llm = get_attr_by_name(module, llm_key)
    assert isinstance(llm, PreTrainedModel), f"{llm_key} is not a PreTrainedModel"

    # if cli_args.simi:
    #     llm.simi = simi
    # elif cli_args.CMC:
    #     llm.cmc = cmc
    # elif cli_args.adaptiv:
    #     llm.adaptiv = adaptiv

    llm.forward = MethodType(llm_forward, llm)

    # Handle multiple decoder keys if provided, otherwise use single decoder_key
    decoder_keys_to_process = decoder_keys if decoder_keys is not None else [decoder_key]
    
    for current_decoder_key in decoder_keys_to_process:
        decoder_layers = get_attr_by_name(llm, current_decoder_key)
        for i, decoder_layer in enumerate(decoder_layers):
            assert isinstance(decoder_layer, nn.Module), f"{current_decoder_key}[{i}] is not a nn.Module"

            # if cli_args.simi:
            #     decoder_layer.simi = simi
            # elif cli_args.CMC:
            #     decoder_layer.cmc = cmc
            # elif cli_args.adaptiv:
            #     decoder_layer.adaptiv = adaptiv

            decoder_layer.forward = MethodType(decoder_forward, decoder_layer)

            # ensure accelerate hooks are not removed
            if hasattr(decoder_layer, "_hf_hook"):
                decoder_layer._old_forward = MethodType(decoder_forward, decoder_layer)
                add_hook_to_module(decoder_layer, decoder_layer._hf_hook)

            attention = get_attr_by_name(decoder_layer, attention_key)
            assert isinstance(attention, nn.Module), f"{current_decoder_key}[{i}].self_attn is not a nn.Module"

            # replace the forward method of the attention layer
            # if cli_args.simi:
            #     attention.simi = simi
            # elif cli_args.CMC:
            #     attention.cmc = cmc
            # elif cli_args.adaptiv:
            #     attention.adaptiv = adaptiv

            attention.forward = MethodType(attention_forward, attention)

            # replace the forward method of the mlp layer
            mlp = get_attr_by_name(decoder_layer, mlp_key)
            assert isinstance(mlp, nn.Module), f"{current_decoder_key}[{i}].mlp is not a nn.Module"

            # if cli_args.simi:
            #     mlp.simi = simi
            # elif cli_args.CMC:
            #     mlp.cmc = cmc
            # elif cli_args.adaptiv:
            #     mlp.adaptiv = adaptiv

            mlp.forward = MethodType(mlp_forward, mlp)