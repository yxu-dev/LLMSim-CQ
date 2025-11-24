# main.py
from typing import List
import copy

import torch
from torch import nn
import numpy as np
import pandas as pd
import os

from lm_eval.similarity.utils import AverageMeter

import matplotlib.pyplot as plt

TEXT_TOKEN = -1
IGNORE_TOKEN = -2


class Similarity(nn.Module):
    def __init__(self, 
    sparse_mode="element_wise", 
    spatial_temporal_mode="spatiotemporal", 
    fast_mode=True, 
    threshold=0.1, 
    zero_multiplier=1.0, 
    threshold_list=None, 
    tile_size=32, 
    record_process=False, 
    use_L1_norm=False, 
    match_range=1, 
    alpha=0.7, 
    export_simi_sparse=False, 
    partition_size=None,
    alpha_list="",
    selected_layers="",
    mask_positions=None,
    token_wise_only=False,
    simi_merge=False,
    simi_merge_threshold=0.8,
    model_name="",
    dataset_name="",
    ):
        super(Similarity, self).__init__()
        self.base_threshold = threshold
        self.threshold = threshold
        self.threshold_S = 0.001
        self.threshold_list = threshold_list
        assert sparse_mode in ['element_wise', 'vector_wise']
        assert spatial_temporal_mode in ['temporal', 'spatiotemporal', 'adaptive']
        assert fast_mode in [True, False]
        self.sparse_mode = sparse_mode
        self.spatial_temporal_mode = spatial_temporal_mode
        self.fast_mode = fast_mode
        self.tile_size = tile_size
        self.vector_size = 8
        self.zero_multiplier = zero_multiplier
        self.record_process = record_process
        self.maximum_sparsity = 0.0
        self.maximum_sparsity_idx = -1
        self.cur_idx = 0
        self.sparsity_dict = {}
        self.use_L1_norm = use_L1_norm
        self.match_range = match_range
        self.selected_layer = [int(layer) for layer in selected_layers.split(",")] if selected_layers != "" else []
        self.alpha = [float(alpha) for alpha in alpha_list.split(",")] if alpha_list != "" else []
        
        self.training = False

        self.token_wise_only = token_wise_only
        self.simi_merge = simi_merge
        self.simi_merge_threshold = simi_merge_threshold

        self.sparsity_dict_cur = {}
        self.token_importance = None
        self.start_drop = False
        self.vit_mask = None

        self.partition_size = partition_size
        self.export_simi_sparse = export_simi_sparse
        self.limit = 1
        self.mask_positions = [int(position) for position in mask_positions.split(",")] if mask_positions != "" else []

        self.sparse_layer_by_layer = False
        self.similarity_analysis_mode = False

        # self.extract_attention_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        self.extract_attention_layer = []

        self.model_name = model_name
        self.dataset_name = dataset_name

    def prepare(self, patch_type, num_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length, query_token_start_index=None, query_token_length=None):
        self.patch_type = patch_type
        self.patch_num = patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_frames = num_frames

        self.frame_stride = frame_stride
        self.height_stride = height_stride
        self.width_stride = width_stride

        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.query_token_start_index = query_token_start_index if query_token_start_index is not None else image_token_end_index + 1
        self.query_token_length = query_token_length if query_token_length is not None else original_length - self.query_token_start_index
        self.original_length = original_length
        self.image_token_length_cur = image_token_length

        self.token_importance = None
        self.start_drop = False
        self.retained_ids = None

        self.attn_cnt = 0
        
        self.act_dict = {}

        self.cur_layer = 0
        if self.export_simi_sparse:
            if not hasattr(self, 'simi_sparse_info'):
                self.simi_sparse_info = []

            num_tile = (3584 + self.tile_size - 1) // self.tile_size
            num_tile_intermediate = (18944 + self.tile_size - 1) // self.tile_size
            
            mask_zero_dict = {"q_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "o_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "query": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "gate_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "down_proj": torch.empty(28, 1, self.image_token_length, num_tile_intermediate, dtype=torch.bool),}

            mask_similar_dict = {"q_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "o_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "query": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "gate_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.bool),
                   "down_proj": torch.empty(28, 1, self.image_token_length, num_tile_intermediate, dtype=torch.bool),}
            
            group_idx_dict = {"q_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.int32),
                   "o_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.int32),
                   "query": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.int32),
                   "gate_proj": torch.empty(28, 1, self.image_token_length, num_tile, dtype=torch.int32),
                   "down_proj": torch.empty(28, 1, self.image_token_length, num_tile_intermediate, dtype=torch.int32),}

            self.info_dict = {"mask_zero": mask_zero_dict,
                              "mask_similar": mask_similar_dict,
                              "group_idx": group_idx_dict}

        self.stop_merge = False


    def set_token_importance(self, attn_weights):
        b, num_head, q_len_0, q_len_1 = attn_weights.size()
        if q_len_0 == 1:
            return
        assert q_len_0 == q_len_1 
        assert b == 1

        # get text to image attentions
        image_start_index = self.image_token_start_index
        query_token_length = self.query_token_length

        text_to_image_attn = attn_weights[:, :, -query_token_length:, image_start_index:-query_token_length]

        # store text to image attn
        # torch.save(text_to_image_attn, 'output/simi/text_to_image_attn.pth')
        # raise NotImplementedError("text_to_image_attn is not used in the current version")

        text_to_image_attn = text_to_image_attn[0].max(dim=0)[0].max(dim=0)[0]
        # assert text_to_image_attn.shape == (self.image_token_length,)

        self.token_importance = text_to_image_attn
        # self.visualize_token_importance(self.token_importance)

        # get 50% quantile of the token importance
        # p50 = torch.quantile(self.token_importance.float(), 0.5, dim=-1, keepdim=True)
        # self.token_importance = self.token_importance / p50

    def visualize_token_importance(self, token_importance):
        median_value = torch.median(token_importance).item()
        max_value = torch.max(token_importance).item()
        min_value = torch.min(token_importance).item()
        # add median max min to the title
        plt.title(f'token_importance_{self.attn_cnt}_{median_value}_{max_value}_{min_value}')
        plt.hist(token_importance.cpu().numpy(), bins=100)
        plt.savefig(f'output/token_importance_{self.attn_cnt}.png')
        self.attn_cnt += 1
        # clear the figure
        plt.clf()

    def post_process(self):
            
        config_dict = {"q_proj": 3584 * 3584 + 3584 * 512 * 2,
                   "o_proj": 3584 * 3584,
                   "query": 3584 * 11648,
                   "gate_proj": 3584 * 18944 * 2,
                   "down_proj": 3584 * 18944,}
        
        # assert all keys in sparsity_cur_dict is in config_dict
        for key in self.sparsity_dict_cur.keys():
            assert key in config_dict, f"{key} not in config_dict"

        # assert len(self.sparsity_dict_cur) == len(config_dict), f"sparsity_dict_cur: {self.sparsity_dict_cur}, config_dict: {config_dict}"

        # get the weighted average sparsity according to config_dict
        weighted_avg_sparsity = 0
        sum_weight = 0
        for key, value in self.sparsity_dict_cur.items():
            weighted_avg_sparsity += value.avg * config_dict[key]
            sum_weight += config_dict[key]
        weighted_avg_sparsity /= sum_weight

        if weighted_avg_sparsity > self.maximum_sparsity:
            self.maximum_sparsity = weighted_avg_sparsity
            self.maximum_sparsity_idx = self.cur_idx

        self.cur_idx += 1
        self.sparsity_dict_cur = {}

        if not hasattr(self, 'weighted_avg_sparsity_list'):
            self.weighted_avg_sparsity_list = []
        self.weighted_avg_sparsity_list.append(weighted_avg_sparsity)

        if len(self.weighted_avg_sparsity_list) >= self.limit:
            # Convert list to tensor for easier computation
            sparsity_tensor = torch.tensor(self.weighted_avg_sparsity_list)
                
            # Calculate median value
            median_value = torch.median(sparsity_tensor).item()
                
            # Find the index of the median value (or closest to it if there are duplicates)
            median_idx = torch.argmin(torch.abs(sparsity_tensor - median_value)).item()
                
            # Store the results as instance variables
            self.median_sparsity = median_value
            self.median_sparsity_idx = median_idx
            self.weighted_avg_sparsity_list = []

        if self.export_simi_sparse:

            # append a deep copy of self.info_dict to self.simi_sparse_info and clear self.info_dict
            self.simi_sparse_info.append(copy.deepcopy(self.info_dict))
            self.info_dict = {}

            if len(self.simi_sparse_info) >= self.limit:

                # save the simi_sparse_info to a file
                median_info_dict = self.simi_sparse_info[median_idx]
                torch.save(median_info_dict, f'output/tile_size/tile_size_{self.tile_size}.pth')

                self.simi_sparse_info = []
            

        if self.record_process:
            torch.save(self.act_dict, 'output/simi/act_dict.pth')

    def forward(self, hidden_states: torch.Tensor, L1_norm=None, is_attention=False, name=None) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (bsz, seq_len, hidden_dim) containing the input embeddings.

        Returns:
            torch.Tensor: A tensor of shape (bsz, seq_len, seq_len) that is approximated through simisparse
        """
        if self.sparsity_dict.get(name) is None:
            self.sparsity_dict[name] = AverageMeter()

        if self.sparsity_dict_cur.get(name) is None:
            self.sparsity_dict_cur[name] = AverageMeter()

        if len(hidden_states.shape) == 3:
            assert is_attention == False
            bsz, q_len, dim = hidden_states.size()
            assert bsz == 1
        elif len(hidden_states.shape) == 4:
            assert is_attention == True
            bsz, num_head, q_len, dim_per_head = hidden_states.size()
            assert bsz == 1
        else:
            raise NotImplementedError
        if q_len == 1:
            return hidden_states
        
        hidden_states_out = hidden_states.clone()

        if self.start_drop:
            hidden_states_out = self.recover_tokens(hidden_states_out)


        if self.record_process and name is not None:
            self.act_dict[name+'_before'] = hidden_states.clone().detach().cpu()


        if is_attention:
            image_tokens = hidden_states_out[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :].clone()
        else:
            image_tokens = hidden_states_out[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :].clone()

        q_len_image = image_tokens.size(1) if len(image_tokens.shape) == 3 else image_tokens.size(2)

        if self.token_wise_only:
            # count number of all zero tokens
            if len(image_tokens.shape) == 3:
                num_zero_tokens = torch.sum(torch.all(image_tokens == 0.0, dim=-1))
                sparsity = num_zero_tokens / q_len_image
            elif len(image_tokens.shape) == 4:
                num_zero_tokens = torch.sum(torch.all(image_tokens == 0.0, dim=(1,-1)))
                sparsity = num_zero_tokens / q_len_image

            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            if sparsity >= 0.7:
                self.stop_merge = True

            return hidden_states

        if self.similarity_analysis_mode:
            if is_attention:
                image_tokens = image_tokens.squeeze(0)
            self.similarity_analysis(image_tokens, name, self.cur_layer)

            if name == "down_proj":
                self.cur_layer += 1

            return hidden_states

        if self.vit_mask is not None:
            assert self.vit_mask.shape == (self.patch_width * self.patch_height, self.num_frames)
            if self.frame_stride == 1:
                mask = self.vit_mask.view(1, self.patch_width * self.patch_height * self.num_frames)
            elif self.width_stride == 1:
                mask = self.vit_mask.permute(1, 0).contiguous().view(1, self.patch_width * self.patch_height * self.num_frames)
            else:
                raise NotImplementedError("Only HWT or THW layout is supported for vit_mask")
            mask = mask.unsqueeze(-1)
            # extend mask to the image_tokens shape
            mask = mask.expand(-1, -1, image_tokens.shape[-1])
            assert mask.shape == image_tokens.shape, f"mask shape {mask.shape} does not match image_tokens shape {image_tokens.shape}"
            image_tokens = image_tokens * mask
                

        if self.sparse_mode == 'element_wise' and self.spatial_temporal_mode == 'temporal':
            if is_attention:
                image_tokens = image_tokens.transpose(1, 2).contiguous().view(bsz, q_len, num_head * dim_per_head)
            image_tokens = self.element_wise_simisparse_temporal(image_tokens, L1_norm)
            if is_attention:
                image_tokens = image_tokens.view(bsz, q_len, num_head, dim_per_head).transpose(1, 2).contiguous().view(bsz, num_head, q_len, dim_per_head)

        elif self.sparse_mode == 'vector_wise' and self.spatial_temporal_mode == 'temporal':
            if is_attention:
                image_tokens = image_tokens.squeeze(0)

            image_tokens, sparsity = self.vector_wise_simisparse_temporal(image_tokens, L1_norm)
            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            if is_attention:
                image_tokens = image_tokens.unsqueeze(0)

        elif self.sparse_mode == 'element_wise' and self.spatial_temporal_mode == 'spatiotemporal' and self.fast_mode:
            if is_attention:
                image_tokens = image_tokens.squeeze(0)
                
            image_tokens, sparsity = self.element_wise_simisparse_spatiotemporal_fast(image_tokens, L1_norm)
            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            # print(f"{name} sparsity: {sparsity}")

            if is_attention:
                image_tokens = image_tokens.unsqueeze(0)
        elif self.sparse_mode == 'element_wise' and self.spatial_temporal_mode == 'spatiotemporal':
            if is_attention:
                image_tokens = image_tokens.transpose(1, 2).contiguous().view(bsz, q_len, num_head * dim_per_head)
            image_tokens = self.element_wise_simisparse_spatiotemporal(image_tokens, L1_norm)
            if is_attention:
                image_tokens = image_tokens.view(bsz, q_len, num_head, dim_per_head).transpose(1, 2).contiguous().view(bsz, num_head, q_len, dim_per_head)
        elif self.sparse_mode == 'vector_wise' and self.spatial_temporal_mode == 'spatiotemporal' and self.fast_mode:
            if is_attention:
                image_tokens = image_tokens.squeeze(0)

            image_tokens, sparsity = self.vector_wise_simisparse_spatiotemporal_fast(image_tokens, L1_norm)
            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            # print(f"{name} sparsity: {sparsity}")
            if is_attention:
                image_tokens = image_tokens.unsqueeze(0)
        elif self.sparse_mode == 'vector_wise' and self.spatial_temporal_mode == 'spatiotemporal':
            image_tokens = self.vector_wise_simisparse_spatiotemporal(image_tokens, L1_norm)
        elif self.sparse_mode == 'vector_wise' and self.spatial_temporal_mode == 'adaptive':
            if is_attention:
                image_tokens = image_tokens.transpose(1, 2).contiguous().view(bsz, -1, num_head * dim_per_head)

            # Handle padding for tile_size compatibility specifically for adaptive_range
            original_image_tokens_shape = image_tokens.shape
            padding_size = 0
            
            # Get the hidden dimension (last dimension)
            hidden_dim = image_tokens.shape[-1]
            
            # Check if hidden_dim is divisible by tile_size
            if hidden_dim % self.tile_size != 0 and hidden_dim > self.tile_size:
                # Pad the image_tokens to make hidden_dim divisible by tile_size
                image_tokens, padding_size = self._pad_hidden_dim(image_tokens, self.tile_size)

            image_tokens, sparsity = self.vector_wise_simisparse_adaptive_range(image_tokens, name, L1_norm)
            
            # Remove padding if it was added
            if padding_size > 0:
                image_tokens = self._unpad_hidden_dim(image_tokens, padding_size)
            
            # print(f"name: {name}, sparsity: {sparsity}")
            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            if self.sparse_layer_by_layer:
                # save the sparsity to a csv file by appending a new row
                # the name should contain tile_size
                with open(f'output/sparsity_{self.tile_size}.csv', 'a') as f:
                    f.write(f"{name},{sparsity}\n")

            # print(f"{name} sparsity: {sparsity}")
            if is_attention:
                image_tokens = image_tokens.view(bsz, -1, num_head, dim_per_head).transpose(1, 2).contiguous()
        else:
            raise NotImplementedError

        if is_attention:
            hidden_states_out[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens
        else:
            hidden_states_out[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens

        if self.start_drop:
            hidden_states_out = self.drop_tokens(hidden_states_out)

        if self.record_process and name is not None:
            self.act_dict[name+'_after'] = hidden_states_out.clone().detach().cpu()

        if name == "down_proj":
            self.cur_layer += 1

        return hidden_states_out


    def element_wise_simisparse_temporal(self, image_tokens: torch.Tensor, L1_norm=None) -> torch.Tensor:
        num_elements = image_tokens.numel()
        num_similar = 0

        num_frames = self.image_token_length // self.patch_num
        image_tokens = image_tokens.view(num_frames, self.patch_num, image_tokens.shape[-1])
        for i in range(num_frames - 1):
            cur_frame = image_tokens[i]
            next_frame = image_tokens[i + 1]
            diff = cur_frame - next_frame
            if L1_norm is not None:
                mask = torch.abs(diff * L1_norm) < self.threshold_simi
            else:
                mask = torch.abs(diff) < self.threshold_simi
            image_tokens[i + 1] = torch.where(mask, cur_frame, next_frame)
            num_similar += torch.sum(mask)

        self.sparsity_list.append(num_similar.item() / num_elements)
        image_tokens = image_tokens.view(-1, image_tokens.shape[-1])
        

        return image_tokens
    
    def element_wise_simisparse_spatiotemporal(self, image_tokens: torch.Tensor, L1_norm=None) -> torch.Tensor:
        num_frames = self.image_token_length // self.patch_num
        image_tokens = image_tokens.view(num_frames, self.patch_height, self.patch_width, image_tokens.shape[-1])

        num_elements = image_tokens.numel()
        num_similar = 0

        for frame_id in range(num_frames):
            for i in range(self.patch_height):
                for j in range(self.patch_width):
                    num_similar_array = [0, -1, -1, -1] # blank, frame, left, up
                    cur_patch = image_tokens[frame_id, i, j]
                    if frame_id != 0:
                        prev_patch_frame = image_tokens[frame_id - 1, i, j]
                        diff_frame = torch.abs(prev_patch_frame - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_frame - cur_patch)
                        mask_frame = diff_frame < self.threshold_simi
                        num_similar_array[1] = torch.sum(mask_frame)
                    if i != 0:
                        prev_patch_left = image_tokens[frame_id, i - 1, j]
                        diff_left = torch.abs(prev_patch_left - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_left - cur_patch)
                        mask_left = diff_left < self.threshold_simi
                        num_similar_array[2] = torch.sum(mask_left)
                    if j != 0:
                        prev_patch_up = image_tokens[frame_id, i, j - 1]
                        diff_up = torch.abs(prev_patch_up - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_up - cur_patch)
                        mask_up = diff_up < self.threshold_simi
                        num_similar_array[3] = torch.sum(mask_up)

                    
                    # get the maximum from num_similar
                    max_index = torch.argmax(torch.tensor(num_similar_array)).item()
                    num_similar += num_similar_array[max_index]
                    if max_index == 1:
                        image_tokens[frame_id, i, j] = torch.where(mask_frame, prev_patch_frame, cur_patch)
                    elif max_index == 2:
                        image_tokens[frame_id, i, j] = torch.where(mask_left, prev_patch_left, cur_patch)
                    elif max_index == 3:
                        image_tokens[frame_id, i, j] = torch.where(mask_up, prev_patch_up, cur_patch)

        self.sparsity_list.append(num_similar.item() / num_elements) if num_elements > 0 else 0
        image_tokens = image_tokens.view(-1, image_tokens.shape[-1])
        return image_tokens

        
    def element_wise_simisparse_spatiotemporal_fast(self, image_tokens: torch.Tensor, L1_norm=None) -> torch.Tensor:
        """
        Supports two input shapes:
        - Single-head: (1, sequence_length, hidden_dim)
        - Multi-head:  (num_head, sequence_length, dim_per_head)
        
        In both cases, we assume:
        sequence_length == self.image_token_length
        hidden_dim or dim_per_head can be split as:
            num_tiles * tile_size,
        and F = self.image_token_length // self.patch_num,
            H = self.patch_height,
            W = self.patch_width.
        
        The output shape will be:
        - Single-head: (sequence_length, hidden_dim)
        - Multi-head:  (num_head, sequence_length, dim_per_head)
        """
        device = image_tokens.device

        # Determine if we are in multi-head mode.
        # For single-head, image_tokens is expected to have shape (1, seq_length, hidden_dim)
        # For multi-head, shape is (num_head, seq_length, dim_per_head)
        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")
        

        
        num_heads, seq_len, dim_per_head = image_tokens.shape
        if seq_len == dim_per_head:
            # it is attention score matrix, use a new threshold
            used_threshold = self.threshold_S
        else:
            used_threshold = self.threshold


        # Compute spatial parameters from the configuration.
        F = self.image_token_length // self.patch_num  # e.g. number of frames
        H = self.patch_height
        W = self.patch_width
        # Last dimension: either hidden_dim (for single-head) or dim_per_head (for multi-head)

        # pad the image_tokens to make sure the hidden_dim is divisible by tile_size
        hidden_dim = image_tokens.shape[-1]

        if hidden_dim % self.tile_size != 0:
            pad_size = self.tile_size - hidden_dim % self.tile_size
            image_tokens = torch.cat([image_tokens, torch.zeros(image_tokens.shape[0], image_tokens.shape[1], pad_size).to(device)], dim=-1)
            hidden_dim = image_tokens.shape[-1]
        else:
            pad_size = 0

        assert hidden_dim % self.tile_size == 0, f"The hidden dimension must be divisible by tile_size. {hidden_dim} % {self.tile_size} != 0"
        num_tiles = hidden_dim // self.tile_size

        # Reshape image_tokens to shape: (num_heads, F, H, W, num_tiles, tile_size)
        image_tokens = image_tokens.view(num_heads, F, H, W, num_tiles, self.tile_size)

        # Prepare multiplier. For proper broadcasting, if L1_norm is given (shape: (num_tiles, tile_size))
        # then reshape it to (1,1,1,1,num_tiles,tile_size)
        if L1_norm is not None:
            multiplier = L1_norm.view(1, 1, 1, 1, num_tiles, self.tile_size)
        else:
            multiplier = 1

        # --- Temporal differences (frame neighbor) ---
        if F > 1:
            # Compute differences for frames 1...F-1 with the previous frame along the F axis.
            temporal_diff = multiplier * torch.abs(image_tokens[:, 1:, ...] - image_tokens[:, :-1, ...])
            mask_temporal = temporal_diff < used_threshold  # shape: (num_heads, F-1, H, W, num_tiles, tile_size)
            count_temporal = mask_temporal.sum(dim=-1)  # shape: (num_heads, F-1, H, W, num_tiles)
        else:
            count_temporal = torch.empty(0, device=device)

        # --- Vertical differences (using previous row, i.e. along H) ---
        if H > 1:
            vertical_diff = multiplier * torch.abs(image_tokens[:, :, 1:, :, :] - image_tokens[:, :, :-1, :, :])
            mask_vertical = vertical_diff < used_threshold  # shape: (num_heads, F, H-1, W, num_tiles, tile_size)
            # mask_vertical = vertical_diff < 0.0
            count_vertical = mask_vertical.sum(dim=-1)  # shape: (num_heads, F, H-1, W, num_tiles)
        else:
            count_vertical = torch.empty(0, device=device)

        # --- Horizontal differences (using previous column, i.e. along W) ---
        if W > 1:
            horizontal_diff = multiplier * torch.abs(image_tokens[:, :, :, 1:, :] - image_tokens[:, :, :, :-1, :])
            mask_horizontal = horizontal_diff < used_threshold  # shape: (num_heads, F, H, W-1, num_tiles, tile_size)
            # mask_horizontal = horizontal_diff < 0.0
            count_horizontal = mask_horizontal.sum(dim=-1)  # shape: (num_heads, F, H, W-1, num_tiles)
        else:
            count_horizontal = torch.empty(0, device=device)


        if self.token_importance is not None:

            self.token_importance = self.token_importance.view(1, self.num_frames, self.patch_height, self.patch_width, 1, 1)
            token_multiplier = self.token_importance * 0.5
            # token_multiplier = 1
        else:
            token_multiplier = 1

        # --- Zero differences (compare with zeros) ---
        zero_diff = multiplier * torch.abs(image_tokens) * self.zero_multiplier * token_multiplier
        mask_zero = zero_diff < used_threshold  # shape: (num_heads, F, H, W, num_tiles, tile_size)
        count_zero = mask_zero.sum(dim=-1)  # shape: (num_heads, F, H, W, num_tiles)

        # --- Build full count maps for each neighbor type (target shape: num_heads x F x H x W x num_tiles) ---
        count_temporal_full = torch.full((num_heads, F, H, W, num_tiles), -1, dtype=torch.int64, device=device)
        if F > 1:
            count_temporal_full[:, 1:] = count_temporal.to(torch.int64)
        
        count_vertical_full = torch.full((num_heads, F, H, W, num_tiles), -1, dtype=torch.int64, device=device)
        if H > 1:
            count_vertical_full[:, :, 1:] = count_vertical.to(torch.int64)
        
        count_horizontal_full = torch.full((num_heads, F, H, W, num_tiles), -1, dtype=torch.int64, device=device)
        if W > 1:
            count_horizontal_full[:, :, :, 1:] = count_horizontal.to(torch.int64)
        
        # Baseline candidate: do nothing (all counts 0)
        baseline_count = torch.zeros((num_heads, F, H, W, num_tiles), dtype=torch.int64, device=device)
        
        # Stack counts: candidate index 0 is baseline, 1 is temporal, 2 is vertical, 3 is horizontal, 4 is zero.
        candidate_counts = torch.stack(
            [baseline_count, count_temporal_full, count_vertical_full, count_horizontal_full, count_zero], dim=0
        )
        # candidate_counts: shape (5, num_heads, F, H, W, num_tiles)

        # For each tile, select the candidate with the maximum count.
        best_option = candidate_counts.argmax(dim=0)  # shape: (num_heads, F, H, W, num_tiles)
        best_count = candidate_counts.max(dim=0).values  # shape: (num_heads, F, H, W, num_tiles)
        num_similar = best_count.sum()  # total number of similar elements

        # --- Compute candidate outputs ---
        # Candidate 0: baseline (original tokens)
        candidate_baseline = image_tokens

        # Candidate 1: Temporal update
        candidate_temporal = image_tokens.clone()
        if F > 1:
            candidate_temporal[:, 1:] = torch.where(
                mask_temporal, image_tokens[:, :-1, ...], image_tokens[:, 1:, ...]
            )

        # Candidate 2: Vertical update
        candidate_vertical = image_tokens.clone()
        if H > 1:
            candidate_vertical[:, :, 1:, :, :] = torch.where(
                mask_vertical, image_tokens[:, :, :-1, :, :], image_tokens[:, :, 1:, :, :]
            )

        # Candidate 3: Horizontal update
        candidate_horizontal = image_tokens.clone()
        if W > 1:
            candidate_horizontal[:, :, :, 1:, :] = torch.where(
                mask_horizontal, image_tokens[:, :, :, :-1, :], image_tokens[:, :, :, 1:, :]
            )

        # Candidate 4: Zero update
        candidate_zero = torch.where(mask_zero, torch.zeros_like(image_tokens), image_tokens)
        
        # Stack candidate outputs along a new candidate dimension.
        # Shape becomes (5, num_heads, F, H, W, num_tiles, tile_size)
        candidates = torch.stack(
            [candidate_baseline, candidate_temporal, candidate_vertical, candidate_horizontal, candidate_zero], dim=0
        )

        # --- Select final output per tile based on best_option using advanced indexing ---
        # Build index tensors for all dimensions: head, F, H, W, and tile.
        head_idx = torch.arange(num_heads, device=device).view(num_heads, 1, 1, 1, 1).expand(num_heads, F, H, W, num_tiles)
        F_idx    = torch.arange(F, device=device).view(1, F, 1, 1, 1).expand(num_heads, F, H, W, num_tiles)
        H_idx    = torch.arange(H, device=device).view(1, 1, H, 1, 1).expand(num_heads, F, H, W, num_tiles)
        W_idx    = torch.arange(W, device=device).view(1, 1, 1, W, 1).expand(num_heads, F, H, W, num_tiles)
        tile_idx = torch.arange(num_tiles, device=device).view(1, 1, 1, 1, num_tiles).expand(num_heads, F, H, W, num_tiles)

        # candidates has shape (5, num_heads, F, H, W, num_tiles, tile_size)
        # best_option has shape (num_heads, F, H, W, num_tiles) with values in {0,1,2,3,4}.
        final_output = candidates[
            best_option,         # candidate index (selects from the first dim)
            head_idx, F_idx, H_idx, W_idx, tile_idx, :
        ]
        # final_output: shape (num_heads, F, H, W, num_tiles, tile_size)

        # Compute sparsity value (ratio of similar elements over total elements)
        total_elements = image_tokens.numel()  # includes num_heads * F * H * W * num_tiles * tile_size
        sparsity_value = (num_similar / total_elements).item()
        # self.sparsity_list.append(sparsity_value.item())

        # Reshape back to the original "sequence" layout.
        # For multi-head, we return (num_heads, sequence_length, dim_per_head)
        final_output = final_output.view(num_heads, -1, num_tiles * self.tile_size)

        # Remove padding if added.
        if pad_size > 0:
            final_output = final_output[:, :, :-pad_size]
        
        # Optionally, if you want to mimic the original function for single-head inputs, you could:
        # if num_heads == 1:
        #     final_output = final_output.squeeze(0)
        
        return final_output, sparsity_value


    def vector_wise_simisparse_spatiotemporal(self, image_tokens: torch.Tensor, L1_norm=None) -> torch.Tensor:
        # sparsity type: dense: 0,1,2, all zero: 6,7,8, 4:8: 4,5,6

        used_threshold = self.threshold

        num_frames = self.image_token_length // self.patch_num
        image_tokens = image_tokens.view(num_frames, self.patch_height, self.patch_width, image_tokens.shape[-1])

        num_elements = image_tokens.numel()
        num_similar = 0

        vector_length = 8
        num_vector = image_tokens.shape[-1] // vector_length
        assert image_tokens.shape[-1] % vector_length == 0

        for frame_id in range(num_frames):
            for i in range(self.patch_height):
                for j in range(self.patch_width):

                    num_similar_array = torch.full((4, num_vector), -1).to(image_tokens.device) # blank, frame, left, up
                    num_similar_array[0,:] = 0

                    cur_patch = image_tokens[frame_id, i, j]
                    if frame_id != 0:
                        prev_patch_frame = image_tokens[frame_id - 1, i, j]
                        diff_frame = torch.abs(prev_patch_frame - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_frame - cur_patch)
                        diff_frame = diff_frame.view(num_vector, vector_length)
                        mask_frame = diff_frame < used_threshold
                        num_similar_array[1,:] = torch.sum(mask_frame, dim=-1)
                    if i != 0:
                        prev_patch_left = image_tokens[frame_id, i - 1, j]
                        diff_left = torch.abs(prev_patch_left - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_left - cur_patch)
                        diff_left = diff_left.view(num_vector, vector_length)
                        mask_left = diff_left < used_threshold
                        num_similar_array[2,:] = torch.sum(mask_left, dim=-1)
                    if j != 0:
                        prev_patch_up = image_tokens[frame_id, i, j - 1]
                        diff_up = torch.abs(prev_patch_up - cur_patch) * L1_norm if L1_norm is not None else torch.abs(prev_patch_up - cur_patch)
                        diff_up = diff_up.view(num_vector, vector_length)
                        mask_up = diff_up < used_threshold
                        num_similar_array[3,:] = torch.sum(mask_up, dim=-1)

                    max_indices = torch.argmax(num_similar_array, dim=0)
                    frame_max_indices = max_indices == 1
                    left_max_indices = max_indices == 2
                    up_max_indices = max_indices == 3

                    num_similar_frame = torch.where(frame_max_indices, num_similar_array[1,:], torch.zeros_like(num_similar_array[1,:]))
                    num_similar_left = torch.where(left_max_indices, num_similar_array[2,:], torch.zeros_like(num_similar_array[2,:]))
                    num_similar_up = torch.where(up_max_indices, num_similar_array[3,:], torch.zeros_like(num_similar_array[3,:]))

                    # set num_similar_ to 0 if smaller than 2, to 8 if larger than 6, otherwise set to 4
                    num_similar_frame = regularize_to_0_4_8(num_similar_frame)
                    num_similar_left = regularize_to_0_4_8(num_similar_left)
                    num_similar_up = regularize_to_0_4_8(num_similar_up)

                    if frame_id != 0:                    
                        frame_indices_top_4 = torch.topk(diff_frame, 4, dim=-1, largest=False)[1]
                        mask_frame_top_4 = torch.zeros_like(diff_frame).to(torch.bool)
                        batch_indices = torch.arange(diff_frame.shape[0]).unsqueeze(1)
                        mask_frame_top_4[batch_indices, frame_indices_top_4] = True
                        mask_frame = torch.zeros_like(diff_frame).to(torch.bool).to(mask_frame_top_4.device)
                        mask_frame = torch.where(num_similar_frame == 4, mask_frame_top_4, mask_frame)
                        mask_frame = torch.where(num_similar_frame == 8, torch.ones_like(mask_frame, dtype=torch.bool).to(mask_frame.device), mask_frame).view(-1)
                        image_tokens[frame_id, i, j] = torch.where(mask_frame, prev_patch_frame, cur_patch)

                    if i != 0:
                        left_indices_top_4 = torch.topk(diff_left, 4, dim=-1, largest=False)[1]
                        mask_left_top_4 = torch.zeros_like(diff_left).to(torch.bool)
                        batch_indices = torch.arange(diff_left.shape[0]).unsqueeze(1)
                        mask_left_top_4[batch_indices, left_indices_top_4] = True
                        mask_left = torch.zeros_like(diff_left).to(torch.bool).to(mask_left_top_4.device)
                        mask_left = torch.where(num_similar_left == 4, mask_left_top_4, mask_left)
                        mask_left = torch.where(num_similar_left == 8, torch.ones_like(mask_left, dtype=torch.bool).to(mask_left.device), mask_left).view(-1)
                        image_tokens[frame_id, i, j] = torch.where(mask_left, prev_patch_left, image_tokens[frame_id, i, j])

                    if j != 0:
                        up_indices_top_4 = torch.topk(diff_up, 4, dim=-1, largest=False)[1]
                        mask_up_top_4 = torch.zeros_like(diff_up).to(torch.bool)
                        batch_indices = torch.arange(diff_up.shape[0]).unsqueeze(1)
                        mask_up_top_4[batch_indices, up_indices_top_4] = True
                        mask_up = torch.zeros_like(diff_up).to(torch.bool).to(mask_up_top_4.device)
                        mask_up = torch.where(num_similar_up == 4, mask_up_top_4, mask_up)
                        mask_up = torch.where(num_similar_up == 8, torch.ones_like(mask_up, dtype=torch.bool).to(mask_up.device), mask_up).view(-1)
                        image_tokens[frame_id, i, j] = torch.where(mask_up, prev_patch_up, image_tokens[frame_id, i, j])


                    if frame_id != 0:
                        num_similar += torch.sum(mask_frame)
                    if i != 0:
                        num_similar += torch.sum(mask_left)
                    if j != 0:
                        num_similar += torch.sum(mask_up)


        self.sparsity_list.append(num_similar.item() / num_elements) if num_elements > 0 else 0
        image_tokens = image_tokens.view(-1, image_tokens.shape[-1])
        return image_tokens

    
    def vector_wise_simisparse_spatiotemporal_fast(self, image_tokens: torch.Tensor, L1_norm=None, threshold=None) -> tuple[torch.Tensor, float]:
        """   
        Supports two input shapes:
        - Single-head: (1, sequence_length, hidden_dim)
        - Multi-head:  (num_head, sequence_length, dim_per_head)
        
        For each tile (a sub-vector of length tile_size) in the image_tokens,
        four candidate "differences" are computed:
        - Temporal difference (between consecutive frames)
        - Vertical difference (between consecutive rows)
        - Horizontal difference (between consecutive columns)
        - "Self difference" computed via a zero multiplier.
        
        Then, for every tile, the candidate with the smallest difference is selected.
        If that minimal difference is below a computed threshold (used_threshold), the
        current tile is replaced by the candidate's tile value.
        
        Additionally, this function calculates the sparsity value, defined as the fraction
        of positions (tiles) where a substitution occurred.
        """
        device = image_tokens.device

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")

        num_heads, seq_len, hidden_dim = image_tokens.shape
        # Determine which threshold to use.
        if seq_len == hidden_dim:
            used_threshold = self.threshold_S * self.tile_size
        else:
            used_threshold = self.threshold * self.tile_size

        # Compute spatial parameters.
        F = self.image_token_length // self.patch_num  # e.g. number of frames
        H = self.patch_height
        W = self.patch_width

        # Pad image_tokens so that the hidden_dim is divisible by tile_size.
        if hidden_dim % self.tile_size != 0:
            pad_size = self.tile_size - (hidden_dim % self.tile_size)
            pad_tensor = torch.zeros(image_tokens.shape[0], image_tokens.shape[1], pad_size, device=device, dtype=image_tokens.dtype)
            image_tokens = torch.cat([image_tokens, pad_tensor], dim=-1)
            hidden_dim = image_tokens.shape[-1]
        else:
            pad_size = 0

        # Note: Padding is now handled in the forward method, so this assertion is no longer needed
        # assert hidden_dim % self.tile_size == 0, (
        #     f"The hidden dimension must be divisible by tile_size. {hidden_dim} % {self.tile_size} != 0"
        # )
        num_tiles = hidden_dim // self.tile_size

        # Reshape image_tokens to shape: (num_heads, F, H, W, num_tiles, tile_size)
        image_tokens = image_tokens.view(num_heads, F, H, W, num_tiles, self.tile_size)

        # Prepare multiplier for broadcasting if L1_norm is provided.
        if L1_norm is not None:
            multiplier = L1_norm.view(1, 1, 1, 1, num_tiles, self.tile_size).to(image_tokens.dtype)
        else:
            multiplier = 1

        # # instead of compute differences, we can use the cosine similarity
        if F > 1:
            temporal_sim = weighted_cosine_similarity(image_tokens[:, 1:, ...], image_tokens[:, :-1, ...], multiplier)
        else:
            temporal_sim = None

        if H > 1:
            vertical_sim = weighted_cosine_similarity(image_tokens[:, :, 1:, :, :], image_tokens[:, :, :-1, :, :], multiplier)
        else:
            vertical_sim = None

        if W > 1:
            horizontal_sim = weighted_cosine_similarity(image_tokens[:, :, :, 1:, :], image_tokens[:, :, :, :-1, :], multiplier)
        else:
            horizontal_sim = None

        # zero_diff= multiplier * torch.abs(image_tokens) * self.zero_multiplier
        # zero_sim = zero_diff.sum(dim=-1)  # shape: (num_heads, F, H, W, num_tiles)


        # --- Temporal differences (vector-wise) ---
        if F > 1:
            # Compute differences between consecutive frames.
            temporal_diff = multiplier * torch.abs(image_tokens[:, 1:, ...] - image_tokens[:, :-1, ...])
            temporal_diff = temporal_diff.sum(dim=-1)  # shape: (num_heads, F-1, H, W, num_tiles)
        else:
            temporal_diff = None

        # --- Vertical differences (vector-wise) ---
        if H > 1:
            vertical_diff = multiplier * torch.abs(image_tokens[:, :, 1:, :, :] - image_tokens[:, :, :-1, :, :])
            vertical_diff = vertical_diff.sum(dim=-1)  # shape: (num_heads, F, H-1, W, num_tiles)
        else:
            vertical_diff = None

        # --- Horizontal differences (vector-wise) ---
        if W > 1:
            horizontal_diff = multiplier * torch.abs(image_tokens[:, :, :, 1:, :] - image_tokens[:, :, :, :-1, :])
            horizontal_diff = horizontal_diff.sum(dim=-1)  # shape: (num_heads, F, H, W-1, num_tiles)
        else:
            horizontal_diff = None

        # --- Zero differences (vector-wise) ---
        zero_diff = multiplier * torch.abs(image_tokens) * self.zero_multiplier
        zero_diff = zero_diff.sum(dim=-1)  # shape: (num_heads, F, H, W, num_tiles)

        # --- Prepare candidate tensors for substitution ---
        # Use 4 candidates:
        #   0: temporal (from previous frame)
        #   1: vertical (from previous row)
        #   2: horizontal (from previous column)
        #   3: self candidate (current tile itself)
        # Set a large number using the maximum representable value in the input tensor's dtype.
        big_val = torch.finfo(image_tokens.dtype).max
        candidate_diff = torch.full((4, num_heads, F, H, W, num_tiles),
                                    big_val,
                                    device=device,
                                    dtype=image_tokens.dtype)
        candidate_value = torch.zeros((4, num_heads, F, H, W, num_tiles, self.tile_size),
                                    device=device,
                                    dtype=image_tokens.dtype)

        # --- Temporal candidate ---
        if F > 1 and temporal_diff is not None:
            candidate_diff[0, :, 1:, :, :, :] = temporal_diff  # frames 1...F-1
            candidate_value[0, :, 1:, :, :, :] = image_tokens[:, :-1, :, :, :]

        # --- Vertical candidate ---
        if H > 1 and vertical_diff is not None:
            candidate_diff[1, :, :, 1:, :, :] = vertical_diff  # rows 1...H-1
            candidate_value[1, :, :, 1:, :, :] = image_tokens[:, :, :-1, :, :]

        # --- Horizontal candidate ---
        if W > 1 and horizontal_diff is not None:
            candidate_diff[2, :, :, :, 1:, :] = horizontal_diff  # columns 1...W-1
            candidate_value[2, :, :, :, 1:, :] = image_tokens[:, :, :, :-1, :]

        # --- Self candidate ---
        candidate_diff[3] = zero_diff  # cost for keeping the current tile
        candidate_value[3] = torch.zeros_like(image_tokens)  # value for keeping the current tile

        # --- Select candidate with minimum difference ---
        # min_diff and min_idx: shape (num_heads, F, H, W, num_tiles)
        min_diff, min_idx = candidate_diff.min(dim=0)

        # Create mask where the minimal difference is below the threshold.
        substitute_mask = (min_diff < used_threshold)

        # --- Build an updated copy of the tokens ---
        updated_tokens = image_tokens.clone()
        # For positions that meet the condition, replace the tile with the candidate tile.
        for cand in range(4):
            cand_mask = (min_idx == cand) & substitute_mask  # boolean mask
            if cand_mask.any():
                updated_tokens[cand_mask] = candidate_value[cand][cand_mask]

        # Calculate sparsity value: ratio of substituted tiles.
        sparsity_value = substitute_mask.to(torch.float32).mean().item()

        # Reshape updated_tokens back to (num_heads, sequence_length, hidden_dim)
        updated_tokens = updated_tokens.view(num_heads, seq_len, hidden_dim)
        
        return updated_tokens, sparsity_value


    def vector_wise_simisparse_temporal(self, image_tokens: torch.Tensor, L1_norm=None) -> torch.Tensor:
        device = image_tokens.device

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")

        num_heads, seq_len, hidden_dim = image_tokens.shape
        tile_num = hidden_dim // self.tile_size
        # Note: Padding is now handled in the forward method, so this assertion is no longer needed
        # assert hidden_dim % self.tile_size == 0, \
        #     f"The hidden dimension must be divisible by tile_size. {hidden_dim} % {self.tile_size} != 0"

        # Clone and reshape image_tokens to have dimensions:
        # (num_heads, num_frames, patch_height, patch_width, tile_num, tile_size)
        updated_tokens = image_tokens.clone()
        updated_tokens = updated_tokens.view(num_heads, self.num_frames, self.patch_height, 
                                            self.patch_width, tile_num, self.tile_size)

        # If only one frame, no temporal similarity is computed.
        if self.num_frames == 1:
            return updated_tokens.view(num_heads, seq_len, hidden_dim), 0.0

        # Compute cosine similarity between temporal adjacent tiles.
        # (Assuming cosine_similarity is defined appropriately to compare vectors along tile_size)
        temporal_sim = cosine_similarity(updated_tokens[:, 1:, ...], updated_tokens[:, :-1, ...])
        # Pad the first frame with zeros so that temporal_sim has the same number of frames.
        padding_shape = (temporal_sim.shape[0], 1, temporal_sim.shape[2], temporal_sim.shape[3], temporal_sim.shape[4])
        temporal_sim = torch.cat([torch.zeros(padding_shape, device=device), temporal_sim], dim=1)

        # Create mask: True if cosine similarity > threshold, False otherwise.
        mask = temporal_sim > self.threshold

        # Create a frame index for each frame.
        # Shape: (1, num_frames, 1, 1, 1) so it broadcasts correctly.
        frame_idx = torch.arange(self.num_frames, device=device).view(1, self.num_frames, 1, 1, 1)

        # For positions where mask is False (cosine similarity <= threshold), assign the frame index,
        # else set to -1.
        false_frame_idx = torch.where(mask == False, frame_idx, -torch.ones_like(frame_idx))
        
        # Cumulatively propagate the last encountered false frame index along the temporal dimension.
        # After this, assignment_idx has shape (num_heads, num_frames, patch_height, patch_width, tile_num)
        assignment_idx = false_frame_idx.cummax(dim=1)[0]

        # ***** NEW GROUP-AVERAGING PART *****
        # Instead of gathering a single vector per assignment index, we now average over
        # all token vectors that share the same assignment index.
        #
        # First, flatten the dimensions that are "static" over time:
        # B = num_heads * patch_height * patch_width * tile_num.
        # Each row of shape (num_frames, tile_size) now corresponds to a specific
        # spatial/head/tile location through time.
        B = num_heads * self.patch_height * self.patch_width * tile_num
        # Rearranging updated_tokens so that time is the second dimension:
        tokens_flat = updated_tokens.permute(0, 2, 3, 4, 1, 5).contiguous().view(B, self.num_frames, self.tile_size)
        # Rearranging assignment_idx similarly: shape becomes (B, num_frames)
        groups_flat = assignment_idx.permute(0, 2, 3, 4, 1).contiguous().view(B, self.num_frames)

        # Create one-hot encoding of the group indices.
        # (Note: group labels are expected to be in the range [0, self.num_frames-1].)
        num_classes = self.num_frames
        # Clamp any negative indices (if any) to 0. In practice, the first frame should always get its own group.
        groups_one_hot = torch.nn.functional.one_hot(groups_flat.clamp(min=0), num_classes=num_classes).to(tokens_flat.dtype)
        # groups_one_hot now has shape (B, self.num_frames, num_classes)
        
        # Sum the token vectors for each group.
        # Transpose the one-hot so that we can use batch matrix multiplication:
        # This produces a tensor of shape (B, num_classes, tile_size) where each "row"
        # along the num_classes dimension is the sum of tokens for that group label.
        group_sum = groups_one_hot.transpose(1, 2).bmm(tokens_flat)
        # Also get the count of tokens in each group (shape: (B, num_classes)).
        group_count = groups_one_hot.transpose(1, 2).sum(dim=-1)
        # Avoid division by zero by replacing 0 counts with 1.
        group_count_masked = group_count.clone()
        group_count_masked[group_count_masked == 0] = 1
        # Compute the average vector for each group.
        group_avg = group_sum / group_count_masked.unsqueeze(-1)
        
        # Now, for each time instance (for each row) we want to pick the average vector
        # corresponding to its group label.
        # Expand groups_flat so that it can index group_avg.
        groups_expanded = groups_flat.unsqueeze(-1).expand(-1, -1, self.tile_size)  # shape: (B, self.num_frames, tile_size)
        # Use gather to pick, for every time step, the average for that group.
        tokens_avg = torch.gather(group_avg, dim=1, index=groups_expanded)  # shape: (B, self.num_frames, tile_size)
        
        # Reshape tokens_avg back to the original shape of updated_tokens:
        # (num_heads, self.num_frames, patch_height, patch_width, tile_num, tile_size)
        tokens_avg = tokens_avg.view(num_heads, self.patch_height, self.patch_width, tile_num, self.num_frames, self.tile_size)
        # Permute back so that time is the second dimension.
        tokens_avg = tokens_avg.permute(0, 4, 1, 2, 3, 5).contiguous()
        
        # Finally, flatten updated_tokens back to (num_heads, seq_len, hidden_dim)
        updated_tokens = tokens_avg.view(num_heads, seq_len, hidden_dim)

        sparsity = torch.sum(mask == True).item() / (num_heads * self.num_frames * self.patch_height * self.patch_width * tile_num)

        # calculate mse
        # mse = torch.mean((updated_tokens - image_tokens) ** 2)
        # print(f"mse: {mse.item()}")
        
        return updated_tokens, sparsity
    

    def vector_wise_simisparse_adaptive_range(self, image_tokens: torch.Tensor, name: str, L1_norm=None) -> torch.Tensor:
        device = image_tokens.device

        # Define the desired spatial matching window sizes.
        base_match_height = self.match_range
        base_match_width = self.match_range

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")

        num_heads, seq_len, hidden_dim = image_tokens.shape
        if hidden_dim > self.tile_size:
            tile_num = hidden_dim // self.tile_size
            cur_tile_size = self.tile_size
        else:
            tile_num = 1
            cur_tile_size = hidden_dim
        # Note: Padding is now handled in the forward method, so this assertion is no longer needed
        # assert hidden_dim % self.tile_size == 0, (
        #     f"The hidden dimension must be divisible by tile_size. {hidden_dim} % {self.tile_size} != 0"
        # )
        

        updated_tokens = (
            image_tokens
            .clone()
            .view(num_heads, seq_len, tile_num, cur_tile_size)
        )                                                     # <-- NEW-BEGIN

        # ---------- 2. prune zeros ----------
        zero_dist   = torch.sum(torch.abs(updated_tokens), dim=-1)     # [H, S, Tn]
        mask_zero   = zero_dist == 0.0
        num_zero_vector = torch.sum(mask_zero).item()

        # get all zero tokens
        mask_zero_token = torch.all(mask_zero, dim=-1)
        mask_zero_token = torch.all(mask_zero_token, dim=0)


        if self.partition_size is not None:
            # count the number of non_zero tokens
            num_non_zero_token = torch.sum(~mask_zero_token).item()

            # determine partition based on number of nonzero tokens
            num_partition = (num_non_zero_token + self.partition_size - 1) // self.partition_size  # ceiling division

            # create cumulative sum of nonzero tokens to find partition boundaries
            nonzero_cumsum = torch.cumsum(~mask_zero_token, dim=0)

            # create partition index ranges ensuring each partition has simi_sparse_partition nonzero tokens
            partition_indices = []
            for i in range(num_partition):
                target_count = (i + 1) * self.partition_size

                # find the index where cumulative sum reaches target_count
                if target_count <= num_non_zero_token:
                    # find the first index where cumsum >= target_count
                    partition_end_idx = torch.searchsorted(nonzero_cumsum, target_count, right=True)
                else:
                    # for the last partition, use the end of sequence
                    partition_end_idx = torch.tensor(seq_len, device=device, dtype=torch.long)

                # find the start index for this partition
                if i == 0:
                    partition_start_idx = torch.tensor(0, device=device, dtype=torch.long)
                else:
                    prev_target_count = i * self.partition_size
                    partition_start_idx = torch.searchsorted(nonzero_cumsum, prev_target_count, right=True)

                partition_indices.append((partition_start_idx.item(), partition_end_idx.item()))

            # convert to tensor for easier handling
            partition_indices = torch.tensor(partition_indices, device=device, dtype=torch.long)



        # ---------- 3. early exit for single-frame streams ----------
        if self.num_frames == 1:
            # For image case (num_frames == 1), still apply the method but with image-specific adjustments
            # Set matching candidate dimensions to 1 (no temporal dimension)
            match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
            match_width  = base_match_width  if self.patch_width  >= base_match_width  else self.patch_width

            # Container for matching candidates - only spatial matching, no temporal
            matching_candidate = torch.zeros(
                (1, match_height, match_width, num_heads, seq_len, tile_num, cur_tile_size),
                device=device,
            )

            # Build candidate set for same-frame matching only (no previous frame)
            dh_offsets = torch.arange(match_height, device=device).view(match_height, 1, 1, 1, 1, 1)
            dw_offsets = torch.arange(match_width, device=device).view(1, match_width, 1, 1, 1, 1)
            linear_offsets = dh_offsets * self.height_stride + dw_offsets * self.width_stride
            
            pos_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len, 1).expand(match_height, match_width, num_heads, seq_len, tile_num)
            
            # Handle the identity case (dh=0, dw=0) separately
            matching_candidate[0, 0, 0] = updated_tokens
            
            # Vectorized processing for non-identity cases
            for dh in range(match_height):
                for dw in range(match_width):
                    if dh == 0 and dw == 0:
                        continue  # Already handled above
                    
                    linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                    shifted_tokens = _shift_seq(updated_tokens, linear_offset)
                    
                    if self.partition_size is not None:
                        # Calculate candidate positions for this shift
                        candidate_pos = (pos_idx[dh, dw] - linear_offset).clamp(0, seq_len - 1)
                        
                        # Check if candidates are in the same partition as current positions
                        partition_mask = _get_partition_mask_vectorized(candidate_pos)
                        
                        # Set candidates to zero if they're not in the same partition
                        matching_candidate[0, dh, dw] = torch.where(
                            partition_mask.unsqueeze(-1).expand_as(shifted_tokens),
                            shifted_tokens,
                            torch.zeros_like(shifted_tokens)
                        )
                    else:
                        # Original behavior: no partition restrictions
                        matching_candidate[0, dh, dw] = shifted_tokens

            # Remove self-matching
            matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0])

            # Apply arbitrary position masking if specified
            if self.mask_positions is not None:
                matching_candidate = _apply_position_masking(matching_candidate, match_height, match_width)

            # Compute cosine similarity
            updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
                1, match_height, match_width, num_heads, seq_len, tile_num, cur_tile_size
            )
            similarity = cosine_similarity(updated_tokens_exp, matching_candidate)

            # Merge the candidate dimensions (only spatial, no temporal)
            similarity = similarity.view(match_height * match_width, num_heads, seq_len, tile_num)

            # For each token, select the candidate with maximum similarity
            max_similarity, max_index = similarity.max(dim=0)
            assert not torch.isnan(max_similarity).any(), "NaN values found in max_similarity."

            # Create a mask for tokens that meet the similarity threshold
            mask = max_similarity > self.threshold

            # Derive candidate offset components (only spatial, no temporal)
            candidate_offset = max_index  # No need to handle temporal dimension
            offset_i = candidate_offset // match_width  # Row offset
            offset_j = candidate_offset % match_width   # Column offset

            # For tokens that do not meet the threshold, set the offsets to 0 so they match themselves
            offset_i = torch.where(mask, offset_i, torch.zeros_like(offset_i))
            offset_j = torch.where(mask, offset_j, torch.zeros_like(offset_j))

            # Build linear mapping for each token based on candidate offsets (spatial only)
            h_idx   = torch.arange(num_heads, device=device).view(num_heads, 1, 1).expand(num_heads, seq_len, tile_num)
            pos_idx = torch.arange(seq_len,   device=device).view(1,        seq_len, 1).expand(num_heads, seq_len, tile_num)
            tile_idx= torch.arange(tile_num,  device=device).view(1, 1,  tile_num).expand(num_heads, seq_len, tile_num)

            # Spatial offsets only (no temporal)
            spatial_offset  = offset_i * self.height_stride + offset_j * self.width_stride
            candidate_pos   = (pos_idx - spatial_offset).clamp(0, seq_len - 1)

            tokens_per_head = seq_len * tile_num
            linear_idx      = h_idx * tokens_per_head + pos_idx      * tile_num + tile_idx
            candidate_linear= h_idx * tokens_per_head + candidate_pos* tile_num + tile_idx

            # Flatten
            M = num_heads * seq_len * tile_num
            linear_idx_flat       = linear_idx.view(M)
            candidate_linear_flat = candidate_linear.view(M)
            mask_flat             = mask.view(M)

            # Initialize pointer for each token
            pointer = torch.where(mask_flat, candidate_linear_flat, linear_idx_flat)

            # Perform pointer-chasing (union-find style)
            max_iters = 1  # For image case, we don't need multiple iterations
            for _ in range(max_iters):
                new_pointer = pointer[pointer]
                if torch.equal(new_pointer, pointer):
                    break
                pointer = new_pointer

            # Reshape pointer back
            group_idx = pointer.view(num_heads, seq_len, tile_num)

            # Group tokens using the computed group index
            tokens_flat = updated_tokens.reshape(M, cur_tile_size)
            group_idx_flat = group_idx.view(M)

            # Compute group-wise sums and counts
            num_groups = int(group_idx_flat.max().item()) + 1

            group_sum = torch.zeros((num_groups, cur_tile_size), device=device, dtype=torch.float32)
            group_count = torch.zeros(num_groups, device=device, dtype=torch.float32)
            group_sum = group_sum.index_add(0, group_idx_flat, tokens_flat.to(torch.float32))
            ones = torch.ones_like(group_idx_flat, dtype=torch.float32)
            group_count = group_count.index_add(0, group_idx_flat, ones)
            group_count[group_count == 0] = 1
            group_avg = group_sum / group_count.unsqueeze(-1)
            group_avg = group_avg.to(tokens_flat.dtype)

            # Assign the averaged value to each token in the group
            tokens_avg_flat = group_avg[group_idx_flat]
            tokens_avg = tokens_avg_flat.view(num_heads, seq_len, tile_num, cur_tile_size)
            updated_tokens_out = tokens_avg.view(num_heads, seq_len, hidden_dim)

            # Compute sparsity metric
            num_similar_vectors = torch.sum(mask_flat).item()
            num_total_vectors = num_heads * seq_len * tile_num
            sparsity = (num_similar_vectors + num_zero_vector) / num_total_vectors

            if self.export_simi_sparse:
                self.info_dict["mask_zero"][name][self.cur_layer] = mask_zero.cpu()
                self.info_dict["mask_similar"][name][self.cur_layer] = mask.cpu()
                self.info_dict["group_idx"][name][self.cur_layer] = group_idx.cpu()

            return updated_tokens_out, sparsity

        # ---------- 4. spatial window size (same as before) ----------
        match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
        match_width  = base_match_width  if self.patch_width  >= base_match_width  else self.patch_width

        # ---------- 5. container for matching candidates ----------
        matching_candidate = torch.zeros(
            (2, match_height, match_width, num_heads, seq_len, tile_num, cur_tile_size),
            device=device,
        )

        # ---------- helper : shift along sequence axis ----------
        def _shift_seq(x: torch.Tensor, offset: int) -> torch.Tensor:
            """
            Shift `x` (H, S, Tn, Ts) along the sequence axis (dim=1).
            Pads with zeros where data rolled out.
            Positive offset ==> roll _right_ (towards larger index).
            Negative offset ==> roll _left_  (towards smaller index).
            """
            if offset == 0:
                return x
            if offset > 0:
                pad = torch.zeros_like(x[:, :offset])               # left zeros
                return torch.cat([pad, x[:, :-offset]], dim=1)
            else:                                                   # offset < 0
                offset = -offset
                pad = torch.zeros_like(x[:, :offset])               # right zeros
                return torch.cat([x[:, offset:], pad], dim=1)

        # ---------- helper : check if position is in same partition (optimized) ----------
        def _get_partition_mask_vectorized(candidate_positions: torch.Tensor) -> torch.Tensor:
            """
            Vectorized version to check if candidate positions are in the same partition as current positions.
            
            Args:
                candidate_positions: tensor of shape (num_heads, seq_len, tile_num) containing candidate positions
                
            Returns:
                partition_mask: boolean tensor of same shape, True if candidate is in same partition as current position
            """
            # Create current position indices
            current_positions = torch.arange(seq_len, device=device).view(1, seq_len, 1).expand(num_heads, seq_len, tile_num)
            
            # Find which partition each current position belongs to
            current_partition_idx = torch.zeros_like(current_positions, dtype=torch.long)
            candidate_partition_idx = torch.zeros_like(candidate_positions, dtype=torch.long)
            
            # Vectorized partition assignment
            for i, (start, end) in enumerate(partition_indices):
                current_mask = (current_positions >= start) & (current_positions < end)
                candidate_mask = (candidate_positions >= start) & (candidate_positions < end)
                
                current_partition_idx[current_mask] = i
                candidate_partition_idx[candidate_mask] = i
            
            # Check if current and candidate positions are in the same partition
            partition_mask = (current_partition_idx == candidate_partition_idx)
            
            return partition_mask

        # ---------- helper : apply arbitrary position masking ----------
        def _apply_position_masking(matching_candidate: torch.Tensor, match_height: int, match_width: int) -> torch.Tensor:
            """
            Apply masking to specific positions in the matching candidate tensor.
            
            Args:
                matching_candidate: tensor of shape (2, match_height, match_width, num_heads, seq_len, tile_num, tile_size)
                match_height: height of the matching window
                match_width: width of the matching window
                
            Returns:
                masked_matching_candidate: tensor with specified positions masked to zero
            """
            if self.mask_positions is None:
                return matching_candidate
                
            # Create a copy to avoid modifying the original
            masked_candidate = matching_candidate.clone()
            
            # Total number of positions: 2 * match_height * match_width
            total_positions = 2 * match_height * match_width
            
            # Validate mask_positions
            if not isinstance(self.mask_positions, (list, tuple)):
                raise ValueError("mask_positions must be a list or tuple of integers")
            
            for pos_id in self.mask_positions:
                if not isinstance(pos_id, int):
                    raise ValueError(f"All elements in mask_positions must be integers, got {type(pos_id)}")
                if pos_id < 0 or pos_id >= total_positions:
                    raise ValueError(f"Position ID {pos_id} is out of range [0, {total_positions-1}]")
                
                # Convert position ID to tensor indices
                # Position ID mapping:
                # 0 to match_height*match_width-1: same-frame matching (matching_candidate[0, ...])
                # match_height*match_width to 2*match_height*match_width-1: previous-frame matching (matching_candidate[1, ...])
                
                if pos_id < match_height * match_width:
                    # Same-frame matching
                    frame_idx = 0
                    spatial_pos = pos_id
                else:
                    # Previous-frame matching
                    frame_idx = 1
                    spatial_pos = pos_id - match_height * match_width
                
                # Convert spatial position to dh, dw indices
                dh = spatial_pos // match_width
                dw = spatial_pos % match_width
                
                # Apply masking
                masked_candidate[frame_idx, dh, dw] = torch.zeros_like(masked_candidate[frame_idx, dh, dw])
            
            return masked_candidate

        # ============================
        # 1. Build candidate set for same-frame matching (vectorized)
        # ============================
        # Pre-compute all spatial offsets
        dh_offsets = torch.arange(match_height, device=device).view(match_height, 1, 1, 1, 1, 1)
        dw_offsets = torch.arange(match_width, device=device).view(1, match_width, 1, 1, 1, 1)
        linear_offsets = dh_offsets * self.height_stride + dw_offsets * self.width_stride
        
        # Create position indices for all candidates
        pos_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len, 1).expand(match_height, match_width, num_heads, seq_len, tile_num)
        
        # Handle the identity case (dh=0, dw=0) separately
        matching_candidate[0, 0, 0] = updated_tokens
        
        # Vectorized processing for non-identity cases
        for dh in range(match_height):
            for dw in range(match_width):
                if dh == 0 and dw == 0:
                    continue  # Already handled above
                
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                shifted_tokens = _shift_seq(updated_tokens, linear_offset)
                
                if self.partition_size is not None:
                    # Calculate candidate positions for this shift
                    candidate_pos = (pos_idx[dh, dw] - linear_offset).clamp(0, seq_len - 1)
                    
                    # Check if candidates are in the same partition as current positions
                    partition_mask = _get_partition_mask_vectorized(candidate_pos)
                    
                    # Set candidates to zero if they're not in the same partition
                    matching_candidate[0, dh, dw] = torch.where(
                        partition_mask.unsqueeze(-1).expand_as(shifted_tokens),
                        shifted_tokens,
                        torch.zeros_like(shifted_tokens)
                    )
                else:
                    # Original behavior: no partition restrictions
                    matching_candidate[0, dh, dw] = shifted_tokens

        # ============================
        # 2. Build candidate set for previous-frame matching (vectorized)
        # ============================
        for dh in range(match_height):
            for dw in range(match_width):
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                
                # First get the spatial shift
                shifted_tokens = _shift_seq(updated_tokens, linear_offset)
                
                # Then apply temporal shift
                temporal_shifted = _shift_seq(shifted_tokens, self.frame_stride)
                
                if self.partition_size is not None:
                    # Calculate candidate positions for this shift (spatial + temporal)
                    candidate_pos = (pos_idx[dh, dw] - linear_offset - self.frame_stride).clamp(0, seq_len - 1)
                    
                    # Check if candidates are in the same partition as current positions
                    partition_mask = _get_partition_mask_vectorized(candidate_pos)
                    
                    # Set candidates to zero if they're not in the same partition
                    matching_candidate[1, dh, dw] = torch.where(
                        partition_mask.unsqueeze(-1).expand_as(temporal_shifted),
                        temporal_shifted,
                        torch.zeros_like(temporal_shifted)
                    )
                else:
                    # Original behavior: no partition restrictions
                    matching_candidate[1, dh, dw] = temporal_shifted

        # ---------- 3. remove self-matching ----------
        matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0])

        # ---------- 4. apply arbitrary position masking ----------
        if self.mask_positions is not None:
            matching_candidate = _apply_position_masking(matching_candidate, match_height, match_width)

        # ============================
        # 3. Compute cosine similarity between the tokens and all candidates.
        # ============================
        # First, expand updated_tokens so that its shape matches matching_candidate:
        # (2, match_height, match_width, num_heads, num_frames, patch_height, patch_width, tile_num, tile_size)
        # updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
        #     2, match_height, match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, tile_num, self.tile_size
        # )
        updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
            2, match_height, match_width, num_heads, seq_len, tile_num, cur_tile_size
        )

        # Compute cosine similarity along the last dimension (tile_size).
        # (Assume cosine_similarity is defined to compute similarity along the last dimension.)
        similarity = cosine_similarity(updated_tokens_exp, matching_candidate)




        # similarity now has shape:
        # (2, match_height, match_width, num_heads, self.num_frames, patch_height, patch_width, tile_num)
        # Merge the candidate dimensions:
        # similarity = similarity.view(2 * match_height * match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, tile_num)
        similarity = similarity.view(2 * match_height * match_width, num_heads, seq_len, tile_num)

        # For each token, select the candidate with maximum similarity.
        max_similarity, max_index = similarity.max(dim=0)
        
        # mask_zero = max_index == 2 * match_height * match_width
        # assert no nan
        assert not torch.isnan(max_similarity).any(), "NaN values found in max_similarity."

        # Now, max_similarity and max_index have shape:
        # (num_heads, self.num_frames, patch_height, patch_width, tile_num)

        # Create a mask for tokens that meet the similarity threshold.
        mask = max_similarity > self.threshold

        # ============================
        # 4. Derive candidate offset components.
        # ============================
        # We have a total of match_height * match_width candidates for each candidate type.
        total_candidates = match_height * match_width
        # candidate_type: 0 means from the current frame, 1 from the previous frame. 2 means all zero
        candidate_type = max_index // total_candidates  # Shape: same as max_index.
    

        # candidate_offset is the index within the spatial grid.
        candidate_offset = max_index % total_candidates
        offset_i = candidate_offset // match_width  # Row offset
        offset_j = candidate_offset % match_width   # Column offset

        # For tokens that do not meet the threshold, set the offsets to 0 so they match themselves.
        candidate_type = torch.where(mask, candidate_type, torch.zeros_like(candidate_type))
        offset_i = torch.where(mask, offset_i, torch.zeros_like(offset_i))
        offset_j = torch.where(mask, offset_j, torch.zeros_like(offset_j))


        # ----- Step 5: Build a linear mapping for each token based on candidate offsets -----
        # Shapes: (H, S, Tn) after broadcast
        h_idx   = torch.arange(num_heads, device=device).view(num_heads, 1, 1).expand(num_heads, seq_len, tile_num)
        pos_idx = torch.arange(seq_len,   device=device).view(1,        seq_len, 1).expand(num_heads, seq_len, tile_num)
        tile_idx= torch.arange(tile_num,  device=device).view(1, 1,  tile_num).expand(num_heads, seq_len, tile_num)

        # Spatial & temporal offsets ➔ linear shift along sequence axis
        spatial_offset  = offset_i * self.height_stride + offset_j * self.width_stride          # (H,S,Tn)
        temporal_offset = (candidate_type == 1).long() * self.frame_stride                      # (H,S,Tn)
        candidate_pos   = (pos_idx - spatial_offset - temporal_offset).clamp(0, seq_len - 1)    # (H,S,Tn)

        tokens_per_head = seq_len * tile_num
        linear_idx      = h_idx * tokens_per_head + pos_idx      * tile_num + tile_idx          # (H,S,Tn)
        candidate_linear= h_idx * tokens_per_head + candidate_pos* tile_num + tile_idx          # (H,S,Tn)
        ## NEW-END

        # Flatten
        M = num_heads * seq_len * tile_num
        linear_idx_flat       = linear_idx.view(M)
        candidate_linear_flat = candidate_linear.view(M)
        mask_flat             = mask.view(M)

        # Initialize a pointer for each token. 
        # If mask is False, the token points to itself (i.e. linear_idx_flat), otherwise to its candidate.
        pointer = torch.where(mask_flat, candidate_linear_flat, linear_idx_flat)

        # ----- Step 5b: Perform pointer-chasing (union-find style) -----
        # Iteratively update pointer until convergence.
        max_iters = self.num_frames  # or some suitable cap
        for _ in range(max_iters):
            new_pointer = pointer[pointer]  # pointer = pointer(pointer)
            if torch.equal(new_pointer, pointer):
                break
            pointer = new_pointer

        # At this point, `pointer` holds the final group index for each token.

        # Reshape pointer back to shape (num_heads, self.num_frames, patch_height, patch_width, tile_num)
        group_idx = pointer.view(num_heads, seq_len, tile_num)



        # ----- Step 6: Group tokens using the computed group index -----
        # Flatten the updated tokens and group indices:
        tokens_flat = updated_tokens.reshape(M, cur_tile_size)
        group_idx_flat = group_idx.view(M)

        # Compute group-wise sums and counts:
        num_groups = int(group_idx_flat.max().item()) + 1

        group_sum = torch.zeros((num_groups, cur_tile_size), device=device, dtype=torch.float32)
        group_count = torch.zeros(num_groups, device=device, dtype=torch.float32)
        group_sum = group_sum.index_add(0, group_idx_flat, tokens_flat.to(torch.float32))
        ones = torch.ones_like(group_idx_flat, dtype=torch.float32)
        group_count = group_count.index_add(0, group_idx_flat, ones)
        group_count[group_count == 0] = 1
        group_avg = group_sum / group_count.unsqueeze(-1)
        group_avg = group_avg.to(tokens_flat.dtype)

        # Assign the averaged value to each token in the group:
        tokens_avg_flat = group_avg[group_idx_flat]
        tokens_avg = tokens_avg_flat.view(num_heads, seq_len, tile_num, cur_tile_size)
        updated_tokens_out = tokens_avg.view(num_heads, seq_len, hidden_dim)

        # Compute a sparsity metric (e.g. fraction of tokens that had a max similarity above threshold).
        num_similar_vectors = torch.sum(mask_flat).item()
        num_total_vectors = num_heads * seq_len * tile_num
        sparsity = (num_similar_vectors + num_zero_vector) / num_total_vectors

        if self.export_simi_sparse:
            self.info_dict["mask_zero"][name][self.cur_layer] = mask_zero.cpu()
            self.info_dict["mask_similar"][name][self.cur_layer] = mask.cpu()
            self.info_dict["group_idx"][name][self.cur_layer] = group_idx.cpu()


        return updated_tokens_out, sparsity

    def merge_tokens(self, hidden_states, position_embeddings, attention_mask):
        device = hidden_states.device
        if self.stop_merge:
            return hidden_states, position_embeddings, attention_mask

        # Add safety check for state consistency
        if self.start_drop and self.retained_ids is None:
            raise ValueError("start_drop is True but retained_ids is None. This indicates inconsistent state.")

        if self.start_drop:
            hidden_states = self.recover_tokens(hidden_states)
        num_heads, seq_len, hidden_dim = hidden_states.shape

        # Validate tensor dimensions
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states to be 3D, got {hidden_states.dim()}D")
        
        # if position_embeddings[0].shape[1] != seq_len:
        #     raise ValueError(f"Position embeddings dimension mismatch: {position_embeddings[0].shape[1]} != {seq_len}")

        image_tokens = hidden_states[:, self.image_token_start_index:self.image_token_start_index + self.image_token_length, :]
        image_tokens = image_tokens.view(num_heads, self.num_frames, self.patch_height, self.patch_width, hidden_dim)

        base_match_height = self.match_range
        base_match_width = self.match_range
        # If there is only one frame or spatial dimension is 1, skip adaptive range computation.
        if self.num_frames == 1:
            hidden_states = self.drop_tokens(hidden_states)
            return hidden_states, position_embeddings, attention_mask

        # Adjust matching window if the patch dimensions are smaller.
        match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
        match_width = base_match_width if self.patch_width >= base_match_width else self.patch_width

        # Use try-catch for CUDA memory management
        try:
            matching_candidate = torch.zeros(
                (2, match_height, match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, hidden_dim),
                device=device,
            )

            for i in range(match_height):
                for j in range(match_width):
                    if i == 0 and j == 0:
                        matching_candidate[0, i, j] = image_tokens
                    else:
                        # For a nonzero spatial offset, shift the tokens by (i, j) in the spatial domain.
                        # For tokens shifted "up" and "left", we take a subset and then pad to preserve dimensions.
                        # ensure correct slicing when i or j is 0
                        if i == 0:
                            tmp = image_tokens[:, :, :, :-j, :]
                        elif j == 0:
                            tmp = image_tokens[:, :, :-i, :, :]
                        else:
                            tmp = image_tokens[:, :, :-i, :-j, :]

                        pad = (
                            0, 0,  # no padding for hidden_dim
                            j, 0,  # pad width at the left (j tokens missing)
                            i, 0
                            )  # pad height at the top (i tokens missing)
                        tmp = torch.nn.functional.pad(tmp, pad, mode='constant', value=0)
                        matching_candidate[0, i, j] = tmp


            tmp = matching_candidate[0, :, :, :, :-1, ...]  # shape becomes (match_height, match_width, num_heads, num_frames-1, patch_height, patch_width, hidden_dim)
            pad_temporal = (0, 0,  # hidden_dim
                            0, 0,  # patch_width
                            0, 0,  # patch_height
                            1, 0)  # pad one frame at the beginning along the num_frames dimension
            tmp = torch.nn.functional.pad(tmp, pad_temporal, mode='constant', value=0)
            matching_candidate[1] = tmp

            matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0], device=device)

            image_tokens_exp = image_tokens.unsqueeze(0).unsqueeze(0).expand(
                2, match_height, match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, hidden_dim
            )
            similarity = cosine_similarity(image_tokens_exp, matching_candidate)
            similarity = similarity.view(2 * match_height * match_width, num_heads, self.num_frames, self.patch_height, self.patch_width)

            max_similarity, max_index = similarity.max(dim=0)
            mask = max_similarity > self.simi_merge_threshold

            total_candidates = match_height * match_width
            candidate_type = max_index // total_candidates  # Shape: same as max_index.

            candidate_offset = max_index % total_candidates
            offset_i = candidate_offset // match_width  # Row offset
            offset_j = candidate_offset % match_width   # Column offset

            candidate_type = torch.where(mask, candidate_type, torch.zeros_like(candidate_type))
            offset_i = torch.where(mask, offset_i, torch.zeros_like(offset_i))
            offset_j = torch.where(mask, offset_j, torch.zeros_like(offset_j))


            h_idx = torch.arange(num_heads, device=device).view(num_heads, 1, 1, 1).expand(num_heads, self.num_frames, self.patch_height, self.patch_width)
            t_idx = torch.arange(self.num_frames, device=device).view(1, self.num_frames, 1, 1).expand(num_heads, self.num_frames, self.patch_height, self.patch_width)
            r_idx = torch.arange(self.patch_height, device=device).view(1, 1, self.patch_height, 1).expand(num_heads, self.num_frames, self.patch_height, self.patch_width)
            c_idx = torch.arange(self.patch_width, device=device).view(1, 1, 1, self.patch_width).expand(num_heads, self.num_frames, self.patch_height, self.patch_width)

            # Compute candidate coordinates as before.
            candidate_frame = torch.where(candidate_type == 0, t_idx, t_idx - 1).clamp(min=0, max=self.num_frames - 1)
            candidate_row   = (r_idx - offset_i).clamp(min=0, max=self.patch_height - 1)
            candidate_col   = (c_idx - offset_j).clamp(min=0, max=self.patch_width - 1)

            linear_idx = (
                h_idx * (self.num_frames * self.patch_height * self.patch_width)
                + t_idx * (self.patch_height * self.patch_width)
                + r_idx * (self.patch_width)
                + c_idx
            )
            candidate_linear = (
                h_idx * (self.num_frames * self.patch_height * self.patch_width)
                + candidate_frame * (self.patch_height * self.patch_width)
                + candidate_row * (self.patch_width)
                + candidate_col
            )


            M = num_heads * self.num_frames * self.patch_height * self.patch_width
            linear_idx_flat = linear_idx.view(M)
            candidate_linear_flat = candidate_linear.view(M)
            mask_flat = mask.view(M)

            pointer = torch.where(mask_flat, candidate_linear_flat, linear_idx_flat)
            max_iters = self.num_frames  # or some suitable cap
            for _ in range(max_iters):
                new_pointer = pointer[pointer]  # pointer = pointer(pointer)
                if torch.equal(new_pointer, pointer):
                    break
                pointer = new_pointer

            group_idx = pointer.view(num_heads, self.num_frames, self.patch_height, self.patch_width)

            tokens_flat = image_tokens.reshape(M, hidden_dim)
            group_idx_flat = group_idx.view(M)

            # Compute group-wise sums and counts:
            num_groups = int(group_idx_flat.max().item()) + 1

            retained_ids = torch.unique(group_idx_flat)
            retained_ids = torch.sort(retained_ids)[0]

            # Add safety check for retained_ids
            if retained_ids.shape[0] == 0:
                raise ValueError("No tokens retained after merging. This may indicate an error in the merging process.")

            if self.start_drop:

                position_embeddings, attention_mask = self.recover_PE_and_AM(position_embeddings, attention_mask)
                before_image_length = self.image_token_start_index
                after_image_length = seq_len - self.image_token_end_index - 1
                
                # Add bounds checking
                if self.retained_ids.shape[0] <= before_image_length + after_image_length:
                    raise ValueError("retained_ids too short for expected structure")
                
                retained_ids_prev = self.retained_ids[before_image_length:-after_image_length] - self.image_token_start_index
                retained_ids = retained_ids[torch.isin(retained_ids, retained_ids_prev)]
                
                # Check if we still have tokens after filtering
                # if retained_ids.shape[0] == 0:
                #     # Fallback: use all retained_ids from merging
                #     retained_ids = torch.unique(group_idx_flat)
                #     retained_ids = torch.sort(retained_ids)[0]
                
                self.image_token_length_cur = retained_ids.shape[0]
                ids_after_image = self.retained_ids[0:before_image_length]
                ids_before_image = self.retained_ids[-after_image_length:]
                self.retained_ids = torch.cat([ids_after_image, retained_ids + self.image_token_start_index, ids_before_image], dim=0).contiguous()

                # assert no repeat in retained_ids
                assert len(self.retained_ids) == len(torch.unique(self.retained_ids)), "There are repeated ids in retained_ids."

            else:
                self.start_drop = True
                self.image_token_length_cur = retained_ids.shape[0]
                before_image_length = self.image_token_start_index
                after_image_length = seq_len - self.image_token_end_index - 1
                ids_before_image = torch.arange(0, before_image_length.item(), device=device) # maintain the tokens before the image
                ids_after_image = torch.arange(self.image_token_end_index.item() + 1, seq_len, device=device)
                self.retained_ids = torch.cat([ids_before_image, retained_ids + self.image_token_start_index, ids_after_image], dim=0).contiguous()

                # assert no repeat in retained_ids
                assert len(self.retained_ids) == len(torch.unique(self.retained_ids)), "There are repeated ids in retained_ids."

            # Ensure num_groups is large enough
            if num_groups <= retained_ids.max().item():
                num_groups = int(retained_ids.max().item()) + 1

            group_sum = torch.zeros((num_groups, hidden_dim), device=device, dtype=torch.float32)
            group_count = torch.zeros(num_groups, device=device, dtype=torch.float32)
            group_sum = group_sum.index_add(0, group_idx_flat, tokens_flat.to(torch.float32))
            ones = torch.ones_like(group_idx_flat, dtype=torch.float32)
            group_count = group_count.index_add(0, group_idx_flat, ones)
            group_count[group_count == 0] = 1
            group_avg = group_sum / group_count.unsqueeze(-1)
            group_avg = group_avg.to(tokens_flat.dtype)

            # Add bounds checking for retained_ids indexing
            if retained_ids.max().item() >= group_avg.shape[0]:
                raise ValueError(f"retained_ids index {retained_ids.max().item()} out of bounds for group_avg with shape {group_avg.shape}")

            image_tokens_out = group_avg[retained_ids].unsqueeze(0)
            tokens_before_image = hidden_states[:, :self.image_token_start_index, :]
            tokens_after_image = hidden_states[:, self.image_token_end_index + 1:, :]
            hidden_states = torch.cat([tokens_before_image, image_tokens_out, tokens_after_image], dim=1).contiguous()

            # Add dimension validation before indexing
            if attention_mask is not None:
                if self.retained_ids.max().item() >= attention_mask.shape[-1]:
                    raise ValueError(f"retained_ids index {self.retained_ids.max().item()} out of bounds for attention_mask with shape {attention_mask.shape}")
                attention_mask = attention_mask[:, :, self.retained_ids, :][:, :, :, self.retained_ids]
            
            if self.retained_ids.max().item() >= position_embeddings[0].shape[1]:
                raise ValueError(f"retained_ids index {self.retained_ids.max().item()} out of bounds for position_embeddings with shape {position_embeddings[0].shape}")
            
            position_embeddings[0] = position_embeddings[0][:, self.retained_ids, :]
            position_embeddings[1] = position_embeddings[1][:, self.retained_ids, :]

        except RuntimeError as e:
            if "CUDA" in str(e):
                # Clear CUDA cache and try to recover
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA error in merge_tokens: {e}. Try reducing batch size or sequence length.")
            else:
                raise e

        return hidden_states, position_embeddings, attention_mask

    def update_alpha(self, layer_idx):
        # find idx of layer_idx in self.selected_layer
        idx = self.selected_layer.index(layer_idx)
        self.cur_alpha = self.alpha[idx]

    def prune_tokens(self, position_embeddings, attention_mask):
        if self.token_importance is None:
            raise ValueError("Token importance is not set. Please set it before pruning.")
        
        # Add safety check for state consistency
        if self.start_drop and self.retained_ids is None:
            raise ValueError("start_drop is True but retained_ids is None. This indicates inconsistent state.")
        
        # get top k important tokens
        k = int(self.image_token_length * self.cur_alpha)
        if k > self.image_token_length_cur:
            k = self.image_token_length_cur
        
        # Ensure k is at least 1
        if k <= 0:
            k = 1
            
        seq_len = self.original_length
        device = position_embeddings[0].device

        # Validate token_importance
        if self.token_importance.shape[0] == 0:
            raise ValueError("token_importance is empty")
        
        if k > self.token_importance.shape[0]:
            k = self.token_importance.shape[0]

        # attention_threshold = 1
        # tmp = self.token_importance > attention_threshold
        # retained_ids_local = torch.nonzero(tmp, as_tuple=False).squeeze(1)

        try:
            retained_ids_local = torch.topk(self.token_importance, k=k)[1]
            # sort the retained ids
            retained_ids_local = torch.sort(retained_ids_local)[0]

            retained_ids = self.retained_ids if self.retained_ids is not None else torch.arange(0, seq_len, device=device)
            
            # Validate retained_ids structure
            if retained_ids.shape[0] < self.image_token_start_index + self.query_token_length:
                raise ValueError(f"retained_ids too short: {retained_ids.shape[0]} < {self.image_token_start_index + self.query_token_length}")
            
            retained_ids_image = retained_ids[self.image_token_start_index:-self.query_token_length]
            retained_ids_before_image = retained_ids[:self.image_token_start_index]
            retained_ids_after_image = retained_ids[-self.query_token_length:]

            # Validate that we have enough image tokens
            if retained_ids_image.shape[0] == 0:
                raise ValueError("No image tokens found in retained_ids")
            
            # Ensure retained_ids_local indices are within bounds
            if retained_ids_local.max().item() >= retained_ids_image.shape[0]:
                # Clamp indices to valid range
                retained_ids_local = torch.clamp(retained_ids_local, 0, retained_ids_image.shape[0] - 1)
                # Remove duplicates
                retained_ids_local = torch.unique(retained_ids_local)

            # use retained_ids_local to select the retained ids
            retained_ids_image = retained_ids_image[retained_ids_local]

            retained_ids_full = torch.cat([retained_ids_before_image, retained_ids_image, retained_ids_after_image], dim=0).contiguous()

            # Validate the final retained_ids_full
            if retained_ids_full.shape[0] == 0:
                raise ValueError("No tokens retained after pruning")

            # gather the hidden states
            # hidden_states = hidden_states[:, retained_ids_full, :]

            if self.start_drop:
                position_embeddings, attention_mask = self.recover_PE_and_AM(position_embeddings, attention_mask)
            else:
                self.start_drop = True

            # Add bounds checking before indexing
            if retained_ids_full.max().item() >= position_embeddings[0].shape[1]:
                raise ValueError(f"retained_ids_full index {retained_ids_full.max().item()} out of bounds for position_embeddings with shape {position_embeddings[0].shape}")
            
            position_embeddings[0] = position_embeddings[0][:, retained_ids_full, :]
            position_embeddings[1] = position_embeddings[1][:, retained_ids_full, :]

            if attention_mask is not None:
                if retained_ids_full.max().item() >= attention_mask.shape[-1]:
                    raise ValueError(f"retained_ids_full index {retained_ids_full.max().item()} out of bounds for attention_mask with shape {attention_mask.shape}")
                attention_mask = attention_mask[:, :, retained_ids_full, :][:, :, :, retained_ids_full]

            self.retained_ids = retained_ids_full

        except RuntimeError as e:
            if "CUDA" in str(e):
                # Clear CUDA cache and try to recover
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA error in prune_tokens: {e}. Try reducing batch size or sequence length.")
            else:
                raise e

        return position_embeddings, attention_mask
    
    def recover_PE_and_AM(self, position_embeddings, attention_mask):
        B, _, C = position_embeddings[0].shape

        assert self.retained_ids.shape[0] == position_embeddings[0].shape[1], "The number of retained ids must be equal to the number of tokens in the position embeddings."
        tmp = torch.zeros(B, self.original_length, C, device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
        tmp[:, self.retained_ids, :] = position_embeddings[0]
        position_embeddings[0] = tmp
        tmp = torch.zeros(B, self.original_length, C, device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
        tmp[:, self.retained_ids, :] = position_embeddings[1]
        position_embeddings[1] = tmp
        if attention_mask is not None:
            tmp = torch.zeros(B, B, self.original_length, self.original_length, device=attention_mask.device, dtype=attention_mask.dtype)
            tmp[:, :, self.retained_ids, :][:, :, :, self.retained_ids] = attention_mask
            attention_mask = tmp
        return position_embeddings, attention_mask



    def recover_tokens(self, hidden_states):
        """
        Recover the full hidden_states by restoring the pruned positions with zeros.
        """
        # Create a zero tensor of the original shape
        if hidden_states.dim() == 3:
            B, _, C = hidden_states.shape
            recovered = torch.zeros(B, self.original_length, C, device=hidden_states.device, dtype=hidden_states.dtype)
        
            # Scatter the retained hidden states back to their original positions
            recovered[:, self.retained_ids, :] = hidden_states

        elif hidden_states.dim() == 4:
            B, H, _, C = hidden_states.shape
            recovered = torch.zeros(B, H, self.original_length, C, device=hidden_states.device, dtype=hidden_states.dtype)
            # Scatter the retained hidden states back to their original positions
            recovered[:, :, self.retained_ids, :] = hidden_states

        else:
            raise ValueError("hidden_states must be 3D or 4D tensor.")

        return recovered
    
    def drop_tokens(self, hidden_states):
        """
        Drop the tokens that are not retained.
        """
        if hidden_states.dim() == 3:
            dropped = hidden_states[:, self.retained_ids, :]
            # make contiguous
            dropped = dropped.contiguous()
        elif hidden_states.dim() == 4:

            dropped = hidden_states[:, :, self.retained_ids, :]
            # make contiguous
            dropped = dropped.contiguous()
        else:
            raise ValueError("hidden_states must be 3D or 4D tensor.")
        
        return dropped

        

    def set_threshold(self, layer_idx, name=None):
        if self.threshold_list is not None:
            assert len(self.threshold_list) == 28
            self.threshold = self.threshold_list[layer_idx]
        return

    def set_mask_positions(self, mask_positions):
        """
        Set the positions to mask in the matching candidate tensor.
        
        Args:
            mask_positions: list or tuple of integers representing position IDs to mask.
                          Position IDs range from 0 to 2*match_height*match_width-1.
                          - 0 to match_height*match_width-1: same-frame matching positions
                          - match_height*match_width to 2*match_height*match_width-1: previous-frame matching positions
        """
        self.mask_positions = mask_positions

    def similarity_analysis(self, image_tokens, name, layer_idx):
        device = image_tokens.device

        # Define the desired spatial matching window sizes.
        base_match_height = self.match_range
        base_match_width = self.match_range

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")


        num_heads, seq_len, hidden_dim = image_tokens.shape
        if num_heads != 1:
            image_tokens = image_tokens.permute(1, 0, 2).contiguous()
            image_tokens = image_tokens.view(1, seq_len, num_heads * hidden_dim)
            hidden_dim = num_heads * hidden_dim
            num_heads = 1


        # ---------- 4. spatial window size (same as before) ----------
        match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
        match_width  = base_match_width  if self.patch_width  >= base_match_width  else self.patch_width

        # ---------- 5. container for matching candidates ----------
        matching_candidate = torch.zeros(
            (2, match_height, match_width, num_heads, seq_len, hidden_dim),
            device=device,
        )

        # ---------- helper : shift along sequence axis ----------
        def _shift_seq(x: torch.Tensor, offset: int) -> torch.Tensor:
            """
            Shift `x` (H, S, Tn, Ts) along the sequence axis (dim=1).
            Pads with zeros where data rolled out.
            Positive offset ==> roll _right_ (towards larger index).
            Negative offset ==> roll _left_  (towards smaller index).
            """
            if offset == 0:
                return x
            if offset > 0:
                pad = torch.zeros_like(x[:, :offset])               # left zeros
                return torch.cat([pad, x[:, :-offset]], dim=1)
            else:                                                   # offset < 0
                offset = -offset
                pad = torch.zeros_like(x[:, :offset])               # right zeros
                return torch.cat([x[:, offset:], pad], dim=1)


        # ============================
        # 1. Build candidate set for same-frame matching (vectorized)
        # ============================
        # Pre-compute all spatial offsets
        dh_offsets = torch.arange(match_height, device=device).view(match_height, 1, 1, 1, 1)
        dw_offsets = torch.arange(match_width, device=device).view(1, match_width, 1, 1, 1)
        linear_offsets = dh_offsets * self.height_stride + dw_offsets * self.width_stride
        
        # Handle the identity case (dh=0, dw=0) separately
        matching_candidate[0, 0, 0] = image_tokens
        
        # Vectorized processing for non-identity cases
        for dh in range(match_height):
            for dw in range(match_width):
                if dh == 0 and dw == 0:
                    continue  # Already handled above
                
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                shifted_tokens = _shift_seq(image_tokens, linear_offset)
                
                matching_candidate[0, dh, dw] = shifted_tokens

        # ============================
        # 2. Build candidate set for previous-frame matching (vectorized)
        # ============================
        for dh in range(match_height):
            for dw in range(match_width):
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                
                # First get the spatial shift
                shifted_tokens = _shift_seq(image_tokens, linear_offset)
                
                # Then apply temporal shift
                temporal_shifted = _shift_seq(shifted_tokens, self.frame_stride)
                
                matching_candidate[1, dh, dw] = temporal_shifted

        # ---------- 3. remove self-matching ----------
        matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0])


        # ============================
        # 3. Compute cosine similarity between the tokens and all candidates.
        # ============================
        # First, expand updated_tokens so that its shape matches matching_candidate:
        # (2, match_height, match_width, num_heads, num_frames, patch_height, patch_width, tile_num, tile_size)
        # updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
        #     2, match_height, match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, tile_num, self.tile_size
        # )
        image_tokens_exp = image_tokens.unsqueeze(0).unsqueeze(0).expand(
            2, match_height, match_width, num_heads, seq_len, hidden_dim
        )

        image_tokens_exp = image_tokens_exp.view(2 * match_height * match_width, num_heads, seq_len, hidden_dim)
        matching_candidate = matching_candidate.view(2 * match_height * match_width, num_heads, seq_len, hidden_dim)

        # vector_length_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
        vector_length_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        hidden_dim = image_tokens_exp.shape[-1]

        ratio_0_9_list = []
        ratio_0_8_list = []

        for vector_length in vector_length_list:
            image_tokens_exp_padded, padding_size = self._pad_hidden_dim(image_tokens_exp, vector_length)
            matching_candidate_padded, padding_size = self._pad_hidden_dim(matching_candidate, vector_length)

            hidden_dim_padded = image_tokens_exp_padded.shape[-1]

            vector_num = hidden_dim_padded // vector_length
            assert hidden_dim_padded % vector_length == 0, (
                f"The hidden dimension must be divisible by vector_length. {hidden_dim_padded} % {vector_length} != 0"
            )

            image_tokens_exp_padded = image_tokens_exp_padded.view(2 * match_height * match_width, num_heads, seq_len, vector_num, vector_length)
            matching_candidate_padded = matching_candidate_padded.view(2 * match_height * match_width, num_heads, seq_len, vector_num, vector_length)

            similarity = cosine_similarity(image_tokens_exp_padded, matching_candidate_padded)

            max_similarity = torch.max(similarity, dim=0)[0]

            # flatten the max_similarity
            max_similarity = max_similarity.view(-1)

            # plot the CCDF (Complementary CDF) of the max_similarity
            max_sim_np = max_similarity.cpu().numpy()
            sorted_max_sim = np.sort(max_sim_np)

            # save the sorted_max_sim to a file
            np.save(f"output/sorted_max_sim_new/{name}_{layer_idx}_{vector_length}_{self.model_name}_{self.dataset_name}.npy", sorted_max_sim)

            # CCDF shows P(X > x) = 1 - P(X <= x)
            # ccdf_max_sim = 1 - np.arange(1, len(sorted_max_sim) + 1) / len(sorted_max_sim)
            # plt.plot(sorted_max_sim, ccdf_max_sim)
            # plt.xlabel('Max Similarity')
            # plt.ylabel('Ratio of Values > Threshold')
            # plt.title(f'CCDF of Max Similarity - {name} Layer {layer_idx} - Vector Length {vector_length}')
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f"output/ccdf/max_similarity_ccdf_{name}_{layer_idx}_{vector_length}.png")
            # plt.close()
            # plt.clf()

            # Calculate ratios for cosine similarity values larger than thresholds

            ratio_0_9 = np.sum(max_sim_np > 0.9) / (len(max_sim_np))
            ratio_0_8 = np.sum(max_sim_np > 0.8) / (len(max_sim_np))

            ratio_0_9_list.append(ratio_0_9)
            ratio_0_8_list.append(ratio_0_8)


        ratio_0_9_diff = ratio_0_9_list[-1] - ratio_0_9_list[0]
        ratio_0_8_diff = ratio_0_8_list[-1] - ratio_0_8_list[0]
        
        print(f"ratio_0_9_diff in name {self.model_name} layer {layer_idx}: {ratio_0_9_diff}")
        print(f"ratio_0_8_diff in name {self.model_name} layer {layer_idx}: {ratio_0_8_diff}")

        # Create or append to CSV file
        # csv_data = {
        #     'name': [name],
        #     'layer_idx': [layer_idx],
        #     'ratio_0_9_diff': [ratio_0_9_diff],
        #     'ratio_0_8_diff': [ratio_0_8_diff]
        # }
        
        # df = pd.DataFrame(csv_data)
        # csv_path = f"output/cdf_csv/{self.model_name}_{self.dataset_name}_similarity_ratio_diffs.csv"
        
        # # Create output directory if it doesn't exist
        # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # # Append to existing file or create new one
        # if os.path.exists(csv_path):
        #     # Read existing data
        #     existing_df = pd.read_csv(csv_path)
        #     # Combine with new data
        #     combined_df = pd.concat([existing_df, df], ignore_index=True)
        #     # Write back to CSV
        #     combined_df.to_csv(csv_path, index=False)
        # else:
        #     df.to_csv(csv_path, index=False)


            # plot the CDF of the min_mse
            # min_mse_np = min_mse.cpu().numpy()
            # sorted_min_mse = np.sort(min_mse_np)
            # cdf_min_mse = np.arange(1, len(sorted_min_mse) + 1) / len(sorted_min_mse)
            # plt.plot(sorted_min_mse, cdf_min_mse)
            # plt.xlabel('Min MSE')
            # plt.ylabel('Cumulative Probability')
            # plt.title(f'CDF of Min MSE - {name} Layer {layer_idx} - Vector Length {vector_length}')
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f"output/cdf/min_mse_cdf_{name}_{layer_idx}_{vector_length}.png")
            # plt.close()
            # plt.clf()




    def validate_state(self):
        """Validate the current state of the SimiSparse module."""
        if self.start_drop and self.retained_ids is None:
            return False, "start_drop is True but retained_ids is None"
        
        if self.retained_ids is not None:
            if self.retained_ids.shape[0] == 0:
                return False, "retained_ids is empty"
            
            if len(self.retained_ids) != len(torch.unique(self.retained_ids)):
                return False, "retained_ids contains duplicates"
        
        return True, "State is valid"

    def reset_state(self):
        """Reset the state to initial values."""
        self.token_importance = None
        self.start_drop = False
        self.retained_ids = None
        self.image_token_length_cur = self.image_token_length
        torch.cuda.empty_cache()

    def get_state_info(self):
        """Get information about the current state for debugging."""
        info = {
            "start_drop": self.start_drop,
            "retained_ids_shape": self.retained_ids.shape if self.retained_ids is not None else None,
            "image_token_length_cur": self.image_token_length_cur,
            "token_importance_shape": self.token_importance.shape if self.token_importance is not None else None,
            "stop_merge": self.stop_merge
        }
        return info

    def safe_merge_and_prune(self, hidden_states, position_embeddings, attention_mask, layer_idx):
        """
        Safely perform both merge_tokens and prune_tokens operations.
        This method ensures proper state management and error handling.
        """
        # Validate state before operations
        is_valid, error_msg = self.validate_state()
        if not is_valid:
            print(f"Warning: Invalid state detected: {error_msg}. Resetting state.")
            self.reset_state()
        
        # Store original shapes for validation
        original_hidden_shape = hidden_states.shape
        original_pe_shape = position_embeddings[0].shape
        
        try:
            # Perform merge_tokens first
            if hidden_states.shape[1] > 1 and self.simi_merge:
                hidden_states, position_embeddings, attention_mask = self.merge_tokens(
                    hidden_states, position_embeddings, attention_mask
                )
            
            # Update alpha for pruning
            if layer_idx in self.selected_layer:
                self.update_alpha(layer_idx=layer_idx)
                
                # Recover tokens if needed before pruning
                if self.start_drop:
                    hidden_states = self.recover_tokens(hidden_states)
                
                # Perform prune_tokens
                position_embeddings, attention_mask = self.prune_tokens(position_embeddings, attention_mask)
                hidden_states = self.drop_tokens(hidden_states)
            
            # Validate final state
            is_valid, error_msg = self.validate_state()
            if not is_valid:
                raise RuntimeError(f"Invalid state after merge_and_prune: {error_msg}")
            
            return hidden_states, position_embeddings, attention_mask
            
        except Exception as e:
            # Reset state on error
            print(f"Error in safe_merge_and_prune: {e}")
            self.reset_state()
            raise e

    def _pad_hidden_dim(self, hidden_states, target_multiple):
        """
        Pad hidden_states to make the hidden dimension divisible by target_multiple.
        
        Args:
            hidden_states: Input tensor of shape (..., hidden_dim)
            target_multiple: The target multiple for the hidden dimension
            
        Returns:
            padded_hidden_states: Padded tensor
            padding_size: Number of elements padded (to be removed later)
        """
        hidden_dim = hidden_states.shape[-1]
        remainder = hidden_dim % target_multiple
        
        if remainder == 0:
            return hidden_states, 0
        
        padding_size = target_multiple - remainder
        
        # Pad the last dimension
        if hidden_states.dim() == 3:
            # (batch, seq_len, hidden_dim)
            padded = torch.nn.functional.pad(hidden_states, (0, padding_size), mode='constant', value=0)
        elif hidden_states.dim() == 4:
            # (batch, num_heads, seq_len, hidden_dim)
            padded = torch.nn.functional.pad(hidden_states, (0, padding_size), mode='constant', value=0)
        else:
            # For other dimensions, pad the last dimension
            pad_tuple = [0] * (hidden_states.dim() * 2)
            pad_tuple[-2] = 0  # No padding before the last dimension
            pad_tuple[-1] = padding_size  # Padding after the last dimension
            padded = torch.nn.functional.pad(hidden_states, pad_tuple, mode='constant', value=0)
        
        return padded, padding_size

    def _unpad_hidden_dim(self, hidden_states, padding_size):
        """
        Remove padding from hidden_states.
        
        Args:
            hidden_states: Padded tensor
            padding_size: Number of elements to remove from the end
            
        Returns:
            unpadded_hidden_states: Original sized tensor
        """
        if padding_size == 0:
            return hidden_states
        
        # Remove the padding from the last dimension
        if hidden_states.dim() == 3:
            # (batch, seq_len, hidden_dim)
            return hidden_states[:, :, :-padding_size]
        elif hidden_states.dim() == 4:
            # (batch, num_heads, seq_len, hidden_dim)
            return hidden_states[:, :, :, :-padding_size]
        else:
            # For other dimensions, slice the last dimension
            slice_obj = [slice(None)] * (hidden_states.dim() - 1) + [slice(None, -padding_size)]
            return hidden_states[slice_obj]

def count_overlap(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Count the number of elements in tensor b that also appear in tensor a.

    Args:
        a (torch.Tensor): 1D tensor of reference indices.
        b (torch.Tensor): 1D tensor of query indices.

    Returns:
        int: Number of elements in b that are also in a.
    """
    return torch.isin(b, a).sum().item()

def regularize_to_0_4_8(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.where(tensor <= 2, torch.zeros_like(tensor), tensor)
    tensor = torch.where(tensor >= 6, torch.full_like(tensor, 8), tensor)
    tensor = torch.where((tensor > 2) & (tensor < 6), torch.full_like(tensor, 4), tensor)
    # broadcast to shape (num_vector, 8)
    tensor = tensor.unsqueeze(1).expand(-1, 8)

    return tensor

def cosine_similarity(mat1, mat2):
    dot_product = torch.sum(mat1 * mat2, dim=-1)
    norm_vec1 = torch.norm(mat1, dim=-1)
    norm_vec2 = torch.norm(mat2, dim=-1)
    denominator = norm_vec1 * norm_vec2
    return torch.where(denominator != 0, dot_product / denominator, torch.zeros_like(denominator))

def mean_squared_error(mat1, mat2):
    # get the mse along the last dimension
    return torch.mean((mat1 - mat2) ** 2, dim=-1)

def weighted_cosine_similarity(mat1, mat2, weights):

    # Compute the weighted dot product.
    dot_product = torch.sum(weights * mat1 * mat2, dim=-1)
    
    # Compute the weighted norms for each vector.
    norm_vec1 = torch.sqrt(torch.sum(weights * mat1 * mat1, dim=-1))
    norm_vec2 = torch.sqrt(torch.sum(weights * mat2 * mat2, dim=-1))
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


def find_contigious_latter_index(index_tensor: torch.LongTensor) -> torch.Tensor:
    """
    Args:
        index_tensor (torch.LongTensor): A binary tensor containing sequences of ones and zeros.

    Returns:
        torch.Tensor: A tensor where each contiguous sequence of ones in the input tensor
                    is replaced by zeros, except for the last element of each sequence,
                    which is replaced by the length of that sequence.

    Example:
        Input:  torch.tensor([0, 1, 1, 1, 0, 0, 1, 1])
        Output: torch.tensor([0, 0, 0, 3, 0, 0, 0, 2])
    """
    bsz, n = index_tensor.shape
    t_prev = torch.cat([torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device), index_tensor[:, :-1]], dim=1)
    t_next = torch.cat([index_tensor[:, 1:], torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device)], dim=1)

    # Identify the starts and ends of runs of ones
    run_starts = (index_tensor == 1) & (t_prev == 0)
    run_ends = (index_tensor == 1) & (t_next == 0)

    start_indices = torch.nonzero(run_starts, as_tuple=True)
    end_indices = torch.nonzero(run_ends, as_tuple=True)
    run_lengths = (end_indices[1] - start_indices[1] + 1).to(index_tensor.dtype)

    output = torch.zeros_like(index_tensor, dtype=index_tensor.dtype)
    output[end_indices[0], end_indices[1]] = run_lengths

    return output
