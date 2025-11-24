import torch


def register_hooks_for_attn(lm):
    attn_weights = {}

    def get_attn(name):
        def hook(module, input, output):
            with torch.no_grad():  # Disable gradient tracking for efficiency
                if isinstance(output, tuple) and len(output) > 1:
                    attn_tensor = output[1].detach().cpu()  # Move to CPU
                    attn_weights[name].append(attn_tensor)
                    del attn_tensor  # Free memory
                    torch.cuda.empty_cache()  # Clear unused memory

        return hook

    lm.model.config.output_attentions = True
    # from transformers.models.qwen2.modeling_qwen2 import mem_efficient_forward

    # lm.model.forward = mem_efficient_forward.__get__(lm.model, type(lm.model))

    for name, module in lm.named_modules():
        # If the name is model.layers.x.self_attn
        if name.startswith("model.layers") and name.endswith("self_attn"):
            module.register_forward_hook(get_attn(name))
            attn_weights[name] = []

    return attn_weights


def register_hooks_for_linear(lm):
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            assert len(input) == 1
            activations[name].append(input[0].detach().cpu())
            activations[name + "_out"].append(output.detach().cpu())

        return hook

    for name, module in lm.named_modules():
        if name.startswith("model.layers"):
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(get_activation(name))
                activations[name] = []
                activations[name + "_out"] = []
    return activations


def masked_assign(target: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert target.shape == source.shape == mask.shape, "All tensors must have the same shape"
    result = torch.where(mask, source, target)
    return result


def register_sparse_hooks(lm, args):
    def sparsify(name, threshold, num_frames, meta_info_path):
        def hook(module, input):
            in_dim = input[0].shape[2]
            input_tensor = input[0]  # Extract the actual tensor from the tuple
            if input_tensor.shape[1] == 1:
                return input

            with open(meta_info_path, "r") as f:
                # read the line according to the hook call count
                for i in range(module.hook_call_count):
                    f.readline()
                line = f.readline()
                import re

                matches = re.findall(r"\d+", line)
            module.hook_call_count += 1

            before_length = int(matches[0])
            after_length = int(matches[1])

            image_tokens = input_tensor[:, before_length:-after_length, :]
            assert image_tokens.shape[1] % (13 * 14) == 0, "The number of tokens must be a multiple of 13 * 14"
            tensor_cpy = image_tokens.clone()

            tensor_cpy = tensor_cpy.reshape(-1, 13 * 14, in_dim)
            total_frames = tensor_cpy.shape[0]
            num_ops = 0
            num_frames = total_frames  # Ensure this value is correctly set

            for i in range(total_frames // num_frames):
                num_ops += 13 * 14 * in_dim
                for j in range(num_frames - 1):
                    idx = i * num_frames + j
                    tensor_cur_frame = tensor_cpy[idx, :, :]
                    tensor_next_frame = tensor_cpy[idx + 1, :, :]
                    diff = tensor_cur_frame - tensor_next_frame
                    mask = torch.abs(diff * module.L1_norm) < threshold
                    num_ops += 13 * 14 * in_dim - torch.sum(mask).item()

                    tensor_cpy[idx + 1, :, :] = masked_assign(target=tensor_next_frame, source=tensor_cur_frame, mask=mask)

            density = num_ops / tensor_cpy.numel()
            tensor_cpy = tensor_cpy.reshape(1, -1, in_dim)

            # Correct file writing format
            with open(f"output/density_adpat_{threshold}.txt", "a") as f:
                f.write(f"{name}: {density}\n")

            # Return the modified tensor as a tuple (instead of modifying input[0] directly)
            modified_input = input_tensor.clone()
            modified_input[:, before_length:-after_length, :] = tensor_cpy
            return (modified_input,)  # Return as a tuple

        return hook

    num_data_points = "full"
    meta_info_path = f"output/llava_vid_{num_data_points}.txt"
    # this file must exist
    assert os.path.exists(meta_info_path), "Meta info file does not exist"

    for name, module in lm.named_modules():
        if name.startswith("model.layers"):
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(sparsify(name=name, threshold=0.1, num_frames=64, meta_info_path=meta_info_path))
                module.hook_call_count = 0

                wgt = module.weight.data.detach()
                # calculate mae along the last dimension
                L1_norm = torch.mean(torch.abs(wgt), dim=0)
                # assign it to module
                L1_norm = L1_norm / torch.max(L1_norm)
                module.L1_norm = L1_norm

    return
