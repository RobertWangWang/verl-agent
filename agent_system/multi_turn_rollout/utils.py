import torch
import numpy as np
import random
from typing import List
import math
from PIL import Image
from verl import DataProto

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto) -> DataProto:
    size_divisor_ref = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    if bs % size_divisor != 0:
        remainder = bs % size_divisor
        print(f"Current batch size: {bs} cannot be divided by {size_divisor}, randomly deleting {remainder} samples")
        
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    else:
        adjusted_batch = data

    return adjusted_batch
