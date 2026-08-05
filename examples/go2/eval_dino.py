"""
DINO v2. https://github.com/facebookresearch/dinov2
"""
import gc
from pathlib import Path
from typing import List, Tuple, Union

import torch
# from einops import rearrange
from params_proto import PrefixProto, Proto
from PIL import Image
from torchvision import transforms
# from torchvision import transforms as T
from tqdm import tqdm

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_dino_v2_model_types = ["vits14", "vitsb14", "vitl14", "vitg14"]


class DINOv2Args(PrefixProto):
    # model_type: str = Proto("vits14", help="DINO v2 model to use.")
    model_type_l: str = Proto("vitl14", help="DINO v2 model to use.")
    model_type_g: str = Proto("vitg14", help="DINO v2 model to use.")
    batch_size: int = Proto(32, help="Batch size for DINO v2.")
    # batch_size: int = Proto(128, help="Batch size for DINO v2.")
    # batch_size: int = Proto(512, help="Batch size for DINO v2.")


def _validate_dino_v2_args():
    assert (
        DINOv2Args.model_type in _dino_v2_model_types
    ), f"Model type cannot be {DINOv2Args.model_type}, must be one of {_dino_v2_model_types}"


def load_dino_v2_model(model_type="vitg14") -> torch.nn.Module:
    if model_type == "vitg14":
        model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{DINOv2Args.model_type_g}")
    elif model_type == "vitl14":
        model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{DINOv2Args.model_type_l}")
    else:
        raise NotImplementedError
    model.eval()

    model.patch_embed.forward = patch_embed_forward.__get__(model.patch_embed)
    return model


def patch_embed_forward(patch_embed, x):
    """
    Modified from dinov2.layers.PatchEmbed.forward to remove the patch size assert.
    Used so we can handle arbitrary image resolutions.

    We monkey patch the model.patch_embed.forward method to this function.
    """
    _, _, H, W = x.shape

    x = patch_embed.proj(x)  # B C H W
    H, W = x.size(2), x.size(3)
    x = x.flatten(2).transpose(1, 2)  # B HW C
    x = patch_embed.norm(x)
    if not patch_embed.flatten_embedding:
        x = x.reshape(-1, H, W, patch_embed.embed_dim)  # B H W C
    return x


def get_dino_v2_preprocess(
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    center_crop: bool = False,
) -> transforms.Compose:
    """Modified from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L78"""
    if center_crop:
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
        ]
    else:
        # Directly resize to crop size
        transforms_list = [
            transforms.Resize(crop_size, interpolation=interpolation),
        ]

    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def extract_dino_v2_features(images_pil, device: torch.device, model_size='vitg14') -> torch.Tensor:
    """Extract DINO patch-level features for given images"""

    preprocess = get_dino_v2_preprocess()
    tf = transforms.ToPILImage()
    images = torch.stack([preprocess(tf(image.permute(2, 0, 1))) for image in images_pil], dim=0).to(device)
    # with logger.time("load_dino_v2_model"):
    model = load_dino_v2_model().to(device)
    # Monkey patch the patch embedding forward to remove patch size assert, so we can handle arbitrary image sizes
    model.patch_embed.forward = patch_embed_forward.__get__(model.patch_embed)

    with torch.no_grad(): # , logger.time("dino_v2_forward_features")
        # Split into batches so we don't run OOM
        descriptors = []
        progress_bar = tqdm(total=len(images), desc="Extracting DINO v2 features")
        for i in range(0, len(images), DINOv2Args.batch_size):
            image_batch = images[i : i + DINOv2Args.batch_size].to(device)
            # descriptors.append(model.forward_features(image_batch)["x_norm_patchtokens"].detach().cpu())
            descriptors.append(model.forward_features(image_batch)["x_norm_clstoken"].detach().cpu())
            progress_bar.update(image_batch.shape[0])
        descriptors = torch.cat(descriptors, dim=0)

    return descriptors