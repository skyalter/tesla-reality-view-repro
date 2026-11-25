"""Image normalisation and the camera convention shared by the models.

Convention (the same as the stage-1 data reader, ``realityview/data/dataset.py``):

- ``ref_T_cam`` (..., 4, 4): camera-to-reference transform. Camera axes are OpenCV (x right, y down,
  z forward); the reference frame is one chosen camera (the front-center camera at the current moment, so
  its ``ref_T_cam`` is the identity). Translations in metres.
- ``K`` (..., 3, 3): pinhole intrinsics in pixels of the image actually fed to the network.

DA3's camera encoder (``da3/cam_enc.py``) takes **world-to-camera** extrinsics in OpenCV convention
(DA3 README: "opencv w2c"), inverts them to camera-to-world and encodes
``[t (3), quaternion x y z w (4, scalar last), fov_h, fov_w]``. ``da3_extrinsics`` turns our
``ref_T_cam`` into that input: translation divided by ``pose_scale`` (metres per DA3 unit), then inverted.
"""
from __future__ import annotations

import torch

from .da3.build import IMAGENET_MEAN, IMAGENET_STD
from .da3.utils.geometry import affine_inverse
from .da3.utils.transform import extri_intri_to_pose_encoding

DEFAULT_POSE_SCALE = 20.0


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """RGB images (..., 3, H, W), uint8 0..255 or float 0..1, to ImageNet-normalised float32."""
    x = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    mean = x.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def da3_extrinsics(ref_T_cam: torch.Tensor, pose_scale: float = DEFAULT_POSE_SCALE) -> torch.Tensor:
    """World-to-camera 4x4 matrices in DA3 units from camera-to-reference poses in metres (float32)."""
    if ref_T_cam.shape[-2:] != (4, 4):
        raise ValueError(f"ref_T_cam must be (..., 4, 4), got {tuple(ref_T_cam.shape)}")
    c2w = ref_T_cam.float().clone()
    c2w[..., :3, 3] = c2w[..., :3, 3] / pose_scale
    return affine_inverse(c2w)


def da3_pose_encoding(ref_T_cam: torch.Tensor, K: torch.Tensor, image_hw, pose_scale=DEFAULT_POSE_SCALE):
    """The 9-number vector DA3's camera encoder computes from our poses: [t/scale, q_xyzw, fov_h, fov_w].

    Exactly what ``CameraEnc.forward`` does to the output of ``da3_extrinsics``; used to pin the convention.
    """
    c2w = affine_inverse(da3_extrinsics(ref_T_cam, pose_scale))
    return extri_intri_to_pose_encoding(c2w, K.float(), image_hw)
