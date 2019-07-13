import sys
from pathlib import Path

import torch

filepath = Path(__file__).absolute()
tests_dir = filepath.parent
unet_dir = tests_dir.parent
sys.path.append(str(unet_dir))
from unet import UNet2D, UNet3D

residual = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(model, shape):
    x_sample = torch.rand(*shape, device=device)
    with torch.no_grad():
        y = model(x_sample)
    return y

def test_unet_2d():
    model = UNet2D(
        normalization='batch',
        preactivation=True,
        residual=False,
    ).to(device).eval()
    shape = 1, 1, 572, 572
    result = 1, 2, 388, 388
    y = run(model, shape)
    assert tuple(y.shape) == result

def test_unet_2d_residual():
    model = UNet2D(
        normalization='batch',
        preactivation=True,
        residual=True,
    ).to(device).eval()
    shape = 1, 1, 512, 512
    result = 1, 2, 512, 512
    y = run(model, shape)
    assert tuple(y.shape) == result

def test_unet_3d():
    model = UNet3D(
        normalization='batch',
        preactivation=True,
        residual=False,
    ).to(device).eval()
    shape = 1, 1, 132, 132, 116
    result = 1, 2, 44, 44, 28
    y = run(model, shape)
    assert tuple(y.shape) == result

def test_unet_3d_residual():
    model = UNet3D(
        normalization='batch',
        preactivation=True,
        residual=True,
        num_encoding_blocks=2,
        upsampling_type='trilinear',
    ).to(device).eval()
    shape = 1, 1, 64, 64, 56
    result = 1, 2, 64, 64, 56
    y = run(model, shape)
    assert tuple(y.shape) == result
