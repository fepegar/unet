import torch

from unet import UNet1D, UNet2D, UNet3D

residual = False

torch.manual_seed(0)
torch.set_grad_enabled(False)


def run(model, shape):
    x_sample = torch.rand(*shape)
    with torch.no_grad():
        y = model(x_sample)
    return y


def test_unet_1d():
    model = UNet1D(
        normalization="batch",
        preactivation=True,
        residual=False,
    ).eval()
    shape = 1, 1, 572
    result = 1, 2, 388
    y = run(model, shape)
    assert tuple(y.shape) == result


def test_unet_1d_residual():
    model = UNet1D(
        normalization="batch",
        preactivation=True,
        residual=True,
    ).eval()
    shape = 1, 1, 512
    result = 1, 2, 512
    y = run(model, shape)
    assert tuple(y.shape) == result


def test_unet_2d():
    model = UNet2D(
        normalization="batch",
        preactivation=True,
        residual=False,
    ).eval()
    shape = 1, 1, 572, 572
    result = 1, 2, 388, 388
    y = run(model, shape)
    assert tuple(y.shape) == result


def test_unet_2d_residual():
    model = UNet2D(
        normalization="batch",
        preactivation=True,
        residual=True,
    ).eval()
    shape = 1, 1, 512, 512
    result = 1, 2, 512, 512
    y = run(model, shape)
    assert tuple(y.shape) == result


def test_unet_3d():
    model = UNet3D(
        normalization="batch",
        preactivation=True,
        residual=False,
    ).eval()
    shape = 1, 1, 132, 132, 116
    result = 1, 2, 44, 44, 28
    y = run(model, shape)
    assert tuple(y.shape) == result


def test_unet_3d_residual():
    model = UNet3D(
        normalization="batch",
        preactivation=True,
        residual=True,
        num_encoding_blocks=2,
        upsampling_type="trilinear",
    ).eval()
    shape = 1, 1, 64, 64, 56
    result = 1, 2, 64, 64, 56
    y = run(model, shape)
    assert tuple(y.shape) == result
