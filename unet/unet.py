from typing import Optional, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNELS_DIMENSION = 1


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_classes=2,
        dimensions=2,
        num_encoding_blocks=5,
        out_channels_first_layer=64,
        valid=True,
        normalization=None,
        pooling_type='max',
        upsampling_type='conv',
        ):
        super().__init__()
        depth = num_encoding_blocks - 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if dimensions == 2:
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels
        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
        )

        # Decoder
        if dimensions == 2:
            m = depth - 1
        elif dimensions == 3:
            m = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**m
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels,
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization,
        )

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = get_conv_block(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=None,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        return self.classifier(x)


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 2
        kwargs['num_encoding_blocks'] = 5
        kwargs['out_channels_first_layer'] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 4
        kwargs['out_channels_first_layer'] = 32
        kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        pooling_type: str,
        num_encoding_blocks: int,
        normalization: Optional[str],
        ):
        super().__init__()
        self.encoding_blocks = nn.ModuleList()
        for _ in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
            )
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = 2 * out_channels_first
                out_channels_first = in_channels

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        b = self.encoding_blocks[-1]
        return self.encoding_blocks[-1].out_channels


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        num_decoding_blocks: int,
        normalization: Optional[str],
        ):
        super().__init__()
        self.decoding_blocks = nn.ModuleList()
        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        normalization: Optional[str],
        pooling_type: Optional[str],
        ):
        super().__init__()
        self.conv1 = get_conv_block(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
        )
        if dimensions == 2:
            out_channels_second = out_channels_first
        elif dimensions == 3:
            out_channels_second = 2 * out_channels_first
        self.conv2 = get_conv_block(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=normalization,
        )
        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2[0].out_channels


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        normalization: Optional[str],
        ):
        super().__init__()
        if upsampling_type == 'conv':
            in_channels = out_channels = 2 * in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, out_channels)
        else:
            self.upsample = get_upsampling_layer(dimensions, upsampling_type)
        in_channels = in_channels_skip_connection * (1 + 2)
        out_channels = in_channels_skip_connection
        self.conv1 = get_conv_block(
            dimensions, in_channels, out_channels, normalization=normalization)
        in_channels = out_channels
        self.conv2 = get_conv_block(
            dimensions, in_channels, out_channels, normalization=normalization)

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection


def get_conv_block(
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 3,
        activation: str = 'ReLU',
        ):
    block = nn.ModuleList()
    conv_class = getattr(nn, f'Conv{dimensions}d')
    conv_layer = conv_class(in_channels, out_channels, kernel_size)
    block.append(conv_layer)
    if normalization is not None:
        norm_class = getattr(
            nn, f'{normalization.capitalize()}Norm{dimensions}d')
        norm_layer = norm_class(out_channels)
        block.append(norm_layer)
    if activation is not None:
        activation_layer = getattr(nn, activation)()
        block.append(activation_layer)
    return nn.Sequential(*block)

def get_downsampling_layer(
        dimensions: int,
        pooling_type: str,
        kernel_size: int = 2,
    ) -> nn.Module:
    class_ = getattr(nn, f'{pooling_type.capitalize()}Pool{dimensions}d')
    return class_(kernel_size)

def get_upsampling_layer(
        dimensions: int,
        upsampling_type: str,
    ) -> Union[nn.Module, Callable]:
    return lambda x: F.interpolate(x, scale_factor=2, mode=upsampling_type)

def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    conv_class = getattr(nn, f'ConvTranspose{dimensions}d')
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def main():
    model: UNet

    model = UNet2D()
    print(model)
    x_sample = torch.rand(1, 1, 572, 572)
    with torch.no_grad():
        y = model(x_sample)
    print(y.shape)

    model = UNet3D()
    print(model)
    x_sample = torch.rand(1, 1, 132, 132, 116)
    with torch.no_grad():
        y = model(x_sample)
    print(y.shape)


if __name__ == "__main__":
    main()
