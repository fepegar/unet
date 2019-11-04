from typing import Optional

import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            dimensions: int,
            in_channels: int,
            out_channels: int,
            normalization: Optional[str] = None,
            kernel_size: int = 3,
            activation: Optional[str] = 'ReLU',
            preactivation: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__()

        block = nn.ModuleList()

        dilation = 1 if dilation is None else dilation
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        class_name = 'Conv{}d'.format(dimensions)
        conv_class = getattr(nn, class_name)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
        )

        norm_layer = None
        if normalization is not None:
            class_name = '{}Norm{}d'.format(
                normalization.capitalize(), dimensions)
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = 'Dropout{}d'.format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)
