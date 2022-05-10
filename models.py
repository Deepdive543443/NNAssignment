import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn = False, activation=nn.ReLU(), init=None):
        '''
        Configurable convolutional block

        :param in_channels: in channels of conv layer
        :param out_channels: number of filter of convolutional layers
        :param kernel_size: kernel size of conv layers
        :param stride: stride of convolutional layers
        :param padding: padding of coonvolutional layers
        :param use_bn: add a batchnorm layers after conv layers if True
        :param activation: the activation function at the end of block
        :param init: type intialization of conv layers
        '''
        super().__init__()
        layers = []
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False if use_bn else True)
        if init == "He":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "Xavier":
            nn.init.xavier_normal_(self.conv.weight)
        layers.append(self.conv)
        layers.append(nn.BatchNorm2d(out_channels)) if use_bn else None
        layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Convsfig(nn.Module):
    def __init__(self,configs, pooling="max", dropout=None, **kwargs):
        super().__init__()
        '''
        A fully configurable convolutional neural network
        Config written in [[f, k, s, p],(k, s, p),numclass], list as convolutional layers, tuple as padding, int as Lazy fully connected
        '''
        layers = []
        img_channels = 3
        for config in configs:
            if isinstance(config,list):
                layers.append(ConvBlock(img_channels, config[0], kernel_size=config[1], stride=config[2], padding=config[3],**kwargs))
                img_channels = config[0]
            elif isinstance(config,tuple):
                layers.append(nn.MaxPool2d(config[0], config[1], config[2]) if pooling == "max" else nn.AvgPool2d(config[0], config[1], config[2]))
            elif isinstance(config,int):
                layers.append(nn.Flatten())
                layers.append(nn.Dropout(dropout)) if dropout else None
                layers.append(nn.LazyLinear(config))
            elif isinstance(config,str):
                if config == "avg_pool":
                    layers.append(nn.AdaptiveAvgPool2d((1,1)))
                layers.append(nn.Flatten())
            else:
                assert "Incorrect configuration"
        self.seq = nn.Sequential(*layers)

    def forward(self,x):
        return self.seq(x)