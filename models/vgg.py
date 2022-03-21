import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, non_linearity, bias, num_classes=1000, init_weights=True, **non_linearity_kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=bias),
            non_linearity(num_channels=4096, **non_linearity_kwargs),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=bias),
            non_linearity(num_channels=4096, **non_linearity_kwargs),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, non_linearity, bias, batch_norm=False, **non_linearity_kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias and not batch_norm)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=bias), non_linearity(num_channels=v, **non_linearity_kwargs)]
            else:
                layers += [conv2d, non_linearity(num_channels=v, **non_linearity_kwargs)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['A'], non_linearity, bias, **non_linearity_kwargs),
                non_linearity, bias, init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 11-layer model (configuration "A") with batch_normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['A'], non_linearity, bias, batch_norm=True, **non_linearity_kwargs),
                non_linearity, bias,  init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['B'], non_linearity, bias, **non_linearity_kwargs),
                non_linearity, bias, init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 13-layer model (configuration "B") with batch_normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['B'], non_linearity, bias, batch_norm=True, **non_linearity_kwargs),
                non_linearity, bias,  init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['D'], non_linearity, bias, **non_linearity_kwargs),
                non_linearity, bias, init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 16-layer model (configuration "D") with batch_normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['D'], non_linearity, bias, batch_norm=True, **non_linearity_kwargs),
                non_linearity, bias,  init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 19-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['E'], non_linearity, bias, **non_linearity_kwargs),
                non_linearity, bias, init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(non_linearity, bias, init_weights=True, pretrained=False, **non_linearity_kwargs):
    """VGG 19-layer model (configuration "E") with batch_normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        init_weights = False
    model = VGG(make_layers(cfg['E'], non_linearity, bias, batch_norm=True, **non_linearity_kwargs),
                non_linearity, bias,  init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
