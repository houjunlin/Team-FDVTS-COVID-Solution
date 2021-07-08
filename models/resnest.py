import torch
# from .resnet import ResNet, Bottleneck

from models.resnet_xjl import ResNet, Bottleneck, ResNet3D, Bottleneck3D


__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269', 'resnest50_3D', 'resnest101_3D', 'resnest200_3D', 'resnest269_3D']
#from .build import RESNEST_MODELS_REGISTRY

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

#@RESNEST_MODELS_REGISTRY.register()
def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

#@RESNEST_MODELS_REGISTRY.register()
def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

#@RESNEST_MODELS_REGISTRY.register()
def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

#@RESNEST_MODELS_REGISTRY.register()
def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model



#@RESNEST_MODELS_REGISTRY.register()
def resnest50_3D(pretrained=True, root='~/.encoding/models', **kwargs):
    new_num_classes = kwargs['num_classes']
    kwargs['num_classes'] = 1000
    model = ResNet(Bottleneck, [3, 4, 6, 3], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        # print('Loading ckpt')
        # print(model.conv1[0].weight[0])
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
        # print('after loading')
        # print(model.conv1[0].weight[0])

    kwargs['num_classes'] = new_num_classes
    model_3d = ResNet3D(model, Bottleneck3D, [3, 4, 6, 3], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model_3d

#@RESNEST_MODELS_REGISTRY.register()
def resnest101_3D(pretrained=False, root='~/.encoding/models', **kwargs):
    new_num_classes = kwargs['num_classes']
    kwargs['num_classes'] = 1000
    model = ResNet(Bottleneck, [3, 4, 23, 3], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))

    kwargs['num_classes'] = new_num_classes
    model_3d = ResNet(model, Bottleneck3D, [3, 4, 23, 3], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model_3d

#@RESNEST_MODELS_REGISTRY.register()
def resnest200_3D(pretrained=False, root='~/.encoding/models', **kwargs):
    new_num_classes = kwargs['num_classes']
    kwargs['num_classes'] = 1000
    model = ResNet(Bottleneck, [3, 24, 36, 3], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    
    kwargs['num_classes'] = new_num_classes
    model_3d = ResNet(model, Bottleneck3D, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model_3d

#@RESNEST_MODELS_REGISTRY.register()
def resnest269_3D(pretrained=False, root='~/.encoding/models', **kwargs):
    new_num_classes = kwargs['num_classes']
    kwargs['num_classes'] = 1000
    model = ResNet(Bottleneck, [3, 30, 48, 8], 
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))

    kwargs['num_classes'] = new_num_classes
    model_3d = ResNet(model, Bottleneck3D, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model_3d


# net = resnest50_3D(pretrained=True)
# print('Show 3D model')
# ipt = torch.randn(2, 1, 16, 64, 64)
# out = net(ipt)
# print(out.size())

# net = resnest50()
# ipt = torch.randn(2, 3, 64, 64)
# out = net(ipt)
# print(out.size())