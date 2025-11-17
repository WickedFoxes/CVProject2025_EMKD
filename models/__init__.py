from .ENet import ENet
from .RAUNet import RAUNet

def get_model(model_name: str, channels: int):
    assert model_name.lower() in ['deeplabv3+', 'enet', 'erfnet', 'espnet', 'mobilenetv2',
                             'unet++', 'raunet', 'resnet18', 'unet', 'pspnet']
    if model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(num_classes=channels)
    return model