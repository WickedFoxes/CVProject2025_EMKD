from .ENet import ENet
from .RAUNet import RAUNet
from .dinov3_vision_transformer import build_dinov3_base_primus_multiscale_with_new_patch_size


def get_model(model_name: str, channels: int, **kwargs):
    assert model_name.lower() in ['enet', 'raunet']
    if model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(num_classes=channels)
    elif model.name.lower() == 'dinov3':
        model = build_dinov3_base_primus_multiscale_with_new_patch_size(
            num_classes=channels,
            checkpoint_path = kwargs.checkpoint_path,
            new_patch_size=28,
        )
    return model