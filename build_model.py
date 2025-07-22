import torch
import torch.nn as nn
from Models.SparseSwin import SparseSwin
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights


def buildSparseSwin(image_resolution, swin_type, num_classes, 
                    ltoken_num, ltoken_dims, num_heads, 
                    qkv_bias, lf, attn_drop_prob, lin_drop_prob, 
                    freeze_12, device):
    """
    image_resolution : input image resolution (h x w x 3), input MUST be a squared image and divisible by 16
    swin_type : Swin Transformer model type Tiny, Small, Base 
    num_classes : number of classes 
    """
    dims = {
        'tiny': 96, 
        'small': 96,
        'base': 128
    }
    dim_init = dims.get(swin_type.lower())
    
    if (dim_init == None) or ((image_resolution%16) != 0):
        print('Check your swin type OR your image resolutions are not divisible by 16')
        print('Remember.. it must be a squared image')
        return None 
    
    model = SparseSwin(
        swin_type=swin_type, 
        num_classes=num_classes, 
        c_dim_3rd=dim_init*4, 
        hw_size_3rd=int(image_resolution/16), 
        ltoken_num=ltoken_num, 
        ltoken_dims=ltoken_dims, 
        num_heads=num_heads, 
        qkv_bias=qkv_bias, 
        lf=lf, 
        attn_drop_prob=attn_drop_prob, 
        lin_drop_prob=lin_drop_prob, 
        freeze_12=freeze_12,
        device=device, 
    ).to(device)
    
    return model 


# Now add this plain Swin builder:

def buildPlainSwin(image_resolution=224, swin_type='tiny', num_classes=10, freeze_12=False, device='cuda'):
    swin_type = swin_type.lower()
    if swin_type == 'tiny':
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    elif swin_type == 'small':
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
    elif swin_type == 'base':
        model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Invalid swin_type")

    # freeze first 2 stages if required
    if freeze_12:
        for param in model.features[:4].parameters():
            param.requires_grad = False

    # replace the head for CIFAR classes
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model.to(device)




if __name__ == '__main__': 
    swin_type = 'tiny'
    device = 'cuda'
    image_resolution = 224
    
    model = buildSparseSwin(
        image_resolution=image_resolution,
        swin_type=swin_type, 
        num_classes=100, 
        ltoken_num=49, 
        ltoken_dims=512, 
        num_heads=16, 
        qkv_bias=True,
        lf=2, 
        attn_drop_prob=.0, 
        lin_drop_prob=.0, 
        freeze_12=False,
        device=device
    )
    
