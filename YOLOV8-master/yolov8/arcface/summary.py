#--------------------------------------------#
#   This section of code is only for viewing the network structure and is not for testing.
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.arcface import Arcface

if __name__ == "__main__":
    input_shape = [112, 112]
    backbone = 'mobilefacenet'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Arcface(num_classes=10575, backbone=backbone, mode="predict").to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   `flops * 2` is used because `profile` does not treat convolution
    #   as two operations (multiplication and addition).
    #   Some papers count convolution as both multiplication and addition operations. In such cases, multiply by 2.
    #   Other papers only consider the number of multiplications and ignore additions. In such cases, do not multiply by 2.
    #   This code chooses to multiply by 2, following the YOLOX approach.
    #--------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
