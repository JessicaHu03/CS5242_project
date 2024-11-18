#--------------------------------------------#
#   This section of code is used to inspect the network structure.
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    phi             = 's'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(input_shape, num_classes, phi, False).to(device)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   `flops * 2` is used because `profile` does not count 
    #   convolution as two operations.
    #   Some papers count both multiplication and addition in 
    #   convolutions as two separate operations (in which case, multiply by 2).
    #   Some papers only consider the multiplication operations 
    #   and ignore addition (in which case, do not multiply by 2).
    #   This code chooses to multiply by 2, as referenced in YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
