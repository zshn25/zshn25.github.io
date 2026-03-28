---
layout: notebook
title: "Layers fusion for faster neural network inference"
description: "How to fuse all convolution and batch normalization layers in a Pytorch model during test time for faster inference"
image: images/neural-network-3816319.svg
date: 2021-04-27 15:21:23 -0700
categories: deep-learning pytorch runtime-optimization
author: Zeeshan Khan Suri
published: true
comments: true
nb_path: _notebooks/2021-04-27-Layers-fusion-for-faster-inference.ipynb
---

In the [previous post]({% post_url 2021-03-11-is-convolution-linear %}), we proved that convolutions are linear. There are other linear layers in a neural network such as a batch normalization layer. A batch normalization layer [normalizes](https://en.wikipedia.org/wiki/Normalization_(statistics)) its input batch to have zero mean and unit standard deviation, which are calculated from the input batch.{% fn 1 %} It basically translates/shifts and scales the input batch, thus being a linear operation. In many network architectures such as ResNets{% fn 2 %} and DenseNets{% fn 3 %}, a convolutional layer followed by a batch norm layer is used. 

During training, the mean and standard deviation of the input batch are used in the batch normalization and are eventually learnt. During inference, these estimates of mean and standard deviation are used instead. The idea of this post is to fuse these two consecutive layers during inference, thereby reducing computation and thus inference time.

Note that this must not be done during training since the input batch's mean and standard deviation are not yet learnt and fusing before training will be same as removing the batch normalization completely.

Pytorch provides a utility [function to fuse convolution and batch norm](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py), although this was meant for the use of quantization. In this post, I share the following function to recursively check and fuse all consecutive convolution and batch norm layers.


```python
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def fuse_all_conv_bn(model):
    """
    Fuses all consecutive Conv2d and BatchNorm2d layers.
    License: Copyright Zeeshan Khan Suri, CC BY-NC 4.0
    """
    stack = []
    for name, module in model.named_children(): # immediate children
        if list(module.named_children()): # is not empty (not a leaf)
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))
```

## Test

Fusing all convolution and batch norm layers of ResNet101 makes the resulting model **~25% faster** with negligible difference in the model's output. 


<details class="cell-collapse" open markdown="1">
<summary>Hide code</summary>

```python
import torch
from torchvision.models.resnet import resnet101

model=resnet101(pretrained=True).to('cuda')
model.eval()
rand_input = torch.randn((1,3,256,256)).to('cuda')

# Forward pass
output = model(rand_input)
print("Inference time before fusion:")
%timeit model(rand_input)

# Fuse Conv BN
fuse_all_conv_bn(model)
print("\nInference time after fusion:")
%timeit model(rand_input)
# compare result
print("\nError between outputs before and after fusion:")
torch.norm(torch.abs(output - model(rand_input))).data
```

</details>
    Inference time before fusion:
    43.4 ms ± 17.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
    Inference time after fusion:
    31.2 ms ± 7.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
    Error between outputs before and after fusion:
    




    tensor(3.5947e-05, device='cuda:0')



Note that the same can be done with any network with 2 or more consecutive linear layers to reduce inference time.

© Zeeshan Khan Suri, [<i class="fab fa-creative-commons"></i> <i class="fab fa-creative-commons-by"></i> <i class="fab fa-creative-commons-nc"></i>](http://creativecommons.org/licenses/by-nc/4.0/)

If this article was helpful to you, consider citing


```python
@misc{suri_Layers-fusion-for-faster-inference_2021,
      title={Layers fusion for faster neural network inference},
      url={https://zshn25.github.io/Layers-fusion-for-faster-inference/}, 
      journal={Curiosity}, 
      author={Suri, Zeeshan Khan}, 
      year={2021}, 
      month={Apr}}
```

## References
{{ 'Ioffe, S. & Szegedy, C.. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 37:448-456 [Link](http://proceedings.mlr.press/v37/ioffe15.html).' | fndetail: 1 }}
{{ 'K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778, doi: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). ' | fndetail: 2 }}
{{ 'G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2261-2269, doi: [10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243). ' | fndetail: 3 }}
