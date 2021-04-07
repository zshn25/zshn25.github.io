---
layout: post
title:  "ResNet feature pyramid in Pytorch"
description: "Get feature pyramids from Pytorch's ResNet models"
image: https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Image_pyramid.svg/1200px-Image_pyramid.svg.png
date:   2021-02-09 15:21:23 -0700
categories: deep-learning computer-vision pytorch 
author: Zeeshan Khan Suri
published: false
comments: true
---


<!-- {::options parse_block_html="true" /} -->

[![image pyramid](https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Image_pyramid.svg/1200px-Image_pyramid.svg.png)](https://commons.wikimedia.org/wiki/File:Image_pyramid.svg#/media/File:Image_pyramid.svg)
{:refdef: style="text-align: center;font-size: .8rem;font-style: italic;color: light-grey;"}
Image Pyramid. By [Cmglee](https://commons.wikimedia.org/wiki/User:Cmglee), [CC BY-SA 3.0](https://commons.wikimedia.org/w/index.php?curid=42549151)
{: refdef}

Feature Pyramids are features at different resolutions. Since Neural Networks compute features at various levels, (for e.g. the earliest layers of a CNN produce low level features such as Edges and later layers produce higher level features) it would be great to use not only the higher level features but also the previous ones for further processing.

One of the application of such feature pyramid is to be used in an autoencoder architecture with skip connections from encoder to decoder like U-Net.

[![u-net](https://upload.wikimedia.org/wikipedia/commons/2/2b/Example_architecture_of_U-Net_for_producing_k_256-by-256_image_masks_for_a_256-by-256_RGB_image.png)](https://en.wikipedia.org/wiki/U-Net#/media/File:Example_architecture_of_U-Net_for_producing_k_256-by-256_image_masks_for_a_256-by-256_RGB_image.png)
{:refdef: style="text-align: center;font-size: .8rem;font-style: italic;color: light-grey;"}
U-Net uses it's encoder's feature pyramid as skip connections to the decoder. Image by [Mehrdad Yazdani](https://commons.wikimedia.org/w/index.php?title=User:Crude2refined&action=edit&redlink=1) [CC BY-SA 4.0 ](https://commons.wikimedia.org/w/index.php?curid=81055729)
{:refdef }

In this example, we look at [ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) from [Pytorch](https://pytorch.org/). ResNet is one of the earliest but also one of the best performing network architectures for various tasks. We inherit the ResNet class and write our own forward method to output a pyramid of feature maps instead.

```python
class ResNetFeatures(ResNet):
    def __init__(self,**kwargs):
        super(ResNetFeatures,self).__init__(**kwargs)
        
    def _forward_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return [x0,x1,x2,x3,x4] # returns features with channels [64,64,128,256,512]
```

That's it. Now, to use it, add it to the source of ResNet. I have made a Gist on how to do this [here](https://gist.githubusercontent.com/zshn25/09683dadb6b4e60e8382a380020d144e/raw/f752c2f214913d7e35f27da79c425fc7f753928c/resnet.py). Now you can give an argument `features_only` and it will return the feature pyramid as defined above.

---

## Appendix

For the sake of completion, I include the Gist here. To test it online, click [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gist/zshn25/09683dadb6b4e60e8382a380020d144e/HEAD)

<script src="https://gist.github.com/zshn25/09683dadb6b4e60e8382a380020d144e.js"></script>

---
