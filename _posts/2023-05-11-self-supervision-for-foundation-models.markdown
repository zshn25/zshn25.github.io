---
layout: post
title: "Self-supervised learning for vision foundation models"
image: images/foundation.jpeg
date: 2023-05-11 20:21:23 -0700
categories: computer-vision large-neural-networks foundation-models self-supervision
author: Zeeshan Khan Suri
published: true
comments: true
---

> ## [Performance of deep neural networks scales with data and model size]({{ site.baseurl }}{% post_url 2022-12-1-Software-2 %})

Recent developments in deep learning can be attributed to the growing amount of data and model size. This realization has lead to the rapidly developing field of **foundational models**, i.e. large scale, complex neural networks, for e.g. [GPT 4](https://openai.com/research/gpt-4), [CLIP](https://openai.com/research/clip), [ImageBind](https://facebookresearch.github.io/ImageBind/paper), etc., trained on very large, diverse datasets. By feeding-in huge amounts of high quality labeled data through large artificial neural network models, machines are shown to be taught to outperform humans at multiple tasks. 

{:refdef: style="text-align: center;"}
![s]({{site.baseurl}}/images/performance_data.svg) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*As the training data increases, the performance of a DL model increases but traditional methods do not depend on the training data*
</sup></sub>
{: refdef}

But, collecting such massive amounts of carefully labeled data costs enormous time, effort and money; and are error prone. For e.g. Northcutt et al.[^1] outline errors in the labels of widely used datasets. What if there was a way to use massively available but **unlabled** data?

In this post, I describe the necessity for foundation models as feature representors and in what ways self-supervised learning enables this achievement. Stay tuned!

## History: Feature representations of images

Tradition computer vision methods extract distinctive pixel-level features from input images. High-dimensional representations of these features, known as **feature descriptors** are computed from the detected local features which are then used for various applications, such as tracking, retrieval, structure from motion, localization, etc. The goal of transforming the detected features into high-dimensional descriptors is to embed some notion of semantic and neighboring information to the pixel-level features. The goal is to make them distinctive (good for matching across different viewpoints) and invariant to vairations in 3D viewpoint, illumination, etc.


### Features of good features

{:refdef: style="text-align: center;"}
![sift](https://github.com/cvg/Hierarchical-Localization/raw/master/doc/loc_aachen.svg) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>Good features must be robust against 3D viewpoint, illumination, scaling, etc. Image from [Aachen Day-Night Dataset](https://www.visuallocalization.net/datasets/) [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)
</sup></sub>
{: refdef}

To work robustly in real world scenarios, these feature descriptors must be invariant against view-point changes (rotation and translation of camera in 3D), scaling, illumination changes, and other adverse conditions. As seen in the above example image, such features of good features make matching under varying conditions possible, making them highly useful.

## Learning features

With the rise of deep learning, features and their descriptors are no longer hand-engineered, but learnt from the underlying data. Vision models such as CNNs and vision transformers process input images to extract high-dimensional latent representations, which could be thought as features. Unlike pixel-level features such as SIFT, ORB, latents are **not local** (do not correspond to pixel locations) but **based on the semantics** of the global image contents.

> Checkout my blog post on [how to extract feature pyramids from torchvison models]({{ site.baseurl }}{% post_url 2021-02-01-ResNet-feature-pyramid-in-Pytorch %})

## The need for foundation models

If you're thinking, if existing models already extract features, why is there a need for foundation models?

You are absolutely right. Existing models such as YOLO[^3] object detection, extract features, which are further passed to the object-detection head. But, there are two reasons where foundation models come into play.

1. The features extracted from Yolo are **task-specific**. Features that are useful for object-detection, might not be useful or optimal for other tasks, for example, for depth estimation. We want our ideal features to be **task-agnostic**. The reasoning behind this is the underlying assumption that there exist features that are useful for all kinds of tasks, just as we saw in the traditional feature detectors and descriptors like SIFT.
2. We want to train on lots of data. But, supervised tasks such as object detection require expensive labeled data, and thus are infeasible to scale. 

{:refdef: style="text-align: center;"}
![multi-task]({{site.baseurl}}/images/multi-task.svg) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>The features of a network trained on a pretext task, for e.g., depth estimation, might not necessarily be useful or optimal for other tasks, for e.g., classification, as depicted in green. Cat image by [EVG Kowalievska](https://www.pexels.com/photo/selective-focus-photography-of-orange-tabby-cat-1170986/).
</sup></sub>
{: refdef}

# Self-supervised learning

Animals, including humans, learn a many concepts such as object permanance, etc., without explicit supervision. 

{% include youtube.html content="https://www.youtube.com/embed/OLrYzY3jVPY" %}{: width="100%" .shadow}
{:refdef: style="text-align: center;"}
<sub><sup>
</sup></sub>
{: refdef}

Is it possible to make machines learn general concepts without explicit supervision? If so, how?

If we are able formulate **pretext tasks**, we can use <mark>self-supervised learning to learn good data representations and general concepts from massively-available, unlabelled data</mark>, we can use such models directly (without fine-tuning, Zero-Shot) for multiple **downstream tasks**. (atleast, that's the hope).

{:refdef: style="text-align: center;"}
[![feature-learning](https://upload.wikimedia.org/wikipedia/commons/0/0b/Feature_Learning_Diagram.png)](https://en.wikipedia.org/wiki/Feature_learning#/media/File:Feature_Learning_Diagram.png)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>Implicit feature representations are learned via pretext tasks and can be used as input for specific downstream tasks. Image from [Fgpacini](https://commons.wikimedia.org/wiki/File:Feature_Learning_Diagram.png), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*
</sup></sub>
{: refdef}

## Pretext tasks in computer vision

Prediction/Generation and Similarity are the two main concepts that can be exploited to design self-supervised pretext tasks. 

### Generative pretext tasks

These methods involves artificially manipulating the input and teaching the network to undo the manipulation. Examples of such input manipulation include masking out regions in image[^6], reordering/permuting patches as a Jigsaw puzzle[^5], transformations of the image, such as rotation, and viewpoint change[^7], removing color[^8], etc.  

<!-- A large variety of pretext tasks have been proposed to learn neural representations of the data's underlying structure. For example, Noroozi et al.propose to find a reordering of tiles from a 3x3 grid of a square region cropped from an image as the pretext task; \cite{9879206} propose masked autoencoders, which masks random patches of the input image and reconstructs the missing patches as the pretext task; \cite{pmlr-v139-radford21a}'s CLIP: given an image, predict which out of a set of 32,768 randomly sampled text snippets, was actually paired with it in their dataset as the pretext task. 

Since the main objective in this tasks is to rely on existing unlabeled data for pre-training, one of the main similarities in these tasks is to degrade the input and define a task for the network to reconstruct the original input. For example, \cite{8579073} shuffle and \cite{9879206} remove patches from the image, \cite{rombach2021highresolution} successively apply Gaussian noise to the image -->

{:refdef: style="text-align: center;"}
![pretext](https://production-media.paperswithcode.com/thumbnails/task/task-0000001882-b4b42454.jpg) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>. Image from [Yann LeCunn](https://www.youtube-nocookie.com/watch?v=7I0Qt7GALVk).
</sup></sub>
{: refdef}

# References

[^1]: Northcutt, C. G., Athalye, A., & Mueller, J. (2021). [Pervasive label errors in test sets destabilize machine learning  enchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html). Proceedings of the 35th Conference on Neural Information Processing Systems Track on Datasets and Benchmarks
[^2]: https://www.bpesquet.fr/slides/deconstructing-ai/
[^3]: Redmon, Joseph (2016). "You only look once: Unified, real-time object detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
[^4]: Alexander M. Bronstein, Michael M. Bronstein, Alfred M. Bruckstein, and Ron Kimmel. 2008. Analysis of Two-Dimensional Non-Rigid Shapes. Int. J. Comput. Vision 78, 1 (June 2008), 67–88. https://doi.org/10.1007/s11263-007-0078-4
[^5]: M. Noroozi and P. Favaro. [Unsupervised learning of visual representations by solving jigsaw puzzles](https://arxiv.org/pdf/1603.09246.pdf). In ECCV, 2016.
[^6]: K. He, X. Chen, S. Xie, Y. Li, P. Dollár and R. Girshick, "[Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 15979-15988, doi: 10.1109/CVPR52688.2022.01553.
[^7]: Pulkit Agrawal, Joao Carreira, and Jitendra Malik. 2015. [Learning to See by Moving](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Agrawal_Learning_to_See_ICCV_2015_paper.pdf). In Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV) (ICCV '15). IEEE Computer Society, USA, 37–45. https://doi.org/10.1109/ICCV.2015.13
[^8]: Zhang, Isola, Efros. [Colorful Image Colorization](https://arxiv.org/abs/1603.08511). In ECCV, 2016 


# Resources

Lectures

- Yann LeCunn's [](https://www.facebook.com/epflcampus/videos/1960325127394608), [[Slides](https://drive.google.com/file/d/12pDCno02FJPDEBk4iGuuaj8b2rr48Hh0/view)]

Code

- (https://github.com/facebookresearch/vissl)