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
{:title="The blockquote title"}
{:.no_toc}

<iframe src="https://ourworldindata.org/grapher/artificial-intelligence-training-computation" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
{:refdef: style="text-align: center;"}
<sub><sup>*Model size of notable AI systems over time. Source: [Our World in Data](https://ourworldindata.org/grapher/artificial-intelligence-training-computation)*
</sup></sub>
{: refdef}

Recent developments in deep learning can be attributed to the growing amount of data and model size. This realization has lead to the rapidly developing field of **foundational models**, i.e. large scale, complex neural networks, for e.g. [GPT 4](https://openai.com/research/gpt-4), [DINO v2](https://dinov2.metademolab.com/), [ImageBind](https://facebookresearch.github.io/ImageBind/paper), etc., trained on very large, diverse datasets. By feeding-in huge amounts of high quality labeled data through large artificial neural network models, machines are shown to be taught to outperform humans at multiple tasks.

{:refdef: style="text-align: center;"}
[![s]({{site.baseurl}}/images/performance_data.svg)]({{ site.baseurl }}{% post_url 2022-12-1-Software-2 %})
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*As the training data increases, the performance of a DL model increases but traditional methods do not depend on the training data*
</sup></sub>
{: refdef}

But, collecting such massive amounts of carefully labeled data costs enormous time, effort and money; and are error prone. For e.g. Northcutt et al.[^1] outline errors in the labels of widely used datasets. What if there was a way to use massively available but **unlabled** data?

In this post, I motivate the necessity for large, general-purpose data representation models, known as foundation models, and in what ways self-supervised learning enables this achievement. Stay tuned!

## What are foundation models?

Behind the fancy wording[^12], the idea is to learn a generic model that produces good input representations, is robust to variety of inputs, and can be directly used for multiple other tasks without the need for re-training. Any model that is capable of doing so can be named a foundation model.

> "A foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks"
>
> Bommasani, Rishi, et al.[^19]

The way to realize such a model is to use vast quantities of data and the way to do so is by self-supervised learning. <mark>The scale and diversity of training data is what makes a model foundational.</mark> Also, training on more data leads to generalization and robustness to distribution shifts and adversarial samples[^12]

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

With the rise of deep learning, features and their descriptors are no longer hand-engineered, but learnt from the underlying data. Vision models such as CNNs and transformers process input images to extract high-dimensional latent representations, which could be thought as features. Unlike pixel-level features such as SIFT, ORB, latents are **not local** (do not correspond to pixel locations) but **based on the semantics** of the global image contents.

> Checkout my blog post on [how to extract feature pyramids from torchvison models]({{ site.baseurl }}{% post_url 2021-02-01-ResNet-feature-pyramid-in-Pytorch %})
{:.no_toc}

{:refdef: style="text-align: center;"}
[![feature-learning](https://upload.wikimedia.org/wikipedia/commons/1/18/Denoising-autoencoder.png)](https://commons.wikimedia.org/wiki/File:Denoising-autoencoder.png)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>. Image from [neerajkrbansal1996](https://commons.wikimedia.org/wiki/File:Denoising-autoencoder.png), CC0, via Wikimedia Commons*
</sup></sub>
{: refdef}

As early as in 2009, the idea to use self-supervised learning for

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

<!-- {% include youtube.html content="https://www.youtube.com/embed/OLrYzY3jVPY" %}{: width="100%" .shadow}
{:refdef: style="text-align: center;"}
<sub><sup>
</sup></sub>
{: refdef} -->

> Self-supervised is where the supervision signal comes from data itself, unlike supervised learning, where supervision comes from explicit human labels and unsupervised learning, where there is no supervision.

Is it possible to make machines learn general concepts without explicit supervision? If so, how?

If we are able formulate **pretext tasks**, we can use <mark>self-supervised learning</mark> to <mark>learn good data representations and general concepts from massively-available, unlabelled data</mark>, we can use such models directly (without fine-tuning, Zero-Shot) for multiple **downstream tasks**. (atleast, that's the hope).

{:refdef: style="text-align: center;"}
[![feature-learning](https://upload.wikimedia.org/wikipedia/commons/0/0b/Feature_Learning_Diagram.png)](https://en.wikipedia.org/wiki/Feature_learning#/media/File:Feature_Learning_Diagram.png)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>Implicit feature representations are learned via pretext tasks and can be used as input for specific downstream tasks. Image from [Fgpacini](https://commons.wikimedia.org/wiki/File:Feature_Learning_Diagram.png), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*
</sup></sub>
{: refdef}


## Pretext tasks for images

Any task that doesn't require explitcit labels can be used as a pretext task. The goal is to be able to make use of as much data as possible. This task is not actually the task what the model would finally be required to do, but this task would allow to use much more data via self-supervised learning.

> ## For a more comprehensive list of self-supervised pretext tasks, checkout [this page from Papers with Code](https://paperswithcode.com/methods/category/self-supervised-learning), and [this curated list](https://github.com/jason718/awesome-self-supervised-learning) of research in self-supervised learning

Prediction/Generation and Similarity are the two main concepts that can be exploited to design self-supervised pretext tasks. 

### Generative pretext tasks based on restoration

These methods involve artificially manipulating the input and teaching the network to undo the manipulation. The hope is that in order to restore large a variety of training data, the network learns robust image representations. Examples of such input manipulations include masking out image regions[^6], adding Gaussian noise[^9], reordering/permuting patches as a Jigsaw puzzle[^5], transformations of the image, such as rotation, and viewpoint change[^7], removing color[^8], etc. There is no explicit need to manipulate the original image, for e.g. a GAN was used to regenerate the input image from compressed latent features[^21].

{:refdef: style="text-align: center;"}
![multi-task]({{site.baseurl}}/images/generative_cat.png) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>Examples of image degradations: masking, noising, permuting, rotating, grey-scaling. The pretext task is to recover the original image.
</sup></sub>
{: refdef}

> **Note:** Tasks such as Jigsaw puzzle and transformations of the image are not exactly generative. But, I still like to generalize them as generative. Only that the generation is parameterized to a low dimension, for e.g. classification on hash table for Jigsaw puzzle and regression of angles for rotation.

The difficulty in these methods is the parameterization of degradation is dependent on the exact degradation method, making it hard to combine with other manipulations. For example, the parameters of adding gaussian noise are the mean and standard deviation of the gaussian noise function, which can be estimated by the network to subtract and recover the original image. For rotation, this would be the angle, which means there should now be another head to predict the angle if these both are to be combined.

<!-- A large variety of pretext tasks have been proposed to learn neural representations of the data's underlying structure. For example, Noroozi et al.propose to find a reordering of tiles from a 3x3 grid of a square region cropped from an image as the pretext task; \cite{9879206} propose masked autoencoders, which masks random patches of the input image and reconstructs the missing patches as the pretext task; \cite{pmlr-v139-radford21a}'s CLIP: given an image, predict which out of a set of 32,768 randomly sampled text snippets, was actually paired with it in their dataset as the pretext task. 

Since the main objective in this tasks is to rely on existing unlabeled data for pre-training, one of the main similarities in these tasks is to degrade the input and define a task for the network to reconstruct the original input. For example, \cite{8579073} shuffle and \cite{9879206} remove patches from the image, \cite{rombach2021highresolution} successively apply Gaussian noise to the image -->

### Feature similarity as a pretext task

Rather than having an explicit task, the idea is to maximize similarity of features of similar images. This simply means, collecting data samples which represent similar information and having a loss that minimizes distance between their feature representations.

Just doing this causes the feature representations to collapse, i.e., all data just points to the same representation. To prevent this, negative samples (having distict representation) are introduced, which act as a relative contrast to the positive samples (having similar representation).

#### Contrastive learning

{:refdef: style="text-align: center;"}
![multi-task]({{site.baseurl}}/images/contrastive.svg)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>Contrastive learning example: Cat and it's augmentations (positive samples), shown by green arrows, must have similar feature representations than the Colosseum image chosen at random (negative sample).
</sup></sub>
{: refdef}

One of the most influential self-supervised techniques is contrastive learning which, as the name implies, uses not only positive samples but also contrasts them with negative samples. The figure above outlines an example of contrastive learning. The positive examples are shown by green arrows are images sampled from an image with different augmentations, such as rotation, cropping, color, etc. Another image (Colosseum) is chosen at random which acts as the negative sample. The fundamental assumption of contrastive learning is that features of the positive samples lie closer to each other (**positive samples attract**) than to those of a negative sample (**positive-negative samples repel**). This is realized by minimizing the distance between positive samples while maximizing the distance between negative samples in feature space.

Data sampling of the positives and the negatives becomes the key here and different ways to sample them could be incorporated. Multi-modal information or pseudo-labels, such as associated text, audio, other sensor information, could be used to sample negatives and positives. SimCLR[^17], for example uses various augmentations of the same image and CMC uses various sensor information of the underlying scene for sampling positives.

One could also sample other images as positives based on methods such as clustering in the feature space or if some kind of labels or GT is known.

### Feature similarity vs. Generative pretext tasks

The advantage of feature similarity based methods is that there is no need for exact pixel to pixel mapping of the input and output, as opposed to generative pretext tasks, where the restored output is a one-to-one mapping of the manipulated/distorted input. One could use non-exact data (for example neighboring frames of a video) for sampling feature learning. One could also combine various augmentation methods mentioned in generaive pretext tasks. Feature similarity would essentially be a way to automatically incorporate various handcrafted ad-hoc heuristics via augmentations and positive-negative sampling.

Recent work titled "What Do Self-Supervised Vision Transformers Learn?" compares contrastive learning with a prominant generative pretext task called Masked Image Modeling (MIM)[^6] and finds:

1. CL plays a significant role in the later layers of the ViT model, capturing longer-range global patterns, such as the shape of an object. However, it also leads to reduced diversity of representations, thereby worsening scalability and dense prediction performance.

2. MIM utilizes high-frequency signals of the representations and mainly focuses on the early layers of the ViTs and is more texture-oriented.


### Other pretext tasks

The generative pretext tasks can be combined with contrastive pretext tasks. For e.g., [Learning to See by Looking at Noise](https://mbaradad.github.io/learning_with_noise/) combines the generative and feature similarity by generating synthetic images from random processes and having a contrastive loss between crops of synthetic images and real ones.

# Applications

The learnt features can be applied to a huge range of downstream tasks. Generic tasks include image classification, object detection, segmentation, depth estimation, etc. Here, I will mention some less-known but very interesting downstream applications.

<!--
<div>
    <img src="https://contrastive-learning.github.io/intriguing/assets/img/multi/img_methods_bi.jpeg" alt="" style="object-fit: cover; object-position: 0 0; text-align: center;">
</div>
-->

{:refdef: style="text-align: center;"}
![multi-task](https://contrastive-learning.github.io/intriguing/assets/img/multi/img_methods_bi.jpeg)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup> Local regions grouped together after applying K-means on contrastively learnt features. Source: [Intriguing Properties of Contrastive Losses](https://contrastive-learning.github.io/intriguing/)[^14]
</sup></sub>
{: refdef}

Semantic Similarity [Deep Features as a Perceptual Metric](The Unreasonable Effectiveness of Deep Features as a Perceptual Metric)[^11]

Socretic models https://socraticmodels.github.io/

{In-context learning} enables users to provide a query and a few examples from which a model derives an answer without being trained on such queries. https://www.semanticscholar.org/paper/Foundation-models-in-brief%3A-A-historical%2C-focus-Schneider/775b2dc88cf04993f8596332444a906bec2db807 


# Risks, Challanges

- {homogenization} of models might replace a myriad of task-specific models with fewer very large models controlled by few corporations leading to a shift in power and control over AI 

- Modelling pretext tasks involves manual selection of data augmentation techniques which is based on our expert heuristics and introduces human biases into the model. Best would be to design methods that just take a bunch of data and decide for themselves which ones are similar, which ones are not. Like real unsupervised learning. But, that is difficult to design.

- Most research is done on curated object-centric datasets, where the whole image consists of a single object instance. While there is some evidence that it also works for multiple objects[^14], further investigation is needed to check how good the techniques work for generic scenes. Works such as DetCon[^15] and it's follow-up: ODIN[^16], work towards this goal by relying on pre-processing the whole image into object centric ones.

- Mining positive and negative samples for contrastive learning is crucial for its performance.

- Large amounts of quality data will enable better performance and generalizability to downstream tasks. Different modalities can be used to mine such data, including synthetic data. For filtering high quality data, methods such as active learning need to be incorporated.

> ## [Continuous improvement (Active learning) pipeline for scaling with data]({{ site.baseurl }}{% post_url 2022-12-1-Software-2 %})
{:.no_toc}

# tldr; Conclusion

Performance of supervised models depends on the availability of high quality labeled data, which is tedious to aquire, and thereby limiting. Self-supervised learning not only makes it possible to realize gains via abundant unlabeled data, but also gives way for foundation models that learn a generic neural representation of inputs, whose knowledge can be transformed to application specific downstream tasks. The key to this is designing self-supervised pretext tasks to make use of unlabeled data. The underlying principle of generative pretext tasks is to corrupt the input and make the network recover the original input from the corrupt one. Feature similarity based pretext tasks learn data representations that are similar for similar inputs and dissimilar for dissimilar inputs.

___

# Further Resources

## Language Foundation models (LLM) and beyond

In this post, I focus on vision foundation models. But, there are plenty of resources for launguage foundation models a.k.a. Large Language Models (LLMs).

- [Full Stack LLM Bootcamp, 2023](https://fullstackdeeplearning.com/llm-bootcamp/)

## Lectures / Videos

- Yann LeCunn's [lecture](https://www.facebook.com/epflcampus/videos/1960325127394608), [[Slides](https://drive.google.com/file/d/12pDCno02FJPDEBk4iGuuaj8b2rr48Hh0/view)
- [FSDL 2022 Course](https://fullstackdeeplearning.com/course/2022/)'s Lecture on Language [Foundation Models](https://fullstackdeeplearning.com/course/2022/lecture-7-foundation-models/)
- [MIT FUTURE OF AI: Self-Supervised Learning and Foundation Models](https://futureofai.mit.edu/)
- Self-Supervised Learning: Self-Prediction and Contrastive Learning, [NIPS 21 Tutorial Video](https://www.facebook.com/epflcampus/videos/1960325127394608), [[Slides]](https://nips.cc/media/neurips-2021/Slides/21895.pdf)

## Libraries

- [![](https://github.com/facebookresearch/vissl/raw/main/.github/logo/Logo_Color_Light_BG.png){: style="height: 2.5em; text-align: left;" }](https://github.com/facebookresearch/vissl)
- [![](https://docs.lightly.ai/self-supervised-learning/_static/lightly_logo_crop_white_text.png){: style="height: 2.5em; text-align: left;" }](https://docs.lightly.ai/self-supervised-learning/index.html)

## Read

- Yann LeCunn's blog post [Self-supervised learning: The dark matter of intelligence](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)
- Balestriero, Randall et al. [A Cookbook of Self-Supervised Learning](https://arxiv.org/abs/2304.12210) ArXiv abs/2304.12210 (2023): n. pag.
- Weng, Lilian (2019). [Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) writes about pretext tasks in detail
- Silva, Thalles Santos (2020). [Self-Supervised Learning and the Quest for Reducing Labeled Data in Deep Learning](https://sthalles.github.io/self-supervised-learning/) gives a great introduction on how self-supervised learning can enable foundation models, anagolous to Yann LeCunn's [lecture](https://www.facebook.com/epflcampus/videos/1960325127394608)
- Ozbulak, Utku, et al. "[Know Your Self-supervised Learning: A Survey on Image-based Generative and Discriminative Training.](https://arxiv.org/abs/2305.13689)" arXiv preprint arXiv:2305.13689 (2023)

## Data

- [LAION-5B](https://laion.ai/blog/laion-5b/) Open Large-scale CLIP-filtered image-text paired dataset

# References

[^1]: Northcutt, C. G., Athalye, A., & Mueller, J. (2021). [Pervasive label errors in test sets destabilize machine learning  enchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html). Proceedings of the 35th Conference on Neural Information Processing Systems Track on Datasets and Benchmarks
[^2]: https://www.bpesquet.fr/slides/deconstructing-ai/
[^3]: Redmon, Joseph (2016). "You only look once: Unified, real-time object detection". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
[^4]: Alexander M. Bronstein, Michael M. Bronstein, Alfred M. Bruckstein, and Ron Kimmel. 2008. Analysis of Two-Dimensional Non-Rigid Shapes. Int. J. Comput. Vision 78, 1 (June 2008), 67–88. https://doi.org/10.1007/s11263-007-0078-4
[^5]: M. Noroozi and P. Favaro. [Unsupervised learning of visual representations by solving jigsaw puzzles](https://arxiv.org/pdf/1603.09246.pdf). In ECCV, 2016.
[^6]: K. He, X. Chen, S. Xie, Y. Li, P. Dollár and R. Girshick, "[Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 15979-15988, doi: 10.1109/CVPR52688.2022.01553.
[^7]: Pulkit Agrawal, Joao Carreira, and Jitendra Malik. 2015. [Learning to See by Moving](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Agrawal_Learning_to_See_ICCV_2015_paper.pdf). In Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV) (ICCV '15). IEEE Computer Society, USA, 37–45. https://doi.org/10.1109/ICCV.2015.13
[^8]: Zhang, Isola, Efros. [Colorful Image Colorization](https://arxiv.org/abs/1603.08511). In ECCV, 2016 
[^9]: Xiang, W., Yang, H., Huang, D., & Wang, Y. (2023). [Denoising Diffusion Autoencoders are Unified Self-supervised Learners](https://arxiv.org/abs/2303.09769). ArXiv, abs/2303.09769.
[^10]: Goyal, P., Caron, M., Lefaudeux, B., Xu, M., Wang, P., Pai, V.M., Singh, M., Liptchinsky, V., Misra, I., Joulin, A., & Bojanowski, P. (2021). [Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/abs/2103.01988). ArXiv, abs/2103.01988.
[^11]: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang CVPR 2018.
[^12]: Schmidt, Ludwig, et al. "[Adversarially robust generalization requires more data.](https://proceedings.neurips.cc/paper/2018/file/f708f064faaf32a43e4d3c784e6af9ea-Paper.pdf)" Advances in neural information processing systems 31 (2018).
[^13]: Xu, Jiarui and Xiaolong Wang. [“Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective.”](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Rethinking_Self-Supervised_Correspondence_Learning_A_Video_Frame-Level_Similarity_Perspective_ICCV_2021_paper.html) 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021): 10055-10065.
[^14]: Chen, Ting and Lala Li. “[Intriguing Properties of Contrastive Losses.](https://proceedings.neurips.cc/paper/2021/hash/628f16b29939d1b060af49f66ae0f7f8-Abstract.html)” Neural Information Processing Systems (2020).
[^15]: Hénaff, Olivier J., et al. "[Efficient visual pretraining with contrastive detection.](http://openaccess.thecvf.com/content/ICCV2021/html/Henaff_Efficient_Visual_Pretraining_With_Contrastive_Detection_ICCV_2021_paper.html)" Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
[^16]: Hénaff, Olivier J., et al. "[Object discovery and representation networks.](https://arxiv.org/abs/2203.08777)" Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVII. Cham: Springer Nature Switzerland, 2022.
[^17]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. [A simple framework for contrastive learning of visual representations.](https://dl.acm.org/doi/pdf/10.5555/3524938.3525087) In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 149, 1597–1607.
[^18]: Tian, Y., Krishnan, D., Isola, P. (2020). [Contrastive Multiview Coding.](https://arxiv.org/abs/1906.05849) In: Vedaldi, A., Bischof, H., Brox, T., Frahm, JM. (eds) Computer Vision – ECCV 2020. ECCV 2020. Lecture Notes in Computer Science(), vol 12356. Springer, Cham.
[^19]: Bommasani, Rishi, et al. ["On the opportunities and risks of foundation models."](https://crfm.stanford.edu/report.html) arXiv preprint arXiv:2108.07258 (2021).
[^20]: Park, Namuk, et al. "[What Do Self-Supervised Vision Transformers Learn?.](https://arxiv.org/abs/2305.00729)" arXiv preprint arXiv:2305.00729 (2023).
[^21]: Radford, Alec, Luke Metz, and Soumith Chintala. "[Unsupervised representation learning with deep convolutional generative adversarial networks.](https://arxiv.org/abs/1511.06434)" arXiv preprint arXiv:1511.06434 (2015).

*[SIFT]: Scale-invariant feature transform
*[ORB]: Oriented FAST and rotated BRIEF
*[CNN]: Convolutional neural network
*[LLM]: Large language model
*[MIM]: Masked image modeling
*[GAN]: Generative adversarial network