---
layout: post
title:  "Pose Constraints for Self-supervised Monocular Depth and Ego-Motion"
description: "Enforcing pose network to be consistent improves depth consistency"
# image: https://github.com/nianticlabs/monodepth2/blob/master/assets/teaser.gif?raw=true
date:   2022-10-30 19:21:23 -0700
categories: deep-learning self-supervision 3d-reconstruction structure-from-motion visual-odometry depth-estimation
author: Zeeshan Khan Suri (DENSO ADAS Engineering Services GmbH)
published: false
comments: true
---

{::nomarkdown}

<button name="button">Click me</button>

<div class="publication-links">
    <!-- PDF Link. -->
    <span class="link-block">
      <a href="https://arxiv.org/pdf/2011.12948"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="fas fa-file-pdf"></i>
        </span>
        <span>Paper</span>
      </a>
    </span>
    <span class="link-block">
      <a href="https://arxiv.org/abs/2011.12948"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="ai ai-arxiv"></i>
        </span>
        <span>arXiv</span>
      </a>
    </span>
    <!-- Video Link. -->
    <span class="link-block">
      <a href="https://www.youtube.com/watch?v=MrKrnHhk8IA"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="fab fa-youtube"></i>
        </span>
        <span>Video</span>
      </a>
    </span>
    <!-- Code Link. -->
    <span class="link-block">
      <a href="https://github.com/google/nerfies"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="fab fa-github"></i>
        </span>
        <span>Code</span>
        </a>
    </span>
    <!-- Dataset Link. -->
    <span class="link-block">
      <a href="https://github.com/google/nerfies/releases/tag/0.1"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="far fa-images"></i>
        </span>
        <span>Data</span>
        </a>
  </div>
{:/}

## Abstract

Self-supervised monocular depth estimation approaches suffer not only from scale ambiguity but also infer temporally inconsistent depth maps w.r.t. scale.
{:refdef: style="text-align: center;"}
![]({{site.baseurl}}/images/Li_Enforcing_Temporal_Consistency_in_Video_Depth_Estimation_ICCVW_2021.png)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Two depth estimation results on four frames showing the inconsistency problem in existing MDE methods. Image taken from Li et. al.[^1]*
</sup></sub>
{: refdef}
While disambiguating scale during training is not possible without some kind of ground truth supervision, having scale consistent depth predictions would make it possible to calculate scale once during inference as a post-processing step and use it over-time. With this as a goal, a set of temporal consistency losses that minimize pose inconsistencies over time are introduced. Evaluations show that introducing these constraints not only reduces depth inconsistencies but also improves the baseline performance of depth and ego-motion prediction.


## Introduction

For an introduction to self-supervised monocular depth estimation, checkout my previous blog post on 

___


## BibTeX

```
@misc{suri2023pose,
      title={Pose Constraints for Consistent Self-supervised Monocular Depth and Ego-motion}, 
      author={Zeeshan Khan Suri},
      year={2023},
      eprint={2304.08916},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# References:


[^1]: Li, S., Luo, Y., Zhu, Y., Zhao, X., Li, Y., Shan, Y.: [Enforcing temporal consistency in video depth estimation](https://openaccess.thecvf.com/content/ICCV2021W/PBDL/papers/Li_Enforcing_Temporal_Consistency_in_Video_Depth_Estimation_ICCVW_2021_paper.pdf). In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 1145–1154 (October 2021)

*[MDE]: monocular depth estimation
*[DOF]: degrees of freedom