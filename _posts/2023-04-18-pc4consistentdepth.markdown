---
layout: post
title:  "Pose Constraints for Self-supervised Monocular Depth and Ego-Motion"
description: "Enforcing pose network to be consistent improves depth consistency"
image: https://wsrv.nl/?url=https://zshn25.github.io/images/3dreco/out.gif&h=300
date:   2023-04-17 19:21:23 -0700
categories: deep-learning self-supervision 3d-reconstruction structure-from-motion visual-odometry depth-estimation
author: Zeeshan Khan Suri (DENSO ADAS Engineering Services GmbH)
published: true
comments: true
---

&nbsp;

{::nomarkdown}
<!-- Checkout Bulma buttom elements for Nerfies in custom.styles.scss -->
<div style="text-align: center;">
    <!-- PDF Link. -->
    <span class="link-block">
      <a href="https://link.springer.com/chapter/10.1007/978-3-031-31438-4_23" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="ai ai-springer"></i>
        </span>
        <span>Paper</span>
      </a>
    </span>
    <span class="link-block">
      <a href="https://arxiv.org/abs/2304.08916" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="ai ai-arxiv"></i>
        </span>
        <span>arXiv</span>
      </a>
    </span>
    <!-- Video Link. -->
    <span class="link-block">
      <a href="https://www.youtube.com/watch?v=AN1AGR85N2A"  target="_blank"
          class="external-link button is-normal is-rounded is-dark disabled">
        <span class="icon">
            <i class="fab fa-youtube"></i>
        </span>
        <span>Video</span>
      </a>
    </span>
    <!-- Code Link. -->
    <span class="link-block">
      <a href="https://github.com/zshn25/pc4consistentdepth" target="_blank"
          class="external-link button is-normal is-rounded is-dark"> 
        <span class="icon">
            <i class="fab fa-github"></i>
        </span>
        <span>Code</span>
        </a>
    </span>
    <!-- Dataset Link. -->
    <span class="link-block">
      <a href="https://www.cvlibs.net/datasets/kitti/" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="far fa-images"></i>
        </span>
        <span>Data</span>
        </a>
  </div>
{:/}

&nbsp;

{:refdef: style="text-align: center;"}
![teaser]({{site.baseurl}}/images/3dreco/out.gif){: width="100%" .shadow}
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Consistent Depth without any post-processing. [KITTI dataset](https://www.cvlibs.net/datasets/kitti/index.php)[^2]*
</sup></sub>
{: refdef}

## Abstract

Self-supervised monocular depth estimation approaches suffer not only from scale ambiguity but also infer temporally inconsistent depth maps w.r.t. scale.

{:refdef: style="text-align: center;"}
![Enforcing_Temporal_Consistency_in_Video_Depth_Estimation]({{site.baseurl}}/images/3dreco/Li_Enforcing_Temporal_Consistency_in_Video_Depth_Estimation_ICCVW_2021.png)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Two depth estimation results on four frames showing the inconsistency problem in existing MDE methods. Image taken from Li et. al.[^1]*
</sup></sub>
{: refdef}

While disambiguating scale during training is not possible without some kind of ground truth supervision, having scale consistent depth predictions would make it possible to calculate scale once during inference as a post-processing step and use it over-time. With this as a goal, a set of temporal consistency losses that minimize pose inconsistencies over time are introduced. Evaluations show that introducing these constraints not only reduces depth inconsistencies but also improves the baseline performance of depth and ego-motion prediction.

{:refdef: style="text-align: center;"}
![teaser]({{site.baseurl}}/images/3dreco/md2seqscales.png) 
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Scale factors ($$\frac{\text{GT}}{\text{pred}}$$) within each KITTI sequences are highly varying*
</sup></sub>
{: refdef}


## Introduction

For an introduction to self-supervised monocular depth estimation, checkout my previous blog post on [Self-supervised Monocular Depth Estimation]({{ site.baseurl }}{% post_url 2022-10-31-How-Monocular-Depth-Estimation-works %})

{% include youtube.html content="AN1AGR85N2A" %}{: width="100%" .shadow}
___


## BibTeX

```
@InProceedings{10.1007/978-3-031-31438-4_23,
author="Suri, Zeeshan Khan",
editor="Gade, Rikke and Felsberg, Michael and K{\"a}m{\"a}r{\"a}inen, Joni-Kristian",
title="Pose Constraints for Consistent Self-supervised Monocular Depth and Ego-Motion",
booktitle="Image Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="340--353",
isbn="978-3-031-31438-4",
doi={10.1007/978-3-031-31438-4_23}
}
```

{::nomarkdown}
<!-- Checkout Bulma buttom elements for Nerfies in custom.styles.scss -->
<div style="text-align: center;">
    <!-- PDF Link. -->
    <span class="link-block">
      <a href="https://link.springer.com/chapter/10.1007/978-3-031-31438-4_23" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="ai ai-springer"></i>
        </span>
        <span>Paper</span>
      </a>
    </span>
    <span class="link-block">
      <a href="https://arxiv.org/abs/2304.08916" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="ai ai-arxiv"></i>
        </span>
        <span>arXiv</span>
      </a>
    </span>
    <!-- Video Link. -->
    <span class="link-block">
      <a href="https://www.youtube.com/watch?v=AN1AGR85N2A"  target="_blank"
          class="external-link button is-normal is-rounded is-dark disabled">
        <span class="icon">
            <i class="fab fa-youtube"></i>
        </span>
        <span>Video</span>
      </a>
    </span>
    <!-- Code Link. -->
    <span class="link-block">
      <a href="https://github.com/zshn25/pc4consistentdepth" target="_blank"
          class="external-link button is-normal is-rounded is-dark"> 
        <span class="icon">
            <i class="fab fa-github"></i>
        </span>
        <span>Code</span>
        </a>
    </span>
    <!-- Dataset Link. -->
    <span class="link-block">
      <a href="https://www.cvlibs.net/datasets/kitti/" target="_blank"
          class="external-link button is-normal is-rounded is-dark">
        <span class="icon">
            <i class="far fa-images"></i>
        </span>
        <span>Data</span>
        </a>
  </div>
{:/}


# References:


[^1]: Li, S., Luo, Y., Zhu, Y., Zhao, X., Li, Y., Shan, Y.: [Enforcing temporal consistency in video depth estimation](https://openaccess.thecvf.com/content/ICCV2021W/PBDL/papers/Li_Enforcing_Temporal_Consistency_in_Video_Depth_Estimation_ICCVW_2021_paper.pdf). In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 1145–1154 (October 2021)
[^2]: A Geiger, P Lenz, C Stiller, and R Urtasun. 2013. Vision meets robotics: The KITTI dataset. Int. J. Rob. Res. 32, 11 (September 2013), 1231–1237. https://doi.org/10.1177/0278364913491297

*[MDE]: monocular depth estimation
*[GT]: ground-truth
