---
layout: post
title:  "Self-supervised monocular depth estimation"
description: "How to estimate depth and ego-motion from videos: Neural Networks"
image: https://github.com/nianticlabs/monodepth2/blob/master/assets/teaser.gif?raw=true
date:   2022-10-30 19:21:23 -0700
categories: deep-learning computer-vision 3d-reconstruction structure-from-motion visual-odometry
author: Zeeshan Khan Suri
published: true
comments: true
---

Animals (and in extension, humans) are unable to directly perceive the 3D surroundings around us. Each of our eyes projects the 3D world onto 2D, losing the depth dimension. Instead, we rely on our brain to reconstruct these 2D projections to perceive depth. Having more than one eye allows us to geometrically reconstruct depth via triangulation[^3], but how are creatures with a single eye (for e.g., due to a defective eye, or due to a birth disorder[^1]) still able to perceive it? 

{:refdef: style="text-align: center;"}
[![Odontodactylus scyllarus eyes](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Odontodactylus_scyllarus_eyes.jpg/320px-Odontodactylus_scyllarus_eyes.jpg)](https://en.wikipedia.org/wiki/File:Odontodactylus_scyllarus_eyes.jpg)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Each eye of [Mantis Shrimp](https://en.wikipedia.org/wiki/Mantis_shrimp#Eyes) possesses trinocular vision. [Cédric Peneau](https://commons.wikimedia.org/wiki/File:Odontodactylus_scyllarus_eyes.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*
</sup></sub>
{: refdef}

We rely on monocular cues for perceiving 3D even with a single eye. These monocular cues include those from the static image such as perspective, relative sizes of familiar objects, shading, occlusion, etc.; and those from motion, such as parallax, depth from motion, etc.[^2]


### Can we teach machines to estimate depth from a single image?

If animals are able to use these cues to reason about the relative 3D structure, the question arises, is there a way to make machines do the same? A famous paper from 2009 called [Make3D](http://make3d.cs.cornell.edu/)[^4] says "yes!".

{:refdef: style="text-align: center;"}
<figure class="video_container">
  <iframe width="100%" height="400vh" src="https://www.youtube-nocookie.com/embed/GWWIn29ZV4Q" frameborder="0" allowfullscreen></iframe>
</figure>
{: refdef}

But this post is not about that. We will be looking at more recent neural network based approaches. After all, neural networks can be thought as function approximators and with enough data, should be able to approximate the function $\mathcal{f}$ that maps an RGB pixel $i \in \mathbb{R}^3$ to its depth $d$

{:refdef: style="text-align: center;"}
# $d = \mathcal{f}(i)$
{:refdef}

In-fact we are looking at methods that do not rely on the availability ground-truth (GT) depths. Why? Because it is expensive and tedious to gather such ground truth, making it difficult to scale. But, how can we teach a neural network to estimate the underlying depth without having ground truth? Thanks for asking that! Geometry comes to the rescue. The idea is to synthesize different views of the same scene and compare these synthesized views with the real ones for supervision. It underlies **the brightness constantcy assumption**[^5], where parts of the same scene are assumed to be observed in multiple views. A similar assumption is used in binocular vision, where the content of what a left eye/camera sees is very similar to that of what the right one sees; and in motion/optical-flow estimation, where the motion is the vector $u_{xy}, v_{xy}$ defined by the change in pixel locations as

{:refdef: style="text-align: center;"}
## $I(x, y, t) = I(x + u_{xy}, y + v_{xy}, t + 1)$
{:refdef}


## The stereo case
Godard et.al., propose a method called monodepth[^6], to estimate depth from a single image, trained on stereo image pairs (without the need for depth GT). The idea is similar: synthesize the right view from the left one and compare the synthesized and the real right views as supervision. 



<!-- {:refdef: style="text-align: center;"} -->
![s]({{site.baseurl}}/images/3dreco/input_right.jpg){: width="50%" .shadow} ![right]({{site.baseurl}}/images/3dreco/input_left.jpg){: width="50%" .shadow}
<!-- {: refdef} -->
{:refdef: style="text-align: center;"}
<sub><sup>*Left and right images of the same scene from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/index.php)[^7]*
</sup></sub>
{: refdef}

But, how will this help learn depth? This is because the synthesis is a function of depth. In a calibrated stereo setting, the optical flow is unidirectional (horizontal) and so only its magnitude a.k.a. disparity is to be found. The problem then boils down to finding a per-pixel disparity that when applied to the left image, gives the right image.

{:refdef: style="text-align: center;"}
## $I_{l}(x, y) \stackrel{!}{=} I_r(x + d_{xy}, y)$
{:refdef}

The depth from disparity can be calculated by $\text{depth} = \frac{\text{focal length}\times\text{baseline}}{\text{disparity}}$.

{:refdef: style="text-align: center;"}
![triangulation](https://upload.wikimedia.org/wikipedia/commons/c/cd/Triangulation.svg)
{:refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Depth as a function of disparity via triangulation[^3]. By [fr:Utilisateur:COLETTE](https://commons.wikimedia.org/wiki/File:Triangulation.svg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0), via Wikimedia Commons*
</sup></sub>
{: refdef}

<!-- https://camo.githubusercontent.com/347a28083896fc6b18f12e29933fb7adc3ebfa485ee383897c59fe4a0983f97e/687474703a2f2f76697375616c2e63732e75636c2e61632e756b2f707562732f6d6f6e6f44657074682f6d6f6e6f64657074685f7465617365722e676966 -->

## The Monocular case

While the stereo case is analogous to animals using binocular vision to perceive 3D, what about the monocular case, where creatures are still able to reconstruct the underlying 3D structure of the scene using a single eye? Can the above method be extended for monocular case?

{:refdef: style="text-align: center;"}
[![s](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/2551b04d-fd4e-4ec9-9869-3e8c9ac5e7bf/d93a0lc-e4ecbd7d-6120-4925-be1d-1e250d7e0830.png/v1/fill/w_1024,h_576,q_80,strp/cyclops_greek_mythology_by_nilesdino_d93a0lc-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NTc2IiwicGF0aCI6IlwvZlwvMjU1MWIwNGQtZmQ0ZS00ZWM5LTk4NjktM2U4YzlhYzVlN2JmXC9kOTNhMGxjLWU0ZWNiZDdkLTYxMjAtNDkyNS1iZTFkLTFlMjUwZDdlMDgzMC5wbmciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.kEfvRjpv5KXW_HtOAtsBltiTNTF7DswBz8TwLvRVwyo){: width="50%" .shadow}](https://www.deviantart.com/nilesdino/art/Cyclops-greek-mythology-549701760)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Cyclops by [Nilesdino](https://www.deviantart.com/nilesdino), [CC BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/)*
</sup></sub>
{: refdef}

The answer is not straightforward. Even if we think about exploiting motion, the relative ego-motion of the camera from one frame at time $t$ to the next $t+1$ is not known, unlike in stereo case where the pose between the left and right cameras is known (and uni-directional for calibrated cases). Not only is the camera pose not known between two timesteps, it is also changing and not constant as in stereo case.


# References:


[^1]: [Cyclopia](https://en.wikipedia.org/wiki/Cyclopia)
[^2]: [Monocular cues for depth perception](https://en.wikipedia.org/wiki/Depth_perception#Monocular_cues)
[^3]: [Triangulation](https://en.wikipedia.org/wiki/Triangulation)
[^4]: A. Saxena, M. Sun and A. Y. Ng, "[Make3D: Learning 3D Scene Structure from a Single Still Image](https://ieeexplore.ieee.org/document/4531745)," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 5, pp. 824-840, May 2009, doi: 10.1109/TPAMI.2008.132.
[^5]: [Brightness Constancy](https://www.cs.cmu.edu/~16385/s17/Slides/14.1_Brightness_Constancy.pdf)
[^6]: C. Godard, O. M. Aodha and G. J. Brostow, "[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://github.com/mrharicot/monodepth)," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6602-6611, doi: 10.1109/CVPR.2017.699.
[^7]: A Geiger, P Lenz, C Stiller, and R Urtasun. 2013. Vision meets robotics: The KITTI dataset. Int. J. Rob. Res. 32, 11 (September 2013), 1231–1237. https://doi.org/10.1177/0278364913491297


*[GT]: ground-truth


If this article was helpful to you, consider citing

```
@misc{suri_how_monocular_depth_estimation_2022,
      title={Self-supervised monocular depth estimation},
      url={https://zshn25.github.io/is-convolution-linear/}, 
      journal={Curiosity}, 
      author={Suri, Zeeshan Khan}, 
      year={2022}, 
      month={Oct}}
```