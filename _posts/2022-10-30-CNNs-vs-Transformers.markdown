---
layout: post
title:  "Convolution vs. Attention"
description: "Similarities and differences between convolutions and attention module of Transformers"
image: https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Vision_Transformer.gif/320px-Vision_Transformer.gif
date:   2023-03-09 19:21:23 -0700
categories: deep-learning convolution attention
author: Zeeshan Khan Suri
published: true
comments: true
---


Layers in neural netowrks can be seen as a function that takes in a multi-dimensional input and produces an output. For simplicity, let's assume the input and output dimesnions to be the same.

## Fully-connected Layer

In a fully connected layer each output is just a linear combination of all the inputs.

{:refdef: style="text-align: center;"}
$\textcolor{FF7800}{\textbf{y}} =  \textcolor{9966FF}{W} \textcolor{2EC27E}{\textbf{x}} $ 
{:refdef}

{:refdef: style="text-align: center;"}
![fullyconnected]({{site.baseurl}}/images/fullyconnected.svg)
{:refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*In a fully connected layer, each output depends on (and is connected to) all inputs*
</sup></sub>
{: refdef}

In fact, in Pytorch, a fully-connected layer is just represented by the [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear).

```python
import torch
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

The weights matrix $\textcolor{9966FF}{W}$ is the learnable parameter which learns to assign weights of each output to all of the inputs.


The problem with the fully-connected layer is that it consumes a lot of learnable parameters. Since each output is connected to all inputs, a lot of the weights have to be learnt. But, for most input types it is wasteful to connect to all inputs since dependencies are often local, for e.g. spatio-temporal neighbors, than global.

In principle, connecting to all inputs shouldn't be a problem since given enough training samples and enough time to train, the network should be able to learn to assign meaningful (and sparse) weights to some locations (like the spatio-temporal neighbors) than to the rest, which could be exploited by post-processing steps such as Pruning to fasten inference.

But, especially for data such as images where we know the inherent local dependencies of pixels to their spatial neighbors, it is a good idea to limit the layer to such neighborhood.

## Convolution


Convolutional neural networks (CNNs) are typically used for **spatial** data processing, such as images, where there is a spatial relationship between the data or temporal data such as audio. For example, neighboring pixels (in X or Y direction) are related to each other. A convolutional filter is applied to such data to extract features, such as edges, textures, etc., in images.

{:refdef: style="text-align: center;"}
![convolution]({{site.baseurl}}/images/convolution.svg)
{:refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*A convolutional layer only attends at it's neighbors*
</sup></sub>
{: refdef}


## Attention

Transformers on the other hand are typically used for **sequential** data processing, such as text, natural language, where short-term and long-term dependencies are present. The actual dependencies are not explicit in this case. For example, in the sentence "Alice had gone to the supermarket to meet Bob", one of the verb "meet", is located far-away from the subject "Alice" and this dependency is not spatial but differs a lot. This is even more for longer inputs with multiple paragraphs where the final sentence could have had a dependecy to a sentence somewhere in the beginning. Transformers are based on the so called attention mechanisms which learns these relationships between the elements in the sequence.

{:refdef: style="text-align: center;"}
[![Odontodactylus scyllarus eyes](https://upload.wikimedia.org/wikipedia/commons/3/3e/Vision_Transformer.gif){: width="100%" .shadow}](https://commons.wikimedia.org/wiki/File:Vision_Transformer.gif)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Attentions is the key component of Transformers and have been successfully applied to image data. [Davide Coccomini](https://commons.wikimedia.org/wiki/File:Vision_Transformer.gif), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*
</sup></sub>
{: refdef}

### Basic Self-attention

The basic idea of self-attention is to assign different importance to the inputs based on the inputs themselves. In comparison to convolution, self-attention allows the receptive field to be the entire spatial locations.


{:refdef: style="text-align: center;"}
![attention]({{site.baseurl}}/images/selfattention.svg)
{:refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*An attention layer assigns importance to the inputs based on the inputs themselves*
</sup></sub>
{: refdef}

The weights are computed based on cosine similarity of the inputs $\textcolor{2EC27E}{\textbf{x}}$ to themselves

{:refdef: style="text-align: center;"}
![attention]({{site.baseurl}}/images/xxt.svg)
{:refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*$\textcolor{9966FF}{W} =  \textcolor{2EC27E}{\textbf{x}}  \textcolor{2EC27E}{\textbf{x}^T}$*
</sup></sub>
{: refdef}

{% include alert.html content="Input $\textcolor{2EC27E}{\textbf{x}^T}$ and its transforms are usually multi-dimensional. Vector notation is used for illustration"%}

### Self-attention

Note that the basic version of self-attention does not include any learnable parameters. For this reason, the "Attention is all you need"<sup>\[</sup>[^1]<sup>\]</sup> variation of the self-attention includes 3 learnable weight matrices (key $\textcolor{9966FF}{W_K}$, query $\textcolor{9966FF}{W_Q}$ and value $\textcolor{9966FF}{W_V}$). But, the basic principle remains the same. The key $\textcolor{9966FF}{W_K}$ and query $\textcolor{9966FF}{W_Q}$ matrices are used to transform the input into key $\textcolor{2EC27E}{\textbf{k}} = \textcolor{9966FF}{W_K} \textcolor{2EC27E}{\textbf{x}}$, and query $\textcolor{2EC27E}{\textbf{q}} = \textcolor{9966FF}{W_Q} \textcolor{2EC27E}{\textbf{x}}$, whose similarity $\textcolor{2EC27E}{\textbf{q}} \textcolor{2EC27E}{\textbf{k}^\mathrm{T}}$,  weighs the output (value), $\textcolor{FF7800}{\textbf{v}} =  \textcolor{9966FF}{W_V} \textcolor{2EC27E}{\textbf{x}} $.<sup>\[</sup>[^2]<sup>\]</sup>

{:refdef: style="text-align: center;"}
$\textcolor{FF7800}{\textbf{y}} = \text{softmax}\left(\frac{\textcolor{2EC27E}{\textbf{q}} \textcolor{2EC27E}{\textbf{k}^\mathrm{T}}}{\sqrt{d_k}}\right)\textcolor{FF7800}{\textbf{v}}$
{:refdef}

Also note that in the basic version, the self-similarity of the inputs always causes the diagonal to be of the highest similarity and makes the weight matrix symmetric. This problem is also elevated by transforming the same input using two seperate learnable weight matrices, $\textcolor{9966FF}{W_K}$ and $\textcolor{9966FF}{W_Q}$



## Summary

- Different ways of connecting inputs to each other were discussed. A fully-connected layer connects all inputs to each other. This leads to exponential increase in network parameters and computational complexity. While the network can learn to assign different weights, this can take a lot of data and prolonged training.

- Convolutional layer incorporates desirable inductive biases about the data to reduce computation and connects only to the neighbors. Spatial and temporal data benifits from doing so. Convolution is translation invariant. However, the dimensions of outputs of a convolution depend on the input dimensions.

- A self-attention layer assigns importance to inputs based on their similarity. For e.g., in the sentence "Alice is adventurous and she is in wonderland." the word "she" refers to "Alice" and ideally, their embeddings should be similar, which can be used by the self-attention layer to determine contexts. Similar to fully-connected, far away connections can be established if the input features or embeddings are similar. However, not having enough data may lead to overfitting the inputs.

- In early layers of a neural network for images, spatial relations can be captured by convolutions and the later layers could benifit from long-range receptive fields offered by attention. Hence, both can be combined. Works such as CoAtNet<sup>\[</sup>[^3]<sup>\]</sup> offer layers combining the two. 


## References

[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://dl.acm.org/doi/10.5555/3295222.3295349). Advances in neural information processing systems, 30.
[^2]: [https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#Scaled_dot-product_attention](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#Scaled_dot-product_attention)
[^3]: Dai, Z., Liu, H., Le, Q. V., & Tan, M. (2021). [Coatnet: Marrying convolution and attention for all data sizes](https://papers.nips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html). Advances in Neural Information Processing Systems, 34, 3965-3977.