---
layout: post
comments: true
title: "A Practical Vectorization-based Tensor Calculus for Deep Learning"
date: 2021-08-07
---

> A tutorial on a systematic yet approachable method of calculating matrix and tensor derivatives with applications in machine learning.

<!--more-->
$$\newcommand{\bm}[1]{\boldsymbol{#1}}$$
When introductory machine learning courses cover gradient backpropagation, students often find themselves caught up in a seemingly arbitrary mess of matrices and transposes without explanation. The multivariable versions of the product rule and chain rule are provided without clear instruction on the subleties of applying them.

Some resources avoid these subtleties by computing matrix derivatives using element-wise partial derivatives, but this is confusing to apply and results in a loss of the underlying matrix and tensor structure. And when working with higher-order derivatives like the Hessian, this approach is hardly feasible. Handbooks of identities for matrix calculus like the [Matrix Cookbook](http://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf) are of some value, but don't teach a systematic approach, so readers have trouble moving on to more complex cases. This is all taught so poorly that most people either fearfully avoid analytical calculation of neural network derivatives or resort to guessing and checking dimensions.

While there are more advanced treatments of tensor calculus using [Ricci Calculus](https://en.wikipedia.org/wiki/Ricci_calculus) (tensor index notation), here I will share an approach that I find more straightforward, based on matrix vectorization.

My goal is for this post to leave you with the necessary knowledge and machinery to confidently take derivatives of matrices and tensors. Later on, I will work through some examples, including calculating gradients and Hessians of deep neural networks. If this post is unclear or you find errors, please let me know!

{: class="table-of-content"}
* TOC
{:toc}

## Basic Concepts
In general, for matrices $$\bm{Y} \in \mathbb{R}^{l \times m}, \bm{X} \in \mathbb{R}^{n \times o}$$, the matrix-matrix derivative

$$
\frac{\partial \bm{Y}}{\partial \bm{X}} \in \mathbb{R}^{l \times m \times n \times o}
$$

is a fourth order tensor. Rather than work with tensors and their complex machinery directly, we will simplify our calculations by row-wise vectorizing input and output matrices, defining

$$
\frac{\partial \bm{Y}}{\partial \bm{X}} := \frac{\partial \text{vec}_r(\bm{Y})}{\partial \text{vec}_r(\bm{X})^\top} \in \mathbb{R}^{lm \times no}
$$

to get *matrices* out of matrix-matrix derivatives. This gives a systematic and approachable technique to doing tensor calculus that leverages familiar vectors and matrices.

### Layout Conventions
This is a short summary on layout conventions in matrix calculus. For more information, see [matrix calculus layout conventions](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions) on Wikipedia.

Let $$\bm{y} \in \mathbb{R}^m, \bm{x} \in \mathbb{R}^n$$. Confusingly, different authors use two different layout conventions for vector-vector derivatives $$\frac{\partial \bm{y}}{\partial \bm{x}}$$
1. **Numerator layout** - The derivative $$\frac{\partial \bm{y}}{\partial \bm{x}}$$ is laid out according to $$\bm{y}$$ and $$\bm{x}^\top$$. In this layout, $$\frac{\partial \bm{y}}{\partial \bm{x}} := \frac{\partial \bm{y}}{\partial \bm{x}^\top}$$ is an $$m \times n$$ matrix, like a standard Jacobian.
2. **Denominator layout** - The derivative $$\frac{\partial \bm{y}}{\partial \bm{x}}$$ is laid out according to $$\bm{y}^\top$$ and $$\bm{x}$$. In this layout, $$\frac{\partial \bm{y}}{\partial \bm{x}} := \frac{\partial \bm{y}^\top}{\partial \bm{x}}$$ is an $$n \times m$$ matrix.

In this article, I will stick to the more popular *numerator layout* and use the explicit notation $$\frac{\partial \bm{y}}{\partial \bm{x}^\top}$$ to communicate that I am using the numerator layout. However, when reading other authors, you might find either notation, or sometimes even a mixture of notation explanation.

### The Vectorization Operator
jklj
<br/><br/>

---

<br/>
Now let's take a break with

## References

{% bibliography --cited %}
