---
layout: post
comments: true
title: "Hutchinson's Diagonal Estimator"
date: 2021-08-07
tags: optimization
---

> In this post, we'll take a look at a method of approximating large Hessian matrices using a stochastic diagonal estimator. Hutchinson's method can be used for optimization and loss-landscape analysis in deep neural networks.

<!--more-->

# Hutchinson's Diagonal Estimator

In modern machine learning with large deep models, explicit computation of the Hessian matrix is intractable. However, the Hessian matrix provides valuable information for optimization, studying generalization, and for other purposes. But even if we can't calculate the full Hessian, can we effectively approximate it?

![Neural Network Loss Surfaces]({{ '/assets/images/flat-minimum.png' | relative_url }}) *Fig. 1. Flat minima have been linked to improved generalization. The magnitude of the eigenvalues of the Hessian provide one way to characterize sharpness/flatness {% cite Keskar2017OnLT %}.*

There has been significant work towards efficient approximation of the Hessian, most notably in the form of low-rank updates like [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). This approach work well in traditional optimization settings, but is relatively slow and generally doesn't work in stochastic settings {% cite bollapragada2018progressive %}.

Alternatively, there has also been a variety of work old and new to estimate the diagonal of the Hessian as a way of approximating the matrix as a whole, for pruning {% cite LeCun1989OptimalBD %}, {% cite Hassibi1992SecondOD %} analyzing the loss landscape {% cite Yao2020PyHessianNN %}, and optimization {% cite Yao2021ADAHESSIANAA %}. It has been argued that the diagonal is a good approximation to the Hessian for machine learning problems, and that diagonal elements tend to be much larger than off-diagonal elements.

However, calculating the diagonal of the Hessian is not straightforward. Automatic differentiation libraries are designed to make large computations in parallel, so naively calculating diagonal terms of the Hessian one-by-one $$\frac{\partial^2 L(x)}{\partial x_1^2}, \frac{\partial^2 L(x)}{\partial x_2^2}, ..., \frac{\partial^2 L(x)}{\partial x_n^2}$$ requires $$n$$ backprop operations and isn't computationally feasible. However, we can estimate this diagonal relatively efficiently using randomized linear algebra in the form of Hutchinson's estimator.

## Hessian-Vector Products

While calculating the Hessian as a whole isn't possible, we *can* efficiently estimate Hessian-vector products. There are a variety of ways to do this, the simplest being a finite difference approximation:

### 1. Finite Difference Approximation

$$
H(x) v \approx \frac{g(x + rv) - g(x - rv)}{2r}
$$

The error of this approximation is $$ O ( r) $$, which means it can be quite accurate when $$r$$ is small. However, as $$r$$ becomes small, rounding errors pile up in the numerator. While not accurate enough for every application, the finite difference approximation is simple and cheap (2 gradient evaluations).

### 2. Product Rule Trick

$$
\frac{\partial (g(x)^\top v)}{\partial x} = \frac{\partial g(x)}{\partial x} v + g(\theta) \frac{\partial v}{\partial x} = \frac{\partial g(x)}{\partial x} v = H(x) v
$$

This exact method requires taking the gradient of an expression that includes the gradient. It means that the operations used to backprop must also be tracked so they can be differentiated.

### 3. Perlmutter's Trick (Chain Rule Trick)

$$
\frac{\partial g(x + r v)}{\partial r}|_{r=0} = H(x)v
$$

This method is described in more detail in {% cite Pearlmutter1994FastEM %}. Like the product rule trick, it is exact, and requires differentiating and expression that includes the gradient.

Each method has advantages and disadvantages, but the product rule trick is the implementation you'll find for Hessian-vector products in [pytorch](https://pytorch.org/docs/stable/generated/torch.autograd.functional.vhp.html#torch.autograd.functional.vhp) and in [jax](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).

## Hutchinson's Estimator

Now that we have efficient Hessian-vector multiplication, we can use Hutchinson's estimator.

### Hutchinson's Trace Estimator

Hutchinson's original estimator is for the trace of a matrix. This estimator is more common than the diagonal estimator and has been more thoroughly analyzed, so let's warm up by taking a look at it.

To estimate the trace, we draw random vectors $$z$$ from a distribution with mean zero and variance one (typically the [Rademacher distribution](https://en.wikipedia.org/wiki/Rademacher_distribution)). We can then estimate the trace of a matrix as $$\mathbb{E}[z^\top H z] = \text{trace}(H)$$.

Claim: Let $$z$$ a random vector with $$\mathbb{E}[z]=0$$, $$\mathbb{V}[z]=1$$, and independent entries. Then $$\mathbb{E}[z^\top H z] = \text{trace}(H)$$.

Proof:

$$
\begin{align}
	\mathbb{E}[z^\top H z] &= \mathbb{E}[\begin{bmatrix}z_1 \\ z_2 \\ ... \\ z_n\end{bmatrix}^\top \begin{bmatrix}H_{11} z_1 + H_{12} z_2 + ... + H_{1n} z_n \\ H_{12} z_1 + H_{22} z_2 + ... + H_{2n} z_n \\ ... \\ H_{n1} z_1 + H_{n2} z_2 + ... + H_{nn} z_n\end{bmatrix}] \\
	&= \mathbb{E}[\begin{bmatrix}z_1 \\ z_2 \\ ... \\ z_n\end{bmatrix}^\top \begin{bmatrix}\sum\limits_{i=1}^n H_{1i} z_i \\ \sum\limits_{i=1}^n H_{2i} z_i \\ ... \\ \sum\limits_{i=1}^n H_{ni} z_i\end{bmatrix}] \\
	&= \mathbb{E}[z_1 \sum\limits_{i=1}^n H_{1i} z_i + z_2 \sum\limits_{i=1}^n H_{2i} z_i + ... + z_n \sum\limits_{i=0}^n H_{ni} z_i] \\
	&= \sum\limits_{i=1}^n H_{1i} \mathbb{E}[z_1 z_i] + \sum\limits_{i=1}^n H_{2i} \mathbb{E}[z_2 z_i] + ... + \sum\limits_{i=1}^n H_{ni} \mathbb{E}[z_n z_i] \\
\end{align}
$$

And since the entries of $$z$$ are independent, $$\mathbb{E}[z_i z_j] = \mathbb{E}[z_i]\mathbb{E}[z_j] = 0 \cdot 0 = 0$$ for $$i \neq j$$.

However, when $$i = j$$, then $$\mathbb{E}[z_i z_j] = \mathbb{E}[z_i^2] = \mathbb{V}[z_i] + \mathbb{E}[z_i]^2 = 1 + 0^2 = 1$$.

So

$$
\begin{align}
  \mathbb{E}[z^\top H z] &= \sum\limits_{i=1}^n H_{1i} \mathbb{E}[z_1 z_i] + \sum\limits_{i=1}^n H_{2i} \mathbb{E}[z_2 z_i] + ... + \sum\limits_{i=1}^n H_{ni} \mathbb{E}[z_n z_i] \\
  &= H_{11} + H_{22} + ... + H_{nn} \\
  &= \text{trace}(H)
\end{align}
$$

### Hutchinson's Diagonal Estimator

The basic idea from the trace estimator can be modified to give an estimator for the diagonal rather than the trace.

Claim: Let $$z$$ be a random variable $$z$$ with $$\mathbb{E}[z]=0$$, $$\mathbb{V}[z]=1$$, and independent entries. Then $$\mathbb{E}[z \odot H z] = \text{diag}(H)$$.

Proof:

$$
\begin{align*}
    \mathbb{E}[z \odot Hz] &= \mathbb{E}[
    \begin{bmatrix}z_1 \\ z_2 \\ ... \\ z_n\end{bmatrix} \odot
    \begin{bmatrix}
    H_{11} z_1 + H_{12} z_2 + ... + H_{1n} z_n \\
    H_{21} z_1 + H_{22} z_2 + ... + H_{2n} z_n \\ ... \\
    H_{n1} z_1 + H_{n2} z_2 + ... + H_{nn} z_n
    \end{bmatrix}
    ] \\
    &= \begin{bmatrix}
    H_{11} \mathbb{E}[z_1^2] + H_{12} \mathbb{E}[z_1z_2] + ... + H_{1n}[z_1z_n] \\
    H_{21} \mathbb{E}[z_2z_1] + H_{22} \mathbb{E}[z_2^2] + ... + H_{2n}[z_2z_n] \\ ... \\
    H_{n1} \mathbb{E}[z_nz_1] + H_{n2} \mathbb{E}[z_nz_2] + ... + H_{nn}[z_n^2]
    \end{bmatrix} \\
    &= \begin{bmatrix}
    H_{11} \\ H_{22} \\ ... \\ H_{nn}
    \end{bmatrix} \\
    &= \text{diag}(H)
\end{align*}
$$



Now let's calculate the variance of Hutchinson's diagonal estimator.

Claim: Let $$z \sim \text{Rademacher}$$. Then

$$\mathbb{V}[z \odot H z] = \begin{bmatrix}||H_1||^2 \\ ||H_2||^2 \\ ... \\ ||H_n||^2\end{bmatrix} - \text{diag}(H)^2$$

where $$H_k$$ is the k-th row of $$H$$.

Proof: Let us consider each entry $$k$$ separately:

$$
\begin{align}
    (\mathbb{V}[z \odot H z])_k &= \mathbb{V}[z_k \odot (H z)_k] \\
    &= \mathbb{V}[z_k (H_{k1} z_1 + ... + H_{kk} z_k + ... + H_{kn} z_n)] \\
    &= \mathbb{V}[H_{k1} z_k z_1 + ... + H_{kk} z_k z_k + ... + H_{kn} z_k z_n] \\
    &= \sum\limits_{i=0}^n \sum\limits_{j=0}^n \text{Cov}(H_{ki} z_k z_i, H_{kj} z_k z_j) \\
    &= \sum\limits_{i=0}^n \sum\limits_{j=0}^n \mathbb{E}[(H_{ki} z_k z_i)(H_{kj} z_k z_j)] - \mathbb{E}[H_{ki} z_k z_i] \mathbb{E}[H_{kj} z_k z_j] \\
    &= \sum\limits_{i=0}^n \sum\limits_{j=0}^n H_{ki}H_{kj} (\mathbb{E}[z_k^2 z_i z_)] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j]) \\
\end{align}
$$


When $$i \neq j$$, we also know that at least one of $$i, j \neq k$$. WLOG suppose $$i \neq k$$. Then
$$\mathbb{E}[z_k^2 z_i z_j] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j] = \mathbb{E}[z_k^2 z_j]\mathbb{E}[z_i] - \mathbb{E}[z_k]\mathbb{E}[z_i]\mathbb{E}[z_k z_j] = 0 - 0 = 0$$.

Now since all terms of the sum with $$i \neq j$$ are zero,

$$
\begin{align}
	(\mathbb{V}[z \odot H z])_k &= \sum\limits_{i=0}^n H_{ki} H_{ki} (\mathbb{E}[z_k^2 z_i z_)] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j])
\end{align}
$$

And when $$i = j = k$$, we have $$\mathbb{E}[z_k^2 z_i z_j] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j] = \mathbb{E}[z_k^4] - \mathbb{E}[z_k^2]^2 = 1 - 1^2 = 0$$.

We have $$\mathbb{E}[z_k^4] = 1$$ because for the Rademacher distribution, $$\mathbb{E}[z^4] = \text{Kurtosis}[z] = 1$$. In fact, the Bernoulli distribution has the smallest kurtosis of any distribution (kurtosis 1), and since the Rademacher distribution is just a scaled Bernoulli, it is the optimal distribution from which to draw $$z$$ in terms of reducing variance of this estimator.

So since the terms of the sum when $$i = j = k$$ are zero, we have

$$
\begin{align}
	(\mathbb{V}[z \odot H z])_k &= \sum\limits_{i=0, i \neq k}^n H_{ki} H_{ki} (\mathbb{E}[z_k^2 z_i z_)] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j])
\end{align}
$$

Finally, when $$i = j \neq k$$, $$\mathbb{E}[z_k^2 z_i z_j] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j] = \mathbb{E}[z_k^2 z_i^2] - \mathbb{E}[z_k z_i]^2 = \mathbb{E}[z_k^2]\mathbb{E}[z_i^2] - (\mathbb{E}[z_k]\mathbb{E}[z_i])^2 = 1 \cdot 1 - 0^2 = 1$$

So, we have

$$
\begin{align}
  (\mathbb{V}[z \odot H z])_k &= \sum\limits_{i=0}^n \sum\limits_{j=0}^n H_{ki}H_{kj} (\mathbb{E}[z_k^2 z_i z_)] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j]) \\
    &= \sum\limits_{i=0, i \neq k}^n H_{ki}H_{ki} (\mathbb{E}[z_k^2 z_i z_)] - \mathbb{E}[z_k z_i] \mathbb{E}[z_k z_j]) \\
    &= \sum\limits_{i=0, i \neq k}^n H_{ki}^2 \\
    &= ||H_k||^2 - H_{kk}^2
\end{align}
$$

Putting this all together, we get the full variance

$$
\begin{align}
    \mathbb{V}[z \odot H z] &= \begin{bmatrix}||H_1||^2 \\ ||H_2||^2 \\ ... \\ ||H_n||^2\end{bmatrix} - \text{diag}(H)^2
\end{align}
$$



Note that the entries of vector of variances are proportional to

$$
||H_k^2||
$$

which grows linearly with the number of variables in the Hessian.



## References

{% bibliography --cited %}
