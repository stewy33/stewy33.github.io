# Duality in Optimization - From First Principles

 Despite being exposed to duality in optimization early in undergrad, each time I returned to the subject, I realized how little I understood it. I also found that most people I've worked with, even some optimization professors, lacked a full understanding of duality and where it comes from. Deeper answers can be found in classic optimization references, but they require a lot of investment and can be hard to follow. This is my attempt to write an understandable guide to duality, from first principles, that I wish I had earlier. This post is intended for people with a basic knowledge of convex optimization. If you have questions, I'd love to talk - email me at slocumstewy@gmail.com!

## Review of Convex Analysis

Definition: A set $A$ is a *convex set* if for every pair of elements $x, y \in A$, and for all $\lambda \in [0,1]$, we have $\lambda x + (1 - \lambda) y \in A$.

In other words, $A$ is convex if for any two points in $A$, their entire line segment is also in $A$.

Definition (first definition of a convex function): Let $A$ be a convex set and let $f : A \rightarrow \overline{\mathbb{R}}$. $f$ is said to be a *convex function* if:
$$
f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y) \tag*{$\forall \lambda \in [0,1]$}
$$
This definition says that $f$ is convex if for any two points its domain, their line segment lies above the graph of $f$.

This leads us to an equivalent, second definition of a convex function in terms of convex sets:

Definition: Given a function $f: A \rightarrow \overline{\mathbb{R}}$, we say that $\text{epi}(f) = \{(x, y) : x \in A, y \geq f(x)\}$ is the *epigraph* of $f$.

The epigraph is simply the set of points that lie above the graph of $f$.

Definition (second definition of a convex function): Let $A$ be a convex set and let $f: A \rightarrow \overline{\mathbb{R}}$. $f$ is said to be a *convex function* if $\text{epi}(f)$ is a convex set.

## What is Duality?

Broadly defined, a duality translates a set of (primal) objects or mathematical structures into another set of (dual) objects or mathematical structures. Dualities appear in many areas, including topology, differential geometry, and optimization. As we'll see, some of these are related. Duality theories provide a different perspective for looking at the same things. Sometimes, a dual perspective makes a problem more understandable or easier to solve, analytically or algorithmically.

In optimization, we consider problems of the form $\min\limits_{x \in X} f(x)$ where $f : X \rightarrow \mathbb{R}$ and $X \subseteq \mathbb{R}^n$. Following Ekeland and Télam, we call our original problem the *primal problem* $\mathscr{P}$, which we associate with a *dual problem* $\mathscr{P^*}$.

I've come across two starting points on how to develop a general duality theory, a more common one based on perturbation functions and a second based on the minimax theorem. Both are equivalent.

## History of Duality in Optimization

It appears that our modern concept of primal and dual optimization problems emerged first from linear programming. Duality in linear programming has its own unique geometric intuition and was famously discovered by John von Neuman in the mid-20th century. In 1951, Kuhn and Tucker published a paper proposing the KKT conditions, which expanded the classic method of Lagrange multipliers for constrained nonlinear optimization to the case with inequality constraints. Inspired by the minimax theorem from game theory, they extended the special case of LP duality to general constrained nonlinear optimization, with a dual function based off of the Lagrangian.

### Characteristic of Duality

There are several striking features these different forms of duality share:

1. A *mapping* between a primal minimization problem $\mathscr{P} = \min\limits_{x \in X} f(x)$ in $X$, and a dual maximization problem $\mathscr{P^*} = \max\limits_{y* \in Y^*} q(y^*)$ in a dual space $Y^*$.
2. *Weak duality* between primal and dual: $\mathscr{P^*} = \max\limits_{y* \in Y^*} q(y^*) \leq \min\limits_{x \in X} f(x) = \mathscr{P}$.
3. Under certain conditions, *strong duality* between primal and dual: $\mathscr{P^*} = \max\limits_{y* \in Y^*} q(y^*) = \min\limits_{x \in X} f(x) = \mathscr{P}$. When strong duality holds, the dual of the dual is equivalent to the primal $\mathscr{P^{**}} = \mathscr{P}$.

## Duality from a Perturbation Function

Definition: Given an $f : X \rightarrow \mathbb{R}$, we call a function $\Phi : X \times Y \rightarrow \mathbb{R}$ a *perturbation function* for $f$ if and only if $\Phi(x, 0) = f(x)$.

In other words, $\Phi(x, y)$ is a perturbed version of $f(x)$ where the perturbation is controlled by $y$. We use this function to define perturbed versions of the optimization problem:
$$
\mathscr{P_y} = \min\limits_{x \in X} \Phi(x, y)
$$
Now we define a function $h(y)$, which is the solution to the perturbed optimization problem at a particular perturbation $y$:
$$
h(y) = \mathscr{P_y} = \inf\limits_{x \in X} \Phi(x, y)
$$
By definition of a perturbation function, $h(0) = \inf\limits_{x \in X} \Phi(x, 0) = \inf\limits_{x \in X} f(x)$ expresses the solution to the primal problem. Now we use this same $h$ to derive the dual problem.

Definition: Given a primal problem $\mathscr{P} = \min\limits_{x \in X} f(x)$ and a perturbation function $\Phi(x, y)$, we call
$$
\mathscr{P^*} = h^{**}(0) = \sup\limits_{y^* \in Y^*} -\Phi^*(0, y^*)
$$
the *dual problem*.

First, let's show that $h^{**}(0)$ does indeed equal $\sup\limits_{y^* \in Y^*} -\Phi^*(0, y^*)$. This requires a couple tricks:
$$
\begin{align}
h^{**}(0) &= \sup\limits_{y^* \in Y^*} \langle 0, y^* \rangle - h^*(y^*) \\
&= \sup\limits_{y^* \in Y^*} -h^*(y^*) \\
&= \sup\limits_{y^* \in Y^*} -[\sup\limits_{y \in Y} \langle y^*, y \rangle - h(y)] \\
&= \sup\limits_{y^* \in Y^*} -[\sup\limits_{y \in Y} \langle y^*, y \rangle - \inf\limits_{x \in X} \Phi(x, y)] \\
&= \sup\limits_{y^* \in Y^*} -[\sup\limits_{y \in Y} \langle y^*, y \rangle + \sup\limits_{x \in X} -\Phi(x, y)] \\
&= \sup\limits_{y^* \in Y^*} -[\sup\limits_{x \in X, y \in Y} \langle y^*, y \rangle - \Phi(x, y)] \\
&= \sup\limits_{y^* \in Y^*} -[\sup\limits_{x \in X, y \in Y} \langle \begin{pmatrix}0 \\ y^*\end{pmatrix}, \begin{pmatrix}x \\ y\end{pmatrix} \rangle - \Phi(x, y)] \\
&= \sup\limits_{y^* \in Y^*} -\Phi^*(0, y^*)
\end{align}
$$
Anyways, what exactly is going on here? Why choose to define the dual problem as the biconjugate of $h$ evaluated at zero?

Notice that the necessary features of duality emerge right from this definition:

1. *Dual mapping*: We have a mapping from a primal minimization problem $X$ to a dual a maximization problem in the dual space $Y^*$.
2. *Weak duality*: By definition, the biconjugate of a function is upper bounded by the original function: $h^{**}(0) \leq h(0)$, giving us weak duality.
3. *Strong duality*: By definition of the biconjugate, when $h$ is convex and lower semi-continuous, we have $\mathscr{P^*} = h^{**}(0) = h(0) = \mathscr{P}$. This criteria for strong duality is equivalent to other ways of expressing sufficient conditions for strong duality, like Slater's condition for the convex case. Also note that when strong duality holds, taking the dual of the dual gives us the original primal, $(h^*)^*(0) = h^{**}(0) = h(0)$, meaning the dual completely characterizes the primal, a standard feature of strong duality.

 So from our definition, we immediately obtain a duality mapping, weak duality and strong duality - isn't that cool?



Depending on the perturbation we choose, we will get a different dual problem. See the section on specific duality theories for examples of what these perturbation functions might look like.