---
layout: post
comments: true
title: "What is a Good Explanation? Epistemological Insights for Interpretable Machine Learning"
date: 2022-02-18
tags: optimization
---

> In this post, I provide an epistemological perspective on what defines a good explanation.

<!--more-->

In recent years, there has a been a large (and justified) increase of interest in developing interpretable machine learning methods. Besides just being a useful property of ML systems, it is a necessary requirement for safe application of ML to high-stakes decision-making. Taking a step back further, model transparency will likely be a key ingredient in solutions to the more general problem of [AI alignment](https://en.wikipedia.org/wiki/AI_alignment) (great intro [here](https://80000hours.org/podcast/episodes/paul-christiano-ai-alignment-solutions/)), although there remain [questions about how well this will scale](https://docs.google.com/document/d/1FbTuRvC4TFWzGYerTKpBU7FJlyvjeOvVYF2uYNFSlOc/edit#heading=h.n1wk9bxo847o).

Methods for interpretable deep learning can be separated into those that seek to explain existing models (*post-hoc methods*) and those that build models that are *interpretable by design*. The majority of the work in the field has consisted of post-hoc methods, which are generally more convenient to apply and do not lead to performance reductions. However, these methods come with little guarantee that their generated explanations are faithful to the underlying model's decision-making process. Even the more principled post-hoc methods like SHAP {% cite lundberg2017unified %} can be caused to generate arbitrary explanations for an input using adversarial attacks {% cite slack2020fooling %}. Models that are interpretable by design seek to remedy these issues, since they have access to the causal factors that lead to the final prediction. However, even out of these methods, there is not a unified consensus as to what a faithful and useful explanation is: i.e. they each satisfy different definitions of faithful and useful.

In this post, I will take a step back from machine learning and provide an epistemological perspective on what defines a good explanation. This has applications to machine learning but also to explaining, understanding, and arguing things generally.


<br/><br/>

---

<br/>
Now let's take a break with this landscape by Rembrandt

<img src="{{ '/assets/images/the-mill.jpg' | relative_url }}"/>

**The Mill**
{: .center}

Courtesy National Gallery of Art, Washington
{: .center }

## References

{% bibliography --cited %}
