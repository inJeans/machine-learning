---
title: "What is ML: Why Use Machine Learning?"
teaching: 15
exercises: 15
questions:
- "What is machine learning good for?"
- "How can it help me with scientific discovery?"
objectives:
- "Explain the utility of machine learning in knowledge discovery and data mining."
- "Highlight interesting use cases of machine learning."
keypoints:
- "Machine learning is a path from data to knowledge."
- "Machine learning is generally concerned with making predictions on unseen data, without concern for why those predictions are made."
- "Machine learning is a programmable approach to knowledge, which enables an adpative hands-off approach to modelling and prediction."
---
## Data vs. Knowledge
In the era of "*Big Data*" and "*IoT*" we are in the fourtunate position of having too much data.
More than we know what to do with.
However, simply having a lot of data doesn't necessarilly help.
Machine learning is *a* process that enables us to extract knowledge from data.

![](../fig/data_to_knowledge.png)

## Predicition vs. Understanding

Earlier we discussed the distinction between machine learning and statistical inference, formally separating the two in terms of their objective.
In general, it is fair to say machine learning is predominantly concerned with being able to make predicitons on unseen data.
A goal that is completely distinct from wanting to understand how certain outcomes occur.
This is a very black and white characterisation of machine learning algorithms and it is more accurate to say that for both statistical inference and machine learning different approaches sit on a spectrum in between these two extremes, with machine learning biased more towards the prediction at all costs end.

## Interpretability

Particularly in todays machine learning ecosystem one criticism of ML is its inherent lack of interpretability.
Related to the point above, that machine learning is primarily concerned with prediction at all costs, ML will sacrifice the interpretability of a model in exchange for better prediction accuracy.
Typical of this trade off is deep learning.
Deep learning essentially forms the basis for much of the hype in machine learning and is the reason for so many of the successfull applications encountered in industry today.
Deep learning essentially builds, very large, layered networks which learn to build increasingly more complex representations of their input as you progress deeper.
Typical networks will have something on the order of tens of millions of parameters in state-of-the-art image processing architectures while modern networks for natural language understanding can be hundreds of millions.
This enormity and complexity of algorithm architcture makes them impenetrable in terms of understaning exactly what the network is doing or why it is doing it.
Interpretability is current a hot area of research within machine learning, particularly for application domains that are highly regulated.

![](../fig/interpretability.png)

> ## Discussion
>
> The value proposition for applying machine learning in business is pretty clear, you don't really care why something is happening you just want to be able to predict it so you can optimise your strategy for turning a profit.
> In science, it isn't so clear.
> Can you think of examples where the ability to make predictions without understanding might be useful?
> Or is science with knowledge of the underlying physics not really science?
> 
{: .discussion}

## Adpative

One of the most exciting aspects of machine learning is it's ability to be automated.
The fact that we are designing a system to learn in a completely automated way, without the need for explicit input from humans means we can build completely unmanned feedback loops.
We can design systems that evolve over time, learn from experience or simply adapt to drifts in the underlying distribution.

![](../fig/lifecycle.png)

## Applications

### Industry

- Search
- Facial recognition
- Virtual assistants

### Science

- Extreme weather identification
- Disease diagnosis
- Bioprediction

{% include links.md %}
