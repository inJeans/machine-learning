---
title: "What is ML: A Taxonomy of Machine Learning"
teaching: 15
exercises: 15
questions:
- "What is the difference between classification and regression?"
- "What is the difference between supervised and unsupervised learning?"
- "How can I quantify machine learning algorithm performance?"
objectives:
- "Understand the landscape of machine learning algorithms"
- "Use this understanding to identify the appropriate type of algorithm to use for a given problem."
- "Understand the importance of performance metrics."
keypoints:
- "There are a number of machine learning algorithms available, which one you use depends on the type of data you have, the problem you are trying to solve and your definition of 'what is good'."
- "There is typically more than one way to solve a problem, usually it depends on how you frame what you are doing."
---
![Machine learning Taxonomy](../fig/sklearn-taxonomy.png "ML Taxonomy")
*[scikit-learn algorithm cheat sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)*

The following taxonomy draws heavily from Chapter 5, Machine Learning Basics in [(Goodfellow, Bengio, & Courville, 2016)](#goodfellow2016)

## The Experience, $E$

Typically, the experience a machine learning algorithm encounters during learning is in the form of a dataset, or exposure to a dataset (or subset thereof).
A dataset is a collection of **examples**, each example comprising a set of **features** that have been quantitatively measured from some object or event.
We typically represent an example as a vector $x \in \mathbb{R}^{n}$, where each entry $x_i$ of the vector is another feature.
Broadly speaking, experiences are often categorised as either **unsupervised** or **supervised**.

>**Unsupervised learning algorithms** experience a datset containing many features, then learn useful properties of the structure of this dataset.
>
>**Supervised learning algorithms** experience a dataset containing features, but each example is also associated with a **label** or **target**. 
>
>Roughly speaking, unsupervised learning involves observing several examples of a random vector $\bf{x}$ and attempting to implicitly or explicitly learn the probability distribution $p(\bf{x})$, or some interesting properties of that distribution; while supervised learning involves observing several examples of random vector $\bf{x}$ and an associated value or vector $\bf{y}$, then learning the predict $\bf{y}$ from $\bf{x}$, usually by estimating $p\left(\bf{y}\vert\bf{x}\right)$.
{: .quotation}
[(Goodfellow, Bengio, & Courville, 2016)](#goodfellow2016)

The types of experiences are not necessarilly mutually exclusive.
Often times a single problem may involve the use of either one of the above techniques, most likely both and potentially a hybrid of the two.

For completeneness, when characterising the types of experiences available to a machine learning algorithm we will also include **reinforcement learning**.
Reinforcement learning algorithms work with a dataset that is not necessarilly fixed, these algorithms interact with their environment such that there is a feedback loop between the learning system and its experiences.

> ## Challenge
>
> Try to identify the type of experience for each of the examples below.
> 
> 1. A set of holiday images taken from Flickr with there associated locations.
> 2. A set of satellite images continuosly collected across the globe.
> 3. A time series of temperatures recorded across a range of sites.
> 4. A game of Go.
> 5. A stream of news artciles.
> 
>> ## Solution
>>
>> 1. Supervised.
>> 2. Unsupervised.
>> 3. Supervised.
>> 4. Reinforcement.
>> 5. Unsupervised.
>> 
> {: .solution}
{: .challenge}
> ## Discussion
>
> What type of datasets (experiences) have you worked with in the past?
> Are there any unique experiences you can identify in your domain that might be applicable to a learning algorithm?
> 
{: .discussion}

## The Task, $T$

Many kinds of tasks can be solved with machine learning. Some of the most common machine learning tasks include the following:

- **Classiﬁcation**: In this type of task, the computer program is asked to specify which of $k$ categories some input belongs to.
To solve this task, the learning algorithm is usually asked to produce a function $f:\mathbb{R}^{n}\to \{1, \dots , k\}$. When $y=f(\bf{x})$, the model assigns an input described by vector $\mathbf{x}$ to a category identiﬁed by numeric code $y$.
There are other variants of the classiﬁcation task, for example, where $f$ outputs a probability distribution over classes.
An example of a classiﬁcation task is object recognition, where the input is an image (usually described as a set of pixel brightness values), and the output is a numeric code identifying the object in the image.

- **Classiﬁcation with missing inputs**: Classiﬁcation becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided.
To solve the classiﬁcation task, the learning algorithm only has to deﬁne a single function mapping from a vector input to a categorical output. When some of the inputs may be missing, rather than providing a single classiﬁcation function, the learning algorithm must learn a set of functions.
Each function corresponds to classifying $\mathbf{x}$ with a diﬀerent subset of its inputs missing.
This kind of situation arises frequently in medical diagnosis, because many kinds of medical tests are expensive or invasive.
One way to eﬃciently deﬁne such a large set of functions is to learn a probability distribution over all the relevant variables, then solve the classiﬁcation task by marginalizing out the missing variables.
With $n$ input variables, we can now obtain all $2^{n}$ diﬀerent classiﬁcation functions needed for each possible set of missing inputs, but the computer program needs to learn only a single function describing the joint probability distribution.

- **Regression**: In this type of task, the computer program is asked to predict a numerical value given some input.
To solve this task, the learning algorithm is asked to output a function $f:\mathbb{R}^{n} \to \mathbb{R}$.
This type of task is similar to classiﬁcation, except that the format of output is diﬀerent.
An example of a regression task is the prediction of the expected claim amount that an insured person will make (used to set insurance premiums), or the prediction of future prices of securities.
These kinds of predictions are also used for algorithmic trading.

- **Ranking**: Sometimes, instead of estimating an absolute numeric value, we want to be able to learn relative positions.
For example, in a recommendation system for movies, we want to generate a list ordered by how much we believe the user is likely to enjoy each.

- **Transcription**: In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form.
For example, in optical character recognition, the computer program is shown a photograph containing an image of text and is asked to return this text in the form of a sequence of characters (e.g., in ASCII or Unicode format).
Another example is speech recognition, where the computer program is provided an audio waveform and emits a sequence of characters or word ID codes describing the words that were spoken in the audio recording.

- **Machine translation**: In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.
This is commonly applied to natural languages, such as translating from English to French.

- **Structured output**: Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the diﬀerent elements.
This is a broad category and subsumes the transcription and translation tasks described above, as well as many other tasks.
One example is parsing—mapping a natural language sentence into a tree that describes its grammatical structure by tagging nodes of the trees as being verbs, nouns, adverbs, and so on.
Another example is pixel-wise segmentation of images, where the computer program assigns every pixel in an image to a speciﬁc category. 
The output form need not mirror the structure of the input as closely as in these annotation-style tasks.
For example, in image captioning, the computer program observes an image and outputs a natural language sentence describing the image.
These tasks are called structured output tasks because the program must output several values that are all tightly interrelated.
For example, the words produced by an image captioning program must form a valid sentence.

- **Anomaly detection**: In this type of task, the computer program sifts through a set of events or objects and ﬂags some of them as being unusual or atypical.
An example of an anomaly detection task is credit card fraud detection.
By modeling your purchasing habits, a credit card company candetect misuse of your cards.
If a thief steals your credit card or credit card information, the thief’s purchases will often come from a diﬀerent probability distribution over purchase types than your own.
The credit card company can prevent fraud by placing a hold on an account as soon as that card has been used for an uncharacteristic purchase.

- **Synthesis and sampling**: In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data.
Synthesis and sampling via machine learning can be useful for media applications when generating large volumes of content by hand would be expensive, boring, or require too much time.
For example, videogames can automatically generate textures for large objects or landscapes, rather than requiring an artist to manually label each pixel.
In some cases, we want the sampling or synthesis procedure to generate a speciﬁc kind of output given the input.
For example, in a speech synthesis task, we provide a written sentence and ask the program to emit an audiowaveform containing a spoken version of that sentence.
This is a kind of structured output task, but with the added qualiﬁcation that there is no single correct output for each input, and we explicitly desire a large amount of variation in the output, in order for the output to seem more natural and realistic.

- **Imputation of missing values**: In this type of task, the machine learning algorithm is given a new example $\mathbf{x} \in \mathbb{R}^n$, but with some entries $x_i$ of $\mathbf{x}$ missing.
The algorithm must provide a prediction of the values of the missing entries.

- **Denoising**: In this type of task, the machine learning algorithm is given as input a corrupted example $\tilde{\mathbf{x}} \in \mathbb{R}^n$ obtained by an unknown corruption process from a clean example $\mathbf{x} \in \mathbb{R}^n$.
The learner must predict the clean example $\mathbf{x}$ from its corrupted version $\tilde{\mathbf{x}}$, or more generally predict the conditional probability distribution $p\left(\mathbf{x} \vert \tilde{\mathbf{x}}\right)$.

- **Density estimation or probability mass function estimation**: In the density estimation problem, the machine learning algorithm is asked to learn a function $p_{\text{model}}:\mathbb{R}^n \to \mathbb{R}$, where $p_{\text{model}}(\mathbf{x})$ can be interpreted as a probability density function (if $x$ is continuous) or a probability mass function (if $x$ is discrete) on the space that the examples were drawn from.
To do such a task well (we will specify exactly what that means when we discuss performance measures $P$), the algorithm needs to learn the structure of the data it has seen.
It must know where examples cluster tightly and where they are unlikely too ccur.
Most of the tasks described above require the learning algorithm to at least implicitly capture the structure of the probability distribution.
Density estimation enables us to explicitly capture that distribution.
In principle, we can then perform computations on that distribution to solve the other tasks as well.
For example, if we have performed density estimation to obtain a probability distribution $p(\mathbf{x})$, we can use that distribution to solve the missing value imputation task.
If a value $x_i$ is missing, and all the other values, denoted $\mathbf{x}_{−i}$, are given, then we know the distribution over it is given by $p\left(x_i\vert \mathbf{x}_{-i}\right)$.
In practice, density estimation does not always enable us to solve all these related tasks, because in many cases the required operations on $p(\mathbf{x})$ are computationally intractable.

Of course, many other tasks and types of tasks are possible.
The types of tasks we list here are intended only to provide examples of what machine learning can do, not to deﬁne a rigid taxonomy of tasks.

> ## Challenge
>
> Try to identify the type of task for each of the examples below.
> 
> 1. Estimate required steering wheel angle given an image from a dash-cam.
> 2. Predict the rating a user might assign a particular movie, given a handful of ratings from other movies and users.
> 3. Identify potentially malicious traffic in a computer network.
> 4. Convert page layout sketches into functioning html.
> 5. Identify the sub-surface structure based on sensor readings.
> 
>> ## Solution
>>
>> 1. Regression.
>> 2. Imputation.
>> 3. Anomoly detection
>> 4. Translation.
>> 5. Classification.
>> 
>{: .solution}
{: .challenge}
> ## Discussion
>
> Are any of these tasks applicable to datasets that you have?
> Would any of these tasks solve some interesting science questions you have?
> 
{: .discussion}

## The Performance Measure, $P$

To evaluate the abilities of a machine learning algorithm, we must design a quantitative measure of its performance.
Usually this performance measure $P$ is speciﬁc to the task $T$ being carried out by the system.
For example, tasks such as classiﬁcation, classiﬁcation with missing inputs, and transcription, we often measure the **accuracy** of the model.

Usually we are interested in how well the machine learning algorithm performson data that it has not seen before, since this determines how well it will work when deployed in the real world.
We therefore evaluate these performance measures using a **test** set of data that is separate from the data used for training the machine learning system.

> ## Discussion
>
> What are the useful metrics of performance for some of the tasks you identified above?
> Are they easy to capture or express mathematically?
> 
{: .discussion}

---

> ## Challenge
>
> 1. Assume we are given the task of building a system to distinguish healthy crops from unhealthy crops.
> What is in an unhealthy crop that lets us know that it is unhealthy?
> How can the computer detect an unhealthy crop through image analysis?
> What would we like the computer to do if it detects an unhealthy crop?
> 
> 2. Write the phrase "data school" ten times on a piece of paper.
> Also ask a friend to do the same.
> Analysing these twenty images try to find features, types of strokes, curvatures, loops how you make dots, and so on, that discriminate your handwriting from that of your friends.
> 
> 3. In estimating the price of a used car, it makes more sense to estimate the percent depreciation over the original price than to estimate the absolute price.
> Why?
{: .challenge}

## References

[Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Retrieved from https://www.deeplearningbook.org](https://www.deeplearningbook.org)<a name="goodfellow2016"></a>

{% include links.md %}
