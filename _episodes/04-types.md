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

## The Experience, <img src="../fig/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="13.08219659999999pt" height="22.465723500000017pt"/>

Typically, the experience a machine learning algorithm encounters during learning is in the form of a dataset, or exposure to a dataset (or subset thereof).
A dataset is a collection of **examples**, each example comprising a set of **features** that have been quantitatively measured from some object or event.
We typically represent an example as a vector <img src="../fig/6696f7c13013a578e75270ab031d8208.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="49.48432829999999pt" height="22.648391699999998pt"/>, where each entry <img src="../fig/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="14.045887349999989pt" height="14.15524440000002pt"/> of the vector is another feature.
Broadly speaking, experiences are often categorised as either **unsupervised** or **supervised**.

>**Unsupervised learning algorithms** experience a datset containing many features, then learn useful properties of the structure of this dataset.
>
>**Supervised learning algorithms** experience a dataset containing features, but each example is also associated with a **label** or **target**. 
>
>Roughly speaking, unsupervised learning involves observing several examples of a random vector <img src="../fig/676087755dccba33776355f1e6acc5c8.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> and attempting to implicitly or explicitly learn the probability distribution <img src="../fig/73be1e3e5240a708063b27894a95a487.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="31.033115849999987pt" height="24.65753399999998pt"/>, or some interesting properties of that distribution; while supervised learning involves observing several examples of random vector <img src="../fig/676087755dccba33776355f1e6acc5c8.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> and an associated value or vector <img src="../fig/70f2b77490b3fb663196907887ef684a.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="10.239687149999991pt" height="14.611878600000017pt"/>, then learning the predict <img src="../fig/70f2b77490b3fb663196907887ef684a.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="10.239687149999991pt" height="14.611878600000017pt"/> from <img src="../fig/676087755dccba33776355f1e6acc5c8.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/>, usually by estimating <img src="../fig/a225ea344abe81ca5f2e16183fa3ba0f.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="48.57868124999999pt" height="24.65753399999998pt"/>.
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

## The Task, <img src="../fig/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="11.889314249999991pt" height="22.465723500000017pt"/>

Many kinds of tasks can be solved with machine learning. Some of the most common machine learning tasks include the following:

- **Classiﬁcation**: In this type of task, the computer program is asked to specify which of <img src="../fig/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.075367949999992pt" height="22.831056599999986pt"/> categories some input belongs to.
To solve this task, the learning algorithm is usually asked to produce a function <img src="../fig/4c2a7717f8e85a7e323af81623e4fe5d.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="140.16895694999997pt" height="24.65753399999998pt"/>. When <img src="../fig/e0fbbdca7d434da514590c69bb5e08fd.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="63.146798549999986pt" height="24.65753399999998pt"/>, the model assigns an input described by vector <img src="../fig/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> to a category identiﬁed by numeric code <img src="../fig/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="8.649225749999989pt" height="14.15524440000002pt"/>.
There are other variants of the classiﬁcation task, for example, where <img src="../fig/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.81741584999999pt" height="22.831056599999986pt"/> outputs a probability distribution over classes.
An example of a classiﬁcation task is object recognition, where the input is an image (usually described as a set of pixel brightness values), and the output is a numeric code identifying the object in the image.

- **Classiﬁcation with missing inputs**: Classiﬁcation becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided.
To solve the classiﬁcation task, the learning algorithm only has to deﬁne a single function mapping from a vector input to a categorical output. When some of the inputs may be missing, rather than providing a single classiﬁcation function, the learning algorithm must learn a set of functions.
Each function corresponds to classifying <img src="../fig/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> with a diﬀerent subset of its inputs missing.
This kind of situation arises frequently in medical diagnosis, because many kinds of medical tests are expensive or invasive.
One way to eﬃciently deﬁne such a large set of functions is to learn a probability distribution over all the relevant variables, then solve the classiﬁcation task by marginalizing out the missing variables.
With <img src="../fig/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.86687624999999pt" height="14.15524440000002pt"/> input variables, we can now obtain all <img src="../fig/e60b59f16d95417f1d9fef84963bb57c.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="16.34523329999999pt" height="21.839370299999988pt"/> diﬀerent classiﬁcation functions needed for each possible set of missing inputs, but the computer program needs to learn only a single function describing the joint probability distribution.

- **Regression**: In this type of task, the computer program is asked to predict a numerical value given some input.
To solve this task, the learning algorithm is asked to output a function <img src="../fig/6e98b7acdd117ccebfbe7816c5ef5633.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="81.77873054999999pt" height="22.831056599999986pt"/>.
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

- **Imputation of missing values**: In this type of task, the machine learning algorithm is given a new example <img src="../fig/5c642c8db2955a900121c010e1cc8a81.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="50.06645654999999pt" height="22.648391699999998pt"/>, but with some entries <img src="../fig/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="14.045887349999989pt" height="14.15524440000002pt"/> of <img src="../fig/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> missing.
The algorithm must provide a prediction of the values of the missing entries.

- **Denoising**: In this type of task, the machine learning algorithm is given as input a corrupted example <img src="../fig/a09921e101803fc6ed7d32869412f6c7.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="50.06645654999999pt" height="22.648391699999998pt"/> obtained by an unknown corruption process from a clean example <img src="../fig/5c642c8db2955a900121c010e1cc8a81.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="50.06645654999999pt" height="22.648391699999998pt"/>.
The learner must predict the clean example <img src="../fig/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="14.611878600000017pt"/> from its corrupted version <img src="../fig/ac0f0830227f117bc3a9563f08bc0af7.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.97711604999999pt" height="22.41366929999999pt"/>, or more generally predict the conditional probability distribution <img src="../fig/f7909eca1d90a7ff97e5b5f8dbec42ad.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="48.316111799999995pt" height="24.65753399999998pt"/>.

- **Density estimation or probability mass function estimation**: In the density estimation problem, the machine learning algorithm is asked to learn a function <img src="../fig/35eb17dc3482ab5fa2a8527623438fa0.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="115.58591384999997pt" height="22.648391699999998pt"/>, where <img src="../fig/09492c897c295d8bbd75c6c94e05dc24.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="66.38714445pt" height="24.65753399999998pt"/> can be interpreted as a probability density function (if <img src="../fig/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.39498779999999pt" height="14.15524440000002pt"/> is continuous) or a probability mass function (if <img src="../fig/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="9.39498779999999pt" height="14.15524440000002pt"/> is discrete) on the space that the examples were drawn from.
To do such a task well (we will specify exactly what that means when we discuss performance measures <img src="../fig/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="12.83677559999999pt" height="22.465723500000017pt"/>), the algorithm needs to learn the structure of the data it has seen.
It must know where examples cluster tightly and where they are unlikely too ccur.
Most of the tasks described above require the learning algorithm to at least implicitly capture the structure of the probability distribution.
Density estimation enables us to explicitly capture that distribution.
In principle, we can then perform computations on that distribution to solve the other tasks as well.
For example, if we have performed density estimation to obtain a probability distribution <img src="../fig/48a18a027893eb4fb7f5352c2d3e89a4.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="31.033115849999987pt" height="24.65753399999998pt"/>, we can use that distribution to solve the missing value imputation task.
If a value <img src="../fig/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="14.045887349999989pt" height="14.15524440000002pt"/> is missing, and all the other values, denoted <img src="../fig/568f8a917045afd01da5365d75f4828c.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="14.628015599999989pt" height="14.611878600000017pt"/>, are given, then we know the distribution over it is given by <img src="../fig/2e85ae431ae434dba95ef3d80cd08769.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="68.95357589999999pt" height="24.65753399999998pt"/>.
In practice, density estimation does not always enable us to solve all these related tasks, because in many cases the required operations on <img src="../fig/48a18a027893eb4fb7f5352c2d3e89a4.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="31.033115849999987pt" height="24.65753399999998pt"/> are computationally intractable.

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

## The Performance Measure, <img src="../fig/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="12.83677559999999pt" height="22.465723500000017pt"/>

To evaluate the abilities of a machine learning algorithm, we must design a quantitative measure of its performance.
Usually this performance measure <img src="../fig/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="12.83677559999999pt" height="22.465723500000017pt"/> is speciﬁc to the task <img src="../fig/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" style="display:inline" width="11.889314249999991pt" height="22.465723500000017pt"/> being carried out by the system.
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
