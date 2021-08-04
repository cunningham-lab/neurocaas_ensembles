Literature
==========

Extension to Pose Tracking
--------------------------

How would we generalize the AUM framework from a classification framework to a pose estimation framework? 
One insight here is that the "heatmap" formulation of object localization has been formulated as a multi-label classification problem: Each image is essentially fed into several different classification networks, where each classifier has (image size) categories, and the target is a spatially localized 15 pixel radius bump. Training is done via softmax cross entropy loss.  

One easy extension would be measuring AUM within a target area of the image output: asking how much of the output area contains signal, and tracking that over the course of training. This could be especially useful if we integrate into DeepGraphPose and apply such a method during selection of unlabeled frames. Selection of unlabeled frames during training is fundamentally a way to try and encourage stronger generalization for video-structured data, and would be interesting to study in the context of AUM as a technique that is sensitive to generalizability of data points.   

How can we analyze the fact that in pose tracking, you have to deal with "occluders" and "distractors"? i.e., you could end up learning features of the image that are not relevant for training outside of the edge cases that they represent? Maybe this would be a good place to look into *curriculum learning*. 
There appear to be many different approaches to curriculum learning, where the main issue is that of designing a curriculum correctly, so as to present inputs to the network at the right rate of difficulty increase. A few approaches include: 

  - Heuristic designation of easy and hard samples (Hinton, 2009)
  - Simultaneous training of a teacher and student network (Jiang et al. 2018)  
  - Direct training of weight parameters for classes/instances within the dataset (Saxena et al. 2019)  

    - Take a look at Figure 3! It looks like they learn object detection with occlusion, albeit in a different training paradigm! 
    - **Reference this if you end up doing curriculum learning**- they are the first to report improvements  

All of these may have something to offer, and can be extended to our data in the same spirit as AUM- note that AUM benchmarks against both MentorNet and Data Parameters on image classification, and removing datapoints that are mislabeled does better than either of these approaches. I'd rather not hand over curriculum design to gradients when AUM seems like a more concrete and interpretable solution.   


There are two general streams of literature for this work. The first is the identification of mislabeled data in a training set, for which the main paper that we reference is [Pleiss2020]_. The second is pose estimate via heatmap tracking in deep neural networks, for which I will not include references here.  

Another interesting branch of literature analyzes the training dynamics of Neural Networks as a key component of overall performance, as in [Allen-Zhu2020]_ and [Li2019]_. These papers analyze individual data samples, and ask how the structure of features contained in each sample contribute to the training performance. An interesting concept here is that of *memorization*- how easy is it to generalize from the features in any given data point to the rest of the training set? Geoff's paper picks out the most egregiously hard to generalize data points- those that are formally mislabeled based on the tension between the gradient signal provided by that data point, and all other data points. But what about those datapoints that introduce distractors for localization? Would it be better not to include these in training at all?  

Memorization is also studied in [Feldman2020]_, which estimates the effect of individual data samples via subsample estimates of memorization and influence functions. These subsampled estimates partition subsets of data into those that do and do not contain a certain training example, and measure the effect of "adding back in" that training example on estimating 1) it's own function evaluation, and 2) that of examples in the test set. This is also considered in [Jiang2021]_, which looks at a similar metric, but examines the behavior of the metric as a function of intrinsic dataset size. One interesting observation from this paper is the fact that as datasets get more difficult, subsets of increasing size become more and more representatative of the generalization performance across the whole dataset. 

Many of the papers we're looking at above look for a scalar ordering of the dataset. Notably, performance in the [Feldman2020]_ and [Jiang2021]_ papers increase uniformly as a function of increasing number of samples. Contrast this to Geoff's paper or Deep Double Descent, where you have well documented instances of data removal improving performance. 

Some Synthesis
--------------
Fundamentally, the goal of all of these papers is to say something about the *data distribution* and statistics of data from the distribution, based on how training samples relate to the data distribution by means of the samples and their training dynamics. I.e., being able to recognize outliers, being able to locate examples that are central to a cluster of a data distribution [Jiang2021]_, working off the assumption that the data distribution is long tailed [Feldman2020]_. This is a slightly different goal than semi-supervised approaches that seek to model the data distribution with independent latent variable models/representation learning, before applying some sort of training. These approaches look at the training [dynamics, results, subsamples] themselves to make insights into the structure of the data. 

It's still an open issue of how we should treat data examples that are not representative of the entire dataset. Should we remove outliers, as they reduce accuracy on test performance? Or rather should we keep them, because they represent samples in the long tail that cannot be learned any other way? Fundamentally, this seems like a question that comes down to the task you have in mind, and corresponding confidence estimates. 

How do we get these measures to somehow reflect CONFIDENCE? How should CONFIDENCE be modulated in the presence of outliers? Can data augmentation strategies like mixup help us here? 

An Unexplored Direction (July 26, 2021) 
---------------------------------------
One thing that varies widely between different datasets, and in general between tasks, is the conditional entropy of the data given the response, :math:`H(X\|Y)`. I.e., what is the amount of variability in the data conditioned on that data belonging to a particular response? Fundamentally, the usefulness of metrics proposed by these different papers will depend upon `H(X\|Y)`, as well as the variance in this term across different categories. This measure can also account for things like outliers, long tails, etc. A distinguishing aspect of pose tracking may be that `H(X\|Y)` may be much lower on average than with image classification- in image classification tasks, you generally have a wide ranging corpus of different looking data samples from which you must extract invariances. For animal pose estimation, that's also true in principle, but you may see data categories that you only observe once, or only in one stereotypical configuration (the tongue being out, for example). What's the right way to teach a network about the expectation of this per category data variance?  

Can we further break down this conditional entropy into a per-output term? I.e. :math:`H(X\|Y_i)`. This might be related to the *Information Bottleneck Principle* . However this seems like a whole fight that's not worth getting into (Andrew Saxe paper). 

Data Augmentation and Self Supervision
--------------------------------------
If we trust the idea that the issue is that certain frames do not yield generalization performance because they encode spurious correlations in the image data, an interesting thing to try would be cutout [DeVries2017]_ style image augmentation. By doing so, you can remove spurious correlations in the data simply by occluding the relevant distractors in a semi-random way. Targeted cutout around the target map could be a very useful thing to do. At the moment, the image augmentation applied by default in DeepLabCut is scaling, cropping and jitter, without these extended augmentation techniques (see nature protocols paper) [Mathis2019]_.   

Data Augmentation can also be learned, using methods like FastAutoAugment that combine a variety of data augmentation techniques with evaluation.
These automated strategies can be used to additionally evaluate self-supervised representation learning using instance-contrastive techniques.  
Look at Darrell's Self supervision papers: https://arxiv.org/pdf/2009.07724.pdf and https://arxiv.org/pdf/2103.12718.pdf
Take a look also at https://arxiv.org/pdf/1809.03576.pdf (OOD detection with self supervised ensembles) and https://link.springer.com/chapter/10.1007/978-3-642-38562-9_26. 

We should also think about how we can actually measure generalization: https://openreview.net/pdf?id=SJgIPJBFvH and the resulting recommendation of sharp minima based models: https://arxiv.org/pdf/1609.04836.pdf. Sharpness of minima is a function of the size of training set too? If so, this is part of the issue with using small datasets.  



Training in the Presence of Outliers
------------------------------------

Papers Library: Robust Optimization

Compared to AUM, there are a variety of other approaches to treating the training data based on the different assumptions that people have about the structure of this data. 
- For example, Shah et al. 2020 assume the presence of outliers as samples with a different optimal value of the loss function, and propose a method for SGD using samples with the lowest loss at any given training iteration in linear regression or shallow networks (theory for strongly convex losses). 
- In comparison, Katharololous et al. 2018 make no assumptions on the presence of outliers, but propose training deep nets with importance sampling based on an estimate of the Gradient Norm- data points should be sampled in proportion to their gradient norm after some burn in time, presenting hard examples to the network that will change its weights faster. This corresponds to hardcore work in strongly convex systems by Needel, Srebro, and Ward, and Zhao and Zang, and similar approaches like Loshchilov and Hutter, 2016, that propose using the HIGHEST estimate of the loss in importance sampling. 
- Finally methods like Curriculum Learning and Self Paced Learning (Kumar, Packer and Koller), Data Parameters, and Yixin's Bayesian Learning with outliers paper all suggest dealing with outliers/difficult examples via hidden variable models that can scale to big models. The first two (three?) also suggest that we should present easy examples first when training models.   
- In a stat mech analysis, Dietrich and Opper also suggest a hidden variable model based on EM and MAP estimates to detect outliers emitted from an unstructured noise distribution, that corrupts observations from an equal variance two component Gaussian mixture model. This paper suggests the occurrence of a phase transition in classifier behavior based on the proportion of outlier examples.   
- Finally, referencing the engineering literature, Bartlett and Wegkamp introduce a binary classifier with a "reject classification" option that is taken if the margin is not large enough. This literature has a connection to Supervised Deep InfoMax (Wang and Yiu 2019).   

The Gap (1): There are methods that analyze the effect of outliers in simple settings, and there are methods that provide prescriptions for training schedules for deep networks, but without considering outliers. There is not something that bridges this gap. 
The Gap (2): When should you give up and throw out some training data? Conventional wisdom would seem to say never- either you should see hard examples more often (importance sampling based on loss/grad norm) or you should only see them after other examples. AUM would disagree, as would our data. 
  Answer: when you don't have enough good data to reduce the effect of the bad data? (see Motivation, 10% vs. 30%)
  Alternatively: if you have outliers, what should you do? 
    A) add more good data (up to 50%)
    B) throw out bad data (down to 10%)
    In effect, these do similar things? 

Your data would address the question, is there a critical outlier density at which you're going to run into problems training with SGD? Is this a function of the density of outliers only, or rather the training order as well?  
  * What if most of your time is just spent reducing the effect of outliers? In Katharololous et al, are we really running the "hard" examples when we do importance sampling? Or is there some sort of oscillation between outliers and typical examples that's just going to take a while? 

.. [Pleiss2020] Pleiss et al. 2020, Identifying Mislabeled Data using the Area Under the Margin Ranking
.. [Allen-Zhu2020] Zhu and Li, 2020, Towards understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning 
.. [Li2019] Li et al. 2019 , Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks 
.. [Feldman2020] Feldman and Zhang 2020, What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation   
.. [Jiang2021] Jiang et al. 2021, Characterizing Structural Regularities of Data in Overparametrized Models   


