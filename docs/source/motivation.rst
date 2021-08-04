Motivation
----------

The motivation for this project comes from two observations I made in the course of setting up ensembling methods for pose tracking.

The first observation was that there are definite cases where adding more training examples leads to worse performance:

.. image:: ./images/rmseplot_smoothmedian.png
   
.. image:: ./images/rmseplot_smoothmedian_same_seed_5.png

In these two cases, we gradually add to a dataset and record the performance of ensembles trained on this data using RMSE. We see that going from a 10% split to a 30% split, there is actually a consistent decrease in performance within the ensemble. While this is an extremely small dataset (5 and 12 frames, respectively), we see similar issues looking at the actual detections: 

.. image:: ./images/sample_traces_fixed_seed_10%.png
   
.. image:: ./images/sample_traces_fixed_seed_30%.png

.. image:: ./images/sample_traces_fixed_seed_50%.png

.. image:: ./images/sample_traces_fixed_seed_70%.png

.. image:: ./images/sample_traces_fixed_seed_90%.png

See that in this case, there are a couple challenging periods (frames 100-200, 750-850) where the performance of a few networks actually seems to diverge more and more as we add more training data. Examining a corresponding video suggests that these are periods when the animal's left paw is retracted, and the right paw is more visible. 
   
The effects of bad training data have been reported in much larger datasets, in the context of image classification. In particular, Geoff thinks that one reason for the SOTA performance of AUM training, by simply removing data points over other approaches that involve some sort of curriculum learning (Data Parameters, MentorNet) is that they never really recover from the effect of bad samples early in training. Similar "sample non-monotonicity" has been reported as an effect of model capacity (Deep Double Descent), but in this case its probably much more an effect of the actual sample quality than anything else.      

Hypothesis: the "glitches" that we see in keypoint detection networks are due to the network memorizing the features of occluders/distractors in the training set. This might be less of an issue if we have lots of model capacity and lots of training time (Deep Double Descent Phenomenon Paper: Nakkiran et al.), but it still seems to be an issue now.    

The second observation was that against common wisdom in the ensembling literature, the errors of individual models within our ensemble were highly correlated: 

.. image:: ./images/errorspace_same_seed.png

Here we're quantifying the errors made at each frame in a 1000 frame video by 45 different networks (trained on 10-90% of the data, strict subsets). Calculating the deltas between the ground truth marker and the network outputs gives us a set of 45 time series representing the performance of each network on all body parts. In doing so, we see that most of the errors across all 45 networks can be captured in the top two PC dimensions (~80%). Furthermore, it looks like most networks within a given ensemble make similar errors to a greater or lesser extent. These effects could be in part due to initialization of the ResNet portion of our networks from ImageNet, or the presence of salient distractors/occluders that are seriously biasing performance. The question then becomes, what is driving some networks in an ensemble to perform better than others? While it could be the randomly initialized part localization and location refinement portions of the network, the correlations in the errors would suggest to me that we should look at the minibatch ordering as well.       

Hypotheses
----------

- Hypothesis: The issue is that in scientific datasets, you have a very controlled setting within which you are learning examples. This naturally leads to *spurious correlations* between the keypoints you're interested in tracking and aspects of the natural scene. While some of these correlations can be controlled (location in training frame, image properties), others are much harder to control (vicinity and similarity to outliers). How does the network uncertainty behave as a function of these outliers? We would hope that the reported confidence responds in proportion to the number of examples that it sees. You cannot necessarily break these correlations, because you are recording behavior of the animal and tracking that animal's behavior is important. You could deal with this issue by:
  
  - changing the optimization objective (ERM)
  - changing the target (more custom target maps)
  - leveraging confidence estimates
  - changing the dataset (for the correlation representations)
  - changing the training schedule  
- Hypothesis: training could be improved by determining a curriculum by which to show the network training examples. This curriculum is what makes certain ensemble members robust to noise, and others not. 
- Hypothesis: When you have a few frames to adjusta pretrained network, most frames are memorized, and others are useless. This can be detected via influence functions. 
- Hypothesis: We can predict the pattern of errors made by a network given a certain set of test frames. 

Solutions
---------

Together, these two hypotheses motivate me to look at keypoint detection in neural networks using tools like AUM. Here are some potential solutions: 

- We could use AUM to first locate data points that are considered "difficult". One option would be to then selectively shrink the detection targets of difficult examples, to minimize the effect of occluders on learned part filters.
- A closely related idea would be to apply something like cutout and remove the spurious correlations that we see in the data. Selectively remove areas around the target map and replace them with something. You could even occlude part of the target map and turn off the target map around it.   
- Another option would be to do some sort of curriculum training, based on the AUM ranking (correct ranking still to be determined).
- A third option would be to treat this as a statistical problem: the memorized, low likelihood occluded/distractor features are being overrepresented in the training set relative to the good features, and our network is just acting on the observed statistics of this data. This would also motivate some sort of curriculum based training based on the occurrence ofdifferent kinds of occluders.  
- A different path to success: identify the effect of bad training data in as much generality as you can,/give a nuanced understanding of when this data is good/bad in terms of the task at hand and the stakes (cost of making an error). Then, propose a proxy to these influence functions to detect bad training data efficiently, and show effectiveness in improving performance.   
- Another option: dynamically subset bootstrapped ensembles to identify the effects of different data frames in real time, or something close to it.   





