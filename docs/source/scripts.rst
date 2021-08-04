Scripts
=======

This repository contains experiments with ensembles and pose tracking models in a low data regime. 

pre_analysis
------------

:code:`python pre_analysis.py --labellist FILE --dist1 FILE --dist2 FILE` 

This script takes as input a list of files heuristically labeled as containing outliers or not, as well as two sets of rmses: one corresponding to the rmses obtained by an ensemble without outliers, and one corresponding to an ensemble with outliers. We take the null hypothesis that the main distinguishing feature of these datasets is the presence of outlier or not, and that model performance is otherwise i.i.d. with respect to subsampling of training data.

We first calculate the proportion of 10% and 30% ensembles that should contain an outlier assuming random subsampling using hypergeometric distributions (binomial without replacement). 

Then, assuming a normal distribution for the rmses of models with and without outliers, we can state the number of samples we should collect (we will assume uniform across seeds). We then conduct a power analysis, calculating the number of ensembles that we need to run for a statistical significance of 0.05 given an effect size of 0.8.     
Finally, we calculate the cost of running these ensembles.  

The question now is how we should split this up by seed. Before, we had ensembles with 9 networks each. The benefits of this are that you get a lot more ability to recover the right answer in a single ensemble, but the costs are that you don't get as much training sample diversity. *To answer the question, are the poorly performing networks due to outliers we should reduce the ensemble size.* Noted, it's still nice to see consistency within a single training batch. Given that we have 23/17 networks for the current power analysis, I think a good choice would be 10 seeds total, with 4 models per ensemble, to get 40 ensembles with an estimated 23/17 split of outlier to non outlier networks. *We should choose seeds to sample from that roughly obey this split in numpy*

Issue: We expected around 75% of the samples to not contain outliers given the hypergeometric distribution based derivation. In reality, it is closer to 90%, with 80% if we count only the samples from the 10% distribution. SOLVED: We were not calculating percentages, but rather with 10 and 30 samples. 

Diversion: What is the probability that I will draw 0 red marbles from an urn with 10 marbles, of which 2 are red if I draw 5 marbles? How can I break up the hypergeometric pmf into two separate evaluations for this problem?

Answer: you were adding sequential evaluations, instead of multiplying them! 

get_seeds
---------

:code:`python get_seeds.py --labellist FILE --seed number (multiple)`

This script looks at the labeled data (for the presence of outliers), and a set of random seeds. Each random seed is used to subselect data from the set of labeled data, and we determine if the corresponding 10 and 30% splits contain outliers or not.  

The script was run with a pickled list and seeds as follows: [2,7,12,17,22,27,32,37,42,52] to generate 7 subsamples without outlers, and 13 with. Of these, seed 27 generated the only 30% split with outliers. This might be something to keep in mind. We can proceed with these seeds though.  

UPDATE 7/21. We dug into deepgraphpose and found that it was using a different ordering of frames before applying selection. With this selection, we would only see two subsamples without outliers. Instead, use samples: 2,27,42,52,89,102,122,142,162,7700 

compare_models_groundtruth
--------------------------

:code:`python compare_models_groundtruth.py --ensemblesfolder FILE --labellist FILE --groundtruth FILE`

returns:

    - Plotted distribution of rmses (`neurocaas_ensembles/images/compare_models_groundtruth_output_{groundtruth,ensembles}.png`) 
    - Plotted distribution of rmses split between outliers and nonoutlier including ones (`neurocaas_ensembles/images/compare_models_groundtruth_output_split_{groundtruth,ensembles}.png`)

Generates functions `determine_outliers` (determine if a particular random seed and frame count will have outliers), `parse_modelname` (given a string describing the name of an ensemble, parse out its relevant detauls and save a dictionary of info), `get_training frames` (determine the training frames used by a particular ensemble (direct from the model, not from the seed and frames))
#TODO `get_training_frames` can be turned into a method of the ensemble (and the trained model). Include a warning if the model's recorded seed and frames parameters don't expect the actual training frames seen. 

This script looks at a set of ensembles, having training sets chosen with different random seeds, and a careful accounting of when the training sets contain outliers or not. It then plots the rmse of each individual model on a shared test set that disjoint from all training data. The results are shown below:   

.. image:: images/compare_models_groundtruth_output.png

Points marked with an X represent the rmse of models that were trained with outliers in their training set. Points marked with an O represent the rmse of models that were trained without outliers in their training set. Furthermore, the left column represents models that were trained with 5 frames, while the right column represents models that were trained with 17 frames. An interesting observation is that seeds where the 5 frame split does not contain an outlier seem to do worse/ equivalently when you add 12 more frames. In contrast, seeds where the 5 frame split does contain an outlier already seem to do better when you add more frames. We can see this by splitting by seed into two graphs: one where 5 frame split contains outliers, and one where it does not:   

.. image:: images/compare_models_groundtruth_output_split.png

In order to quantify this effect, let's calculate the consensus per ensemble, and draw a direct relationship between the consensus detection for a single seed at 5 frames and at 17 frames:    

calculate_consensus
-------------------

:code:`python calculate_consensus.py --video-name STRING --groundtruth FILE --partperm STRING --labellist FILE --basefolder DIR --resultfolder DIR` 
returns: 

    - (json) a series of json files containing dictionaries that encode the individual and median performance of different ensembles: `resultfolder/{modelname}_result.json`. 

This script uses `parse_modelname`, `get_training_frames` from the `compare_models_groundtruth` script. 

This script applies the dgp_ensembletools code to ensembles we constructed newly. The results are saved as json files represnting the per network and median performance (`data/ibl/consensus_performance`)
This is one of the messiest scripts- especially when it comes to running multiple different kinds of ensembles. We have an except statement at line 20-23 that handles the case of one or two different neurocaas runs separately, which is not ideal. 

plot_consensus
--------------

:code:`python plot_consensus.py --labellist FILE --ensemblesfolder DIR --resultsfolder DIR`
returns: 
    - (image) the consensus performance for all ensembles with which we can calculate a consensus. Taken by looking at the json files located in `resultsfolder` as well as the ensembles given in `ensemblesfolder`. Note that we've symlinked all of the models into a directory `pose_results_agg` for the purpose of easy comparison. If you want to adapt this script, you still need to give all the different frames you're calculating for. 

This script imports the `determine_outliers` function and `ensemble_template` string from `compare_models_groundtruth`. It creates its own version of `parse_modelname`
#TODO - can these be unified? The difference is between line 71 of the original script and line 24-25 here: Here we assume that our inputs are result.json files in the first argument- string.we can at the very least unify the return: given frames, and seed? It determines what we want to do downstream with this.   

This script plots the results of the previous script, i.e. consensus performance from each ensemble. 

.. image:: images/compare_models_groundtruth_output_consensus.png

NB: the lines indicate pairs of ensembles that share the same seed, i.e. the same subset of 5 training frames. 
We can see more clearly the trends visible in the per-model data- adding frames seems to make performance worse for all but two ensembles. Let's try and investigate what makes these ensembles improve, when all the others do not? 

UPDATE 08/04:
We extended this trend with two model ensembles at 70 and 90 percent of the training data. We see some surprising trends here: 

.. image:: images/compare_models_groundtruth_output_consensus_70_90.png

It looks like the 70% split actually improves performance more reliably than the 90% split does. What's going on here?    

What's next? What should come next is an analysis of the function space, but also an analysis of the individual datasets. If we look at the 17 frame set, it looks like there are three distinct groups of performers. Check this out. Likewise, what distinguishes the two ensembles where you do see improvement from adding more data?   
Also, your characterization of outliers is heuristic at this point. We can improve on this! 

create_data_inclusion_matrix
----------------------------

:code:`python create_data_inclusion_matrix.py --video_name STR --groundtruth FILE --partperm STR --labellist FILE --basefolder DIR --resultsfolder DIR`

This script imports `parse_modelname` from `plot_consensus` (the version with results), and `get_training_frames` from `compare_models_groundtruth` (determine training frames).  

This script considers the ensembles that you've created, and then creates a matrix showing the individual datapoints that are included in each ensemble's training data. It creates two plots: first, a matrix of the ensemble training data and relevant test frames in a random order, and a second with the ensemble seeds sorted according to RMSE (best at the top) and the frames sorted by increasing index. We show this second output here:   

.. image:: images/DataInclusionMatrix_Sortedibl1_labeled.mp4.png

It's hard to tell if there are any immediate patterns based on this data alone, but we can start to ask questions like: what training dataframes actually worsten performance when they're added to the training set. This information might also be useful to set up a regression problem against each network's performance. That's what we should try next: regress the errors made by each network onto the training frames that different networks have access to. Alternatively, first do pca on the error profile and then regress those pca weights onto these datapoints.    

create_pca_model
----------------

:code:`python create_pca_model.py --labellist FILE --groundtruth FILE --ensemblesfolder DIR`

This model is pretty strongly coupled to compare_models_groundtruth (many variables imported) but we don't necessarily care about pursing the analysis further. If we were to, clean up.

This script takes the individual models that you have trained, and evaluates not the RMSE, but the per-frame deviation of the predicted pose output from the groundtruth on all test frames. Once it has collected the per-frame deviation across all models, it flattens the deviation (across xy and body parts) into a single feature vector and performs PCA. The transformed data, model, and labels are stored into a model, `pca_with_labels`. 

plot_pca
--------

:code:`python plot_pca.py --modelpath FILE --nb-parts INT`

This script takes the output of the previous one, and plots interesting features of the PCA model output. 
This script also takes in `colors` and `markers` variables from compare_models_groundtruth. It might be good to standardize formatting for plotting in a separate file. 

.. image:: images/pcafig.png

In this plot, the top left panel shows the variance explained ratio of the top five PCs. We can see that there is a significant concentration of the Variance Explained in the top principal components, which is again perhaps surprising given previous results in ensembling.    

The top right panel shows part of the first principal component vector, reshaped to represent the error of the first body part (the mouse's paw) in XY space. We see that all of the errors are in a particular direction (the X deviation is always positive, and the Y deviation is always negative). 
The bottom right panel shows the same data for the second principal component vector. We see similar locations of deviation, suggesting there could be redundancy in the representation of deviation at individual frames between different PCs.

The bottom left panel shows the distribution of individual models in the PC space. The black X represents the projection of the groundtruth (i.e. 0 deviation everywhere) into PC space: distance from this black X probably correlates with increasing error, although there could be interactions between the differen PCs at the error level (even if the vectors themselves are orthogonal) . Each individual color represents a particular training frame selection seed, and the size of the marker indicates the number of training frames (small = 5, large = 17). We see that in the first two PCs, we see a clustering of some of the 17 frame models in the bottom right hand quadrant of the space, which is interesting.  

estimate_influence
------------------

:code:`python estimate_influence.py --video-name STR --groundtruth FILE --partperm STR --labellist FILE --basefolder DIR`

returns: 
    - (image) marginal change in bias as a function of training frame inclusion (`neurocaas_ensembles/images/influence_mat_{videoname,ensemblefolder}`)
    - (image) marginal change in variance as a function of training frame inclusion (`neurocaas_ensembles/images/influence_mat_var_{videoname,ensemblefolder}`)
    - (image) marginal change in standard error as a function of training frame inclusion (`neurocaas_ensembles/images/influence_mat_se_{videoname,ensemblefolder}`)
    - (pickled dict) frame bias, and raw data saved as `neurocaas_ensembles/script_outputs/influence_data_{videoname,ensemblefolder}`.  

This script takes functions `get_training_frames` and `parse_modelname` from the `compare_models_groundtruth` script. 

This is our first attempt to estimate an influence function across our small training set. It's still not clear what the best way to do this for our case is, so right now what we're measuring is the magnitude of the average deviance and standard error of that deviance from groundtruth as calculated from a set of traces corresponding to models that HAVE seen a particular frame, and those that have not. Quantitatively, we are measuring the bias, variance, and standard error of different models. Given a set of trained networks, :math:`\{\phi\}`, We define these quantities in terms of two ensembles of networks, :math:`\{\phi\}_{i}` and :math:`\{\phi\}_{\i}`, corresponding to those networks that contain training frame :math:`x_i` and those that do not. Furthermore, for each video frame :math:`x_i`, and corresponding part detection :math:`y_i`, with both representing vectors. :math:`\phi(x_i)` represents a given network's approximation of :math:`y_i`.  Given these two quantities, we define the delta bias and standard error as follows: 

.. math::

   \Delta bias_{i}(x_j) = \mathbb{E}_{\{\phi\}_{i}}[\|\phi(x_j)-y_j\|_2] - \mathbb{E}_{\{\phi\}_{\i}}[\|\phi(x_j)-y_j\|_2]

   \Delta se_{i}(x_j) = \frac{\sigma_{\{\phi\}_{i}}[\|\phi(x_j)-y_j\|_2]}{\sqrt{|\{\phi\}_{i}|}} - \frac{\sigma_{\{\phi\}_{\i}}[\|\phi(x_j)-y_j\|_2]}{\sqrt{|\{\phi\}_{\i}|}}

Where :math:`\sigma` is the standard deviation. I.e., if seeing a particular training frame :math:`x_i` improves prediction on a test frame :math:`x_j`, you should see a negative :math:`\Delta bias` and/or :math:`\Delta se`, and if seeing it makes performance worse, you should see a positive values for these quantities. Some open question to this point are:

  1) How are these bias and se measures related to quantities in the bias/variance tradeoff?  
  2) What if we combine this approach with ensembling? Would this be better? 

Just from our first run, we see some interesting things estimating the change in bias in ensembles that do or do not contain a given test frame. Each of these matrices gives a per training and test frame estimate of the quantities above: 

.. figure:: images/influence_mat.png
   :width: 800

   :math:`\Delta bias_{i}(x_j)` for all training frames :math:`x_i` (rows) and all test frames :math:`x_j` (columns)

Each of these matrices represents one body part detection, with the influence values for each training frame (indexed as columns) on the entire labeled video (indexed as rows).   
Note that influence values range from the positive (i.e. adding that frame increases bias), to the strongly negative (adding that frame decreases bias). Furthermore, note that there are certain regions of the full dataset that seem largely insensitive to most of the frame exclusions, and there are other stereotyped portions that are susceptible in a stereotyped way. These susceptible regions can be positively or negatively influenced. However, in general it looks like frames have a consistent overall effect, as helping or hurting prediction across the entire test dataset, and are even consistent across different body parts. 

The bias and variance changes would be interesting to study together, but it's difficult to compare when our include and exclude subsets are of different sizes. Here's the change in the standard error of the mean instead: 

.. figure:: images/influence_mat_se.png
   :width: 800

   :math:`\Delta se_{i}(x_j)` for all training frames :math:`x_i` (rows) and all test frames :math:`x_j` (columns)

analyze_influence
-----------------

:code:`python analyze_influence.py --ensembledict FILE --framedir DIR --videopath PATH`
returns:
    - (image) visualization of memorization factors for all training frames at `neurocaas_ensembles/scripts/script_outputs/memorization{trainframe}_{data_id}.png`
    - (image) visualization of highest influence pairs for each training frame at `neurocaas_ensembles/scripts/script_outputs/influence{trainframe}_{testframe}_{data_id}.png`


The measures of :math:`\Delta bias` and :math:`\Delta se` can be tied into the memorization/influence framework of Feldman 2020. To begin with, we can analyze memorization and influence in terms of bias:  

.. math::

   mem(x_i) = \Delta bias_{i}(x_i)

   inf(x_i;x_j) = \Delta bias_{i}(x_j) 

    
This script plots the memorization values and the largest influence values of different training frames (and some corresponding test time frames). For each training frame, it takes the the change in bias and standard error of keypoint estimates as output by the previous file (`influence_data`), and visualizes it. The resulting files are stored in the directory `scripts/script_outputs`, with `memorization{}.png` files giving estimates of memorization based on the change in bias and change in standard error, and `influence{}_{}.png` files giving estimates of influence of one file upon another. These are in general quite interesting. The memorization estimates almost all demonstrate (as expected) that seeing a particular training frame improves the estimate of that training frame's output location. A particularly singular example is frame 25:  

.. image:: images/memorization25.png
   :width: 800

In red, you can see the outputs of networks that were trained without seeing frame 25. In blue, you see the outputs of networks trained with frame 25. Each panel shows the outputs for a different body part detection, and the change in bias and standard error reflect measurements of difference between the blue and red point clouds. Frame 25 is probably the most memorized training frame, as demonstrated by the strongly negative change in bias and standard error as a function of the frame's inclusion in the training set.    

Correspondingly, we can measure the *influence* of one training frame upon a separate test frame by splitting all the networks that we have trained into those that do/do not include that test frame, and seeing how their predictions differ on a given test frame (this is like Feldman, 2020 for object classification). For each training frame, we only save out the test frame for which it has the highest (positive or negative) influence.

Frame 25, with a high memorization value, also has a very strong influence, reducing bias on some test frames: 

.. image:: images/influence25_105.png
   :width: 800

We can see examples a training frame with a strong bias increasing influence. First, here's a case where correct labels get deflected to the wheel position: 

.. image:: images/influence43_231.png
   :width: 800

Here's another case where correct labels get deflected to the other paw:

.. image:: images/influence439_431.png
   :width: 800


In this latter case, it appears that including a training frame in which the two hands appear in the same plane drags many network detections to the wrong hand in a nearby frame, increasing the output bias and standard error relative to a distribution of detections generated by networks that have not seen this training frame. There exist other examples of training frames like this, where it appears that the wheel could be acting as a labeling distractor as well. These are troubling issues, because they are native to scientfic analyses where you will have strong correlations between the body parts of interest and other relevant features. In effect, we need to teach the network to understand that these correlations are spurious, or to correctly adjust its confidence to reject the associations learned by these spurious correlations. Why, however, do not all networks that see this training example get confused? One network at least is able to correctly localize the paw despite having seen this frame. Is this a frame ordering effect?     

- Caveats: 

Some of the ensemble splits here are quite small: in the Frame 43 example above, there were only four networks that included that frame, and these effects do not adequately account for performance changes from frame correlations in the course of subsampling. 

compute_influence_confidence
----------------------------

:code:`python compute_influence_confidence.py --video-name STR --groundtruth FILE --partperm STR --labellist FILE --ensemblesfolder DIR`
returns:

    - (joblib object) joblib pickled dictionary of diferent models and their estimation of the groundtruth confidence `neurocaas_ensembles/scripts/script_outputs/confidence_data_{videoname,ensemblesfolder}`. 

This script takes the `get_training_frames` and `parse_modelname` functions from `compare_models_groundtruth`

Note that Feldman 2020 calculates influence values based on the probability of being correct or not averaged over model training specifications, and not necessarily taking into account the uncertainty of the network itself. What if we used our heatmap outputs to determine an influence function that accounts for the uncertainty of an individual model instead? 

To get higher resolution on how individual training frames affect the confidence of the network, we also calculate influence not using bias and variance, but using heatmaps. We calculate a score at the groundtruth marker location at each frame, and determine how this changes based on the ensemble that we choose. Note that this is a much more "local" measure of the influence of a frame around the groundtruth output, in that it only cares about how much the heatmap value of the groundtruth location changes as a function of frame inclusion or exclusion. 

analyze_influence_confidence
----------------------------

:code:`python analyze_influence_confidence.py --video-name STR --confidences FILE --labellist FILE --ensemblesfolder DIR`
returns: 

    - (image) matrix showing the influence of each training frame on each output frame due to its inclusion in the training set or not `neurocaas_ensembles/images/influence_confidence_mat_{videoname,ensemblesfolder}`. 
    - (image) Per-training frame histogram of the influences across the entire test set for a given frame `neurocaas_ensembles/images/influence_confidence_frame{i}_hist_{videoname,ensemblesfolder}`.   
    - (pickled dict) dictionary giving processed confidence data as differences `neurocaas_ensembles/scripts/script_outputs/delta_confidence_data_{videoname,ensemblesfolder}` 


This script takes the `get_training_frames` and `parse_modelname` functions from `compare_models_groundtruth`

We analyze the outputs of the previous file here. It's interesting to compare the values that we see here to those that we got from measuring the bias and variance in terms of the maximum position. It looks less clear that certain frames are contributing to/detracting from performance across the board, although we should quantify this to see. We can also see more widespread effects here- whereas the bias effects were limited to individual bands of the test video, it looks like the confidence effects can be seen across a wider range. 

.. image:: images/influence_confidence_mat.png
   :width: 800

It's interesting to interpret the effects of individual frames, like frame 25. Compared to the bias estimates, where it was clearly helping all four of the body parts we looked at, it's not clear that the groundtruth is becoming any more likely due to frame 25, just that the paw was localized to an incorrect (but still closer than otherwise) location. We can also examine the distribution of influences (positive and negative) per frame- we see that each frame calculated this way has some distribution of influences, with some positive and some negative entries. The other interesting thing to note is that calculated this way, it doesn't really look like there are bad frames: there are certainly some frames that detract from the confidence estimates of the ground truth, but they don't seem that bad. Consider that this is because what the bad frames do is that they drastically INCREASE the likelihood of a distractor, without necessarily altering the likelihood of the groundtruth. This gives us some insight into the nature of the input to an uncertainty estimate- adding more training examples adds to the heatmap output, and does not seem to detract from already detected features. 

Note: Feldman 2020 also observes negative influence values (it's in their pre-computed data: https://pluskid.github.io/influence-memorization/) at the very least in their CIFAR data, even looking only at influence between points within the same class assignment. They only consider positive influence datapoints in their study however because their curious about long tails of similar subgroups. It would be cool to make a histogram of the per datapoint influence matrix here and see if we can see some detractor datapoints in the classification data too.  




