Story
=====

A strange observation that we came across when training ensembles is that more data does not always seem to improve performance. If we set random seeds for frame selection, and select an increasing number of frames based off of these random seeds, we can see the marginal improvement offered by a percentage increase in training frames:

.. image:: images/compare_models_groundtruth_output_consensus_70_90.png

Here each plotted point represents the consensus output of an ensemble of networks, trained with the data. The ensembles in this figure contain 4 networks for the 10/30% splits, and 2 networks for the 70/90% splits. However, since the outputs are confidence weighted, we would expect to see consistent performance. This decrease in performance from frame addition is true going from a 10% subset of the training data to a 30% subset, or a 70% subset to a 90% subset.    

This is an interesting effect, but how can we understand it better? One way would be to calculate the *Shapley Value* of each frame: we split all subsets of the training data into those that do and do not contain a given frame. We then look at the difference between the value of a function we care about based upon the average function value in subsets that contain the datapoint of interest and subsets that do not. We haven't done any normalization to match up to the Shapley Value computation, but we're averaging similar kinds of quantities: the average bias in the detection of animal keypoints, or the average change in the confidence estimate at the groundtruth location. 

When we do this model partitioning and calculate the keypoint bias, we see that certain training frames have a strong positive or negative impact on the classification of all other frames, when measured in terms of the increase or decrease in detected keypoint bias relative to the groundtruth: 

.. image:: images/influence_mat_ibl1_labeledpose_results_agg.png

In general, there is a long negative tail, meaning certain frames are able to strongly push the detection closer to the true value. However, there are still frames that increase the bias in detection by around ~20 pixels on average. Take, for example, frame 683 as it affects frame 775:    

.. image:: images/influence683_775_ibl1_labeledpose_results_agg.png

This particular frame seems to push the mass of detections it is a part of (blue) further away from the true location and spread it out.    
Is this a function of the fact that the groundtruth confidence of these values is decreasing? We can examine this idea by looking at the confidence at the groundtruth point: 

.. image:: images/influence_confidence_mat_ibl1_labeledpose_results_agg.png

We see here that the groundtruth confidence does not really seem to change as a function of adding or removing a particular datapoint. This is in contradiction to certain results we saw in the groundtruth when averaging across either just the 10 and 30 percent ensembles or the 70 and 90 percent ensembles. One point could be that by considering all ensembles, you are averaging over more uncertainty and removing the effect of outliers on the final detection.    

This suggests that individual frames are not decreasing the confidence of the groundtruth prediction, but more likely are increasing the confidence of distractors. We can check this by measuring not just the groundtruth confidence, but rather the groundtruth probability, normalized over the entire scoremap. 

We still need to figure out if there's something different about the different ensembles here. Why do some seeds give so much better results than others? 

We should also start looking into data augmentation to remove the effect of these distractors. Consider data augmentation to cut out the distractors near the target map. 

Idea: In general, each point of image data introduces many spurious correlations. Your ability to generalize might depend upon your datasets ability to account for the spurious correlations introduced in each individual datapoint. We still don't know why adding data hurts you sometimes: why does the first plot suggest some kind of history effect of data? is it possible that you're enforcing more spurious correlations than you're getting rid of? 

This is why data augmentation is a great idea. So but then, what is the value of ensembling and/or self supervision here? 
    - Ensembling lets you actually see what issues are bias, instead of variance in a neural network. 
    - an outlier is any point for which its spurious correlations are not correctly accounted for?   
    - also, is there an effect of training order?   



