Diary 
-----

Work log for all things that don't have to do with a particular script. 

July 21, 2021
-------------
The goal for today is to check out the issue you're having where your seed generates different values on your local and remote machine. 

We can check this several ways. Maybe the easiest is to take your pose_aum repo, and run it on the remote to see if seeds are different. Install your pose_aum repo on an ensembledgp instance, and rerun scripts with the seeds you have. if you get something that looks good, who cares? you're good to go. 
+ first trying in an independent env pose_aum -> same as local 
+ try install directly into the dgp env too. -> same as local  
+ so it's probably the code in the deepgraphposerepo itself to select frames.   

+ Next, we should try getting the project folder locally, and then running the same code. Of suspicion is the file "dlc_labels_check.npy", which is what is being indexed into. 
+ It turns out that in the inital processing of the project folder, "neurocaas_ensembles/read_manual_labels.py", we are creating a file dlc_labels_check.npy, which has a keep attribute used to generate new data. This is for some reason seemingly randomly ordered. We have saved the ordering as "pose_aum/data/ibl/unorderedlist", a pickled object. 
  
Conclusion: we created a new pickled object, ordered_classified_list, that respects the order of samples used by deepgraphpose to sample frames. We can look in read_manual_labels to dig further into the way the ordering is generated. 

July 22, 2021 
-------------
How should we evaluate models that are trained on different subsets of the data? Even though we have 10 different training data splits, it should still really only be a small fraction of the total data. First, take the intersection of all of them to generate your eval set. 

+ We have plot for this now, documented in the script compare_models_groundtruth_output. Tomorrow, we'll evaluate the consensus detections, and try a pca analysis to look at the function behavior. 
+ in order to recreate your env on a remote server, you should first install the dgp package, then on top of that neurocaas_ensembles, then on top of that the requiements for pose-aum.  

+ An interesting angle here is that a small data adjustment of a pretrained imagenet is some sort of simplified model, that is an interesting and relevant use case. Imagenet models with small adjustments. 

+ Try adding more models to each ensemble. 2 is a bit small after all, but 4 is pretty good. Just resubmit with a different timestamp.   

July 23, 2021  
-------------
We added more models to the ensemble, and looked at the corresponding results. it's pretty interesting to note that in most cases, the error increases. 
Question: can we find the frames that are causing issues out of the different subsets of frames that we have selected for? If so, could we identify such frames ahead of time, and exclude them from the training process/ come up with some curriculum that handles them correctly? 

What data is actually useful for generalization? AUM paper says not all of it.  

July 26, 2021
--------------
We have our dataset inclusion matrix. We can then phrase our problem as matrix factorization: a binary data inclusion matrix times some matrix B gives the performances. We can learn basis functions then for the different frames and see if this gives us any predictive performance (bootstrap estimates). 

Is it a good idea to consult your training dataset for a particular output category when you're looking at further data? I.e. how similar is this new data to other data you've seen in terms of the network's intrinsic representation? Is this too computationally costly? This could be a good way to characterize the entropy of different output categories. 

July 27, 2021
-------------
Imagine that we make an ensemble that consists of all of these networks together. How does this compare to a network trained on all of the training frames? 

August 3, 2021
--------------
Figured out issues with heatmap code, and replotted. Plotted the per training frame influence histograms as well for the heatmap- would be good to do again but for the probabilities, normalized, not just confidence output. 

Note: Feldman 2020 also observes negative influence values (it's in their pre-computed data: https://pluskid.github.io/influence-memorization/) at the very least in their CIFAR data, even looking only at influence between points within the same class assignment. They only consider positive influence datapoints in their study however because their curious about long tails of similar subgroups. It would be cool to make a histogram of the per datapoint influence matrix here and see if we can see some detractor datapoints in the classification data too.  

August 4, 2021
--------------

In scripts so far, the functions `get_training_frames` and `parse_modelname` see a lot of reuse. It's definitely worth moving these into the source code and testing. For get_training_frames, it's clear how we might do that (include as a method of the model/ensemble), less so for `parse_modelname`. Also look into free env variables, and replace them with default arguments for the scripts. 

The result of calculating bias and SE shows that there's definitely still an effect if we look at 40 and 52 frames of data. 
- [X] An important sanity check point now is to calculate the rmse, and show that it's actually consistent with better performance that we saw in `compare_models_groundtruth` when measured with RMSE. 
  - We calculated the RMSEs here. It looks like there are 16 networks or so that perform poorly- this corresponds to the number of "delinquent networks" that we see. The better performance that we observe at higher training data regimes is true in bulk.   
  - Takeaway 1: you still see this effect at higher frames, but in this case it's tied to the fact that some networks (surprisingly) still fail here. An argument for going higher.   
- [ ] Once you've done the sanity check, go ahead and also calculate based on heatmaps. It would be cool to see heatmap changes and influence function changes.    
  - run compute_influence_confidence.py on the new data. 
- [ ] What about if we make statements across all seeds?
  - run compute_consensus, then take jsons and plot together.   
- [D] What if we restrict our study to 90% ensembles only?   

0. [X] Pull git repo on new instance. 
1. [X] Pull new data to instance. 
2. [X] Run compute_consensus.    
   take json outputs and scp to local, then run plot_consensus on local. + github push.  
3. [X] Run compute_influence_confidence.    
   calculate influence matrices and histograms. 

Observations: 
  - Biggest takeaway is that increasing the number of training frames by one order of magnitude did not help. 
  - some frames seem notably worse for performance at 70 and 90% splits. Maybe this is because their value has been absorbed by other frames that do not detract so strongly from performance. 
  - The effects we see at 10 and 30% we definitely also see at 70 and 90%. Some frames are just bad. In fact, at 70 and 90% more frames seem bad. What do we make of this?   
  - Start looking into data augmentation techniques.   
  - Start looking into frame removal.   
  - BUT: importantly, what if you have a sampling bias at either end?   
  - We can combine the estimates. This gives us something more like the Shapley value of each individual frame. We see that there are still biases and frames that increase bias, but that the magnitude of their effect is much smaller than we would have thought just looking at the 10/30 ensembles or the 70/90 ensembles.   
  - Furthermore, you don't see a negative effect of these frames on the probability of the groundtruth point.   
  - I.e. if they are having an effect, it's by increasing the probabiblity of other points away from the groundtruth.   
  - When you combine the estimates, you see the bias distribution shift strongly negative. Many frames seem to offfer a very positive contribution, although some continue to generate big biases. What could this reduction in negative magnitude of the bias be coming from?   
    - Consider the effect of a few networks that just go crazy- in this context, the negative impact of a single network will be overestimated. 
    - The takeaway is that there is still a negative effect.  
