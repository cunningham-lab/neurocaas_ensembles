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
