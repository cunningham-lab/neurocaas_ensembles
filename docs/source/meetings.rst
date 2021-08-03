Meeting Notes: JPC 6/30
-----------------------

- What is the marginal distribution of X across resampling of data for 10% and 30%? 

    - is it correlated with the presence of distractors? 

- Can we replicate this in a toy dataset? 

    - Let's say a mixture model- is there a phase transition where the presence of outliers is critically affecting the performance of a detector? 
    - Subsetting data trades off variance for speed. This is a tradeoff we're comfortable with (if unbiased) (SGD, Russian Roulette, etc.)
    - However, if you have outliers in your data, your variance blows up, or you may have subsamples that are in practice worthless. 

- Takeaways: 

    - Any time you can say something about ensembles/subsampling, that's a meaty question. 
    - What can we say here? Something about sampling with or without outliers. 
    - Application to SGD: Does the expected presence of outliers affect what size of minibatch you should choose? Look at this paper: https://arxiv.org/pdf/2001.03316.pdf 
    - We could scale an analysis from linear classification all the way to deep nets with AUM (use the AUM ranking as a guide?).   

- Todos:

    - [X] Classify all of your ibl data into clean and outlier frames for some given body part. In this case, outlier can just be are the hands overlapping.  
        - [X] Also classify based on whether the hand is in the holder or not.   
        - we classified based on holder, overlapping, and if the right hand was present or not.  
    - [ ] Resample your 10%, 30%, 50% datasets, and see if the presence of your outlier frames is what's causing issues. Check if that's true for DLC too! If so, that's an interesting question. Is there a critical dataset size at which the presence of outliers severely affects the output of network training?  
        - A good reason to make dlc and ensemble dlc work. 
        - [ ] Review how you are subsampling frames in ensemble-dgp, and mock locally.   
        - [ ] Make sure you have seeds that include outliers and those that don't.   
        - [ ] Estimate how many ensembles you need to run and associated cost based on power.   
        - [ ] Create DLC and Ensemble-DLC by ripping stuff out of dgp.   
    - [ ] correlate with AUM!

Ensembles, Subsampling and the Bias Variance Tradeoff.     

- Bias is :math:`\mathcal{E}(\hat{\theta})-\theta`
- Variance is :math:`\mathcal{E}((\mathcal{E}(\hat{\theta})-\theta)^2)`  

Idea: Increasing the size of the data seems to reduce the variance but increase the bias in the presence of outliers. 
Idea: The *density of outliers* is an important factor in determining model performance. Is this a potential reason why bootstrapping with ensembles fails? (some models get hit by lots of outliers)

Meeting Notes: JPC 7/30
-----------------------

Having discussed the influence function based analysis, we have a few different directions to take this project: 

- dgp + ensembling (this will happen through the PTL library anyway).
- dgp + ensembling + outlier frame self-supervision (just keep chugging ahead with what you have now)  
- the value of ensembling + outlier frame self supervision (provide a more general purpose tool)  
- the above, but with some clever computational tricks to avoid all the cost.   

One idea, inspired by Darrell 2020: self supervision to _remove_ data. In Darrell 2020, they use a self supervised cost that they show correlates well with downstream performance on various tasks. Does this cost correlate well with posture tracking? If not, can we find a self supervised cost that does? If so, we should be able to use it to evaluate the best way to remove frames as a form of data augmentation  

Start a library of papers that show that more data doesn't always help generalization. 
- Pleiss, 2020 (AUM)
- Feldman, 2020  

-----------------------
