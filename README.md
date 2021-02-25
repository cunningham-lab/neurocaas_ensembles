# Github repo for DGP ensembling on NeuroCAAS 

## Borrowed code from Kelly; combine with existing NeuroCAAS DGP instance. 

## Work log as of 2/22: 

Done some ensembling studies with the ibl1 data, training on different fractions of the whole dataset. The different ensembles are stored in s3 as follows: 

- job_ensemble-dgp_444:444:445: a real job, run with ensemble size = 4, on the full ibl data. 
- job_ensemble-dgp_444:444:456: a test job, run with ensemble size = 5, on the full ibl data. 
- job_ensemble-dgp_444:444:457: a test job, run with ensemble size = 5, on the claire's fish data. 
- job_ensemble-dgp_444:444:458: a real job, run with ensemble size = 5, on the claire's fish data. 
- job_ensemble-dgp_2_20_10p_test: a test job, run with ensemble size = 10 (reduced to 9 due to date handling), on 10 % of the ibl data. 
- job_ensemble-dgp_2_20_{1,3,5,7,9}0p_real: real job, run with ensemble size = 10 (reduced to 9 due to date handling), on 90 % of the ibl data. Together these form an ensemble of size 45, with random seed 52 to initialize the training frame selection. 
- job_ensemble-dgp_2_22_{1,3,5,7}0p_real: real job, run with ensemble size = 10 (reduced to 9 due to date handling), on 10,30,50,70 % of the ibl data. These were run to compensate for the above, where I thought that I was running these different splits. The random seed for frame selection corresponds to the number of frames in all cases.  
- job_ensemble-dgp_2_23_{1,3,5,7}0p_real: real job, run with ensemble size = 10 (reduced to 9 due to date handling), on 10,30,50,70 % of the ibl data. These were run to compensate for the above, where I thought that I was running these different splits. The random seed for frame selection was 52 in all cases.  
- job_ensemble-dgp_2_24_{3,5,7,9}0p_real: real job, run with ensemble size = 10 (reduced to 9 due to date handling), on 10,30,50,70 % of the ibl data. These runs were because we disliked the results with seed 52 (they looked too good). The random seed for frame selection was 5 in all cases.  



