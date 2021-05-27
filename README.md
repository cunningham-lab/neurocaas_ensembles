# Github repo for DGP ensembling on NeuroCAAS 

This repository extends the functionality of DeepGraphPose to work with ensembles of models. Although intended to be run in a NeuroCAAS immutable analysis environment, it can in priciple be used anywhere. 

## Installation 
It is highly recommended that you run this package with a GPU backed machine. While it should in theory be possible to run on a CPU, working with ensembles on CPU will be almost prohibitively slow. 
This repository depends upon the DeepGraphPose repository and its dependencies. First follow the instructions to create a virtual environment for DeepGraphPose, at https://github.com/paninski-lab/deepgraphpose. 

Then, once you have installed DGP, clone and install this repository: 

```
cd "{DGP_DIR}"
git clone https://github.com/cunningham-lab/neurocaas_ensembles.git
cd neurocaas_ensembles/src
pip install -e .
```

You should now be able to use the dgp_ensembletools python package inside your environment. You can check this by running the following in IPython, or any interactive console: 

```
>>> from dgp_ensembletools import models
```


## Using ensembling tools
To use the tools described here, we assume that you have a set of trained DGP models located in a single folder, as follows: 
```bash
/path/to/modelsfolder
|-{task}-{scorer}-{date}1 
  |-config.yaml
  |-dlc-models/
    |-iteration-0/
    ...
  |-labeled-data/
  |-training-datasets/
  |-videos/
    |-video1.mp4
  |-videos_pred/
    |-video1_labeled.mp4 
    |-video1_labeled.h5 
    |-video1_labeled.csv 
|-{task}-{scorer}-{date}2 
|-{task}-{scorer}-{date}3 
... 

```
Where it is assumed that dlc-models contains a trained model. 
Note that the name of the folders does not matter- we give the {task}-{scorer} naming scheme because it is automatically generated by DeepLabCut and DeepGraphPose usage. 

Assuming this structure, one can form an ensemble as follows: 
```
>>> ensemble = models.Ensemble(topdir="/path/to/modelsfolder",modeldirs=["{task}-{scorer}-{date}1","{task}-{scorer}-{date}2","{task}-{scorer}-{date}3",...],ext= "mp4",memory = joblib.Memory(location))
```

Note that the memory parameter is optional. By default, this package will try and save the output of running ensembling to disk using joblib, preventing us from recalculating scoremaps more often than necessary. If you would prefer not to have this happen, you can set memory = None. 

In order to generate the consensus output, this package assumes that you have already run prediction on a video. If you have not yet run prediction on a video, you can do so across the ensemble with:  
```
>>> [model.predict("path/to/video") for model in self.models.values()]
```
Which will run prediction for all models in the ensemble on a new video, and dump the video into the `videos_pred` folder of each model. 

Once this is done, you can get the consensus output by running:

```
>>> ref_x,ref_y = ensemble.get_mean_pose("video.mp4",range(startframe,endframe))
```
Where range specifies the frames over which you want to calculate the consensus, and the video name is the pur base name of the video.
The outputs `ref_x,ref_y` are numpy arrays giving the x and y positions, with shape (time,body part), and body part indices corresponding to the ordering of the body parts in the model configuration files. 

Code is documented further in docstrings at src/dgp_ensembletools/models.py- API reference forthcoming. 

If you experience any issues, please let us know at https://github.com/cunningham-lab/neurocaas_ensembles/issues ! 


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



