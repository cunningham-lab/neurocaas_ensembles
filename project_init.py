"""
Initialize project, don't take into account cropping
TO DO: to take into account cropping you need to read other crop from original file
"""
import os
os.environ['DLClight'] = 'True'
import deeplabcut
import numpy as np
import pandas as pd
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from pathlib import Path
import numpy as np
from deepgraphpose.utils_data import local_extract_frames_md
from skimage.util import img_as_ubyte
from deepgraphpose.preprocess.get_morig_labeled_data import create_labels_md
import yaml
import sys
#"""
task = sys.argv[1]#"ibl1"
scorer = sys.argv[2]#"kelly"
date = sys.argv[3]#"2020-07-15"
basepath = str(sys.argv[4])
#import pdb; pdb.set_trace()
print('sys',sys.argv)
"""
task="iblright"
scorer="kelly"
date="2030-01-01"
basepath="/datahd2a/datasets/tracki/iblright/"
"""
#exit()
n_iter = 1000000
training_fraction = 1.0


#%% set up dirs
#basepath = '/home/ekb2154/data/libraries/dgp_paninski/data/'+task+'/'
working_directory = basepath + 'model_data/'
if not os.path.exists(basepath):
    os.makedirs(basepath)
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

with open(basepath+'raw_data/raw.yaml') as file:
    documents = yaml.full_load(file)

    for item, doc in documents.items():
        if item == 'video_sets':
            video_sets = doc
            print(item, ":", doc)
        if item == 'bodyparts':
            bodyparts = doc
            print(item, ":", doc)
        if item == 'skeleton':
            skeleton = doc
            print(item, ":", doc)

video_path = []
for v in video_sets:
    vname, vtype = v.rsplit('.',1)
    video_path += [basepath+'raw_data/'+vname+'/'+v]
print(video_path)
#%%
vtype ='.' + vtype

#%% create project folder
print("Creating project...")
config_path = deeplabcut.create_new_project(
    task, scorer, video_path, working_directory=working_directory, copy_videos=True, date=date,
    videotype=vtype,
)
print("Project created.")
print('config_path is ', config_path)

#%% read config
print("Reading config...")
cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))

#% edit config if needed
cfg["move2corner"] = False
cfg["TrainingFraction"] = [training_fraction]
cfg["cropping"] = False
cfg["dotsize"] = 5
cfg["batch_size"] = 10
cfg['pos_dist_thresh'] = 8
cfg["skeleton"] = skeleton
cfg["bodyparts"] = bodyparts
cfg["default_net_type"] = "resnet_50" #resnet_101
cfg["numframes2pick"] = None

#% rewrite config file
print("Rewriting config...")
deeplabcut.auxiliaryfunctions.write_config(config_path, cfg)

#%% Extract labeled frames from videos
print("Extracting labeled frames...")
numframes2pick = create_labels_md(config_path, video_path, scorer)
#cfg["numframes2pick"] = numframes2pick

#%% Check the labeled frames
print("Checking labels...")
deeplabcut.check_labels(config_path)
print("Labels checked.")

#%% Create training set
print("Creating train dataset...")
deeplabcut.create_training_dataset(config_path)
print("Train dataset created.")

#%% Edit pose config
print("Editing pose config...")
model_folder = auxiliaryfunctions.GetModelFolder(
    training_fraction, 1, cfg)
project_path = cfg['project_path']
pose_config_path = os.path.join(project_path, model_folder, "train/pose_cfg.yaml")
edits = {
    "global_scale": 1.0,
    "crop": False,
    "batch_size": 1,
    "save_iters": 100,
    "display_iters": 100 // 2,
    "multi_step": [[0.001, 10000],[0.005, 430000],[0.002, 730000],[0.001, 1030000]],
    "optimizer": "sgd",
    "cropratio": 0.0,
    "scale_jitter_up": 1.0,
    "scale_jitter_lo": 1.0,
    "pos_dist_thresh": 8
}

deeplabcut.auxiliaryfunctions.edit_config(pose_config_path, edits)
print("Pose config edited.")
print("Now you can start training!!!!")

#%%
