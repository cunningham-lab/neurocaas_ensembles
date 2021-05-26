"""
2021-01-03 : read a dlc project to check quality
2021-01-25: read a dlc project to make multiple copies
videos must be .mp4
"""
#%%
import os
os.environ['DLClight'] = 'True'
from pathlib import Path
import numpy as np
import h5py
from moviepy.editor import VideoFileClip
from shutil import copyfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# read manual labels
import sys
#"""
base_dir = Path( sys.argv[1])
full_dir = Path(sys.argv[2])
vtype = sys.argv[3]
if not vtype.startswith("."):
    vtype = "."+vtype
print('sys',sys.argv)
"""
base_dir = Path('/data/datasets/tracki/iblright-kelly-2021-01-03')
full_dir = '/datahd2a/datasets/tracki/iblfingers'
"""
labeled_data_dir = base_dir/'labeled-data'
videos_dir = base_dir /'videos'
raw_dir = Path(full_dir) / 'raw_data'
FROM_DGP_RUN = True # from DGP RUN it already has skeleton information
CHECK_LABELS = True

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

#%%
datasets = os.listdir(labeled_data_dir)
#print(datasets)
#%%
# filter those with short
datasets_short= [dat_ for dat_ in datasets if '_labeled' not in dat_]
datasets_short= [dat_ for dat_ in datasets_short if '.DS_Store' not in dat_]
print('\n\n --* Datasets in folder *---\n\n')
print(datasets_short)
#%%
raw_datasets = []
for dataname in datasets_short :
    #%%
    data_files = os.listdir(str(Path(labeled_data_dir) / dataname))
    data_h5 = [dat_ for dat_ in data_files if (('.h5' in dat_) and not ('.backup' in dat_))]
    data_csv = [dat_ for dat_ in data_files if (('.csv' in dat_) and not ('.backup' in dat_))]
    #%%
    if not(len(data_h5) == 1):
        print(len(data_h5))
        #import pdb; pdb.set_trace()
        #pass
        continue
    assert len(data_h5) == 1
    raw_datasets.append(dataname + vtype)
    data_h5 = str(Path(labeled_data_dir) / dataname / data_h5[0])
    data_csv = str(Path(labeled_data_dir) / dataname / data_csv[0])
    data_video = str(Path(videos_dir) / (dataname +vtype))
    print(data_h5, data_csv)
    df = pd.read_csv(data_csv)
    df = df.values
    #import pdb; pdb.set_trace()
    dfxy = df[2:, 1:].reshape(df.shape[0] - 2, -1, 2)
    dfxy = dfxy.astype(np.float)
    # dfxy = dfxy[:, :, :-1]
    xr = dfxy[:, :, 0]
    yr = dfxy[:, :, 1]
    #lr = dfxy[:, :, 2]
    xr = np.transpose(xr, [1, 0]) # 8 x 25
    yr = np.transpose(yr, [1, 0])
    #lr = np.transpose(lr, [1, 0])
    # in some cases the likelihoods are bad. This is probably because the labels should not be there
    #mask_data = lr
    n_labels, n_labeled_frames = xr.shape
    body_parts = list(df[0][1::2])
    print(body_parts)
    #%%
    frame_names = np.asarray([int(frame_name_.split('/')[-1].rsplit('.')[0][3:]) for frame_name_ in df[2:,0]]) # (8,)
    assert frame_names.size == xr.shape[1]
    #%%
    # Make 2 dictionaries
    clip = VideoFileClip(data_video)
    fps = clip.fps
    duration = clip.duration
    n_total_frames = int(np.ceil(duration*fps))
    #%% # Initialize the array
    xr_total = np.zeros((n_labels, n_total_frames))
    yr_total = np.zeros((n_labels, n_total_frames))
    xr_total[:,frame_names]= xr # D x T
    yr_total[:,frame_names]= yr
    # make two files
    dict1 = {'xr': xr_total, 'yr': yr_total, 'frame_indices': frame_names,
             'body_parts': body_parts,
             'keep': list(frame_names)}

    dict2 = {'keep': list(frame_names), 'discard':[]}
    #%%
    raw_dir_data = raw_dir / dataname
    if not os.path.exists(raw_dir_data):
        os.makedirs(raw_dir_data)
    np.save('{}.npy'.format(str(raw_dir_data /dataname)), dict1)
    np.save('{}.npy'.format(str(raw_dir_data /'dlc_labels_check')), dict1)
    copyfile(data_video, str(raw_dir_data /  (dataname +vtype)))

    #%%
    # check labels
    if CHECK_LABELS:
        # make folder called images
        raw_img_data = raw_dir_data / 'labeled-images'
        if not os.path.exists(raw_img_data):
            os.makedirs(raw_img_data)
        for frame_name in frame_names:
            frame_name_sec = frame_name*1.0/fps
            frame_ = clip.get_frame(frame_name_sec)
            fig, ax = plt.subplots(1,1)
            ax.imshow(frame_)
            ax.plot(xr_total[:,frame_name], yr_total[:,frame_name], 'o', markersize=5)
            ax.set_title('Frame {}'.format(frame_name))
            plt.tight_layout()
            plt.savefig(str(raw_img_data / ('frame_{}.png'.format(frame_name))))
            plt.close()
    clip.close()
    #%%
#%%
#   Make yaml file
raw_yaml = raw_dir / 'raw.yaml'
#%%
if FROM_DGP_RUN:
    os.environ['DLClight'] = 'True'
    from easydict import EasyDict as edict
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    config_path = str(base_dir / "config.yaml")
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(config_path))
    cfg_dict = edict(cfg)
    skeleton = [list(i) for i in cfg_dict['skeleton']]
    #import pdb; pdb.set_trace()
    # the
    assert len(body_parts) == len(cfg['bodyparts'])
    #body_parts = cfg['bodyparts']
else:
    skeleton = body_parts
#%%

#%%
import yaml

dict_file1 = {'video_sets' : raw_datasets}
dict_file2 = {'bodyparts' : body_parts}
dict_file3 = {'skeleton': skeleton}

with open(raw_yaml, 'w') as file:
    #documents0 =yaml.dump("Project configuration",file)
    documents1 = yaml.dump(dict_file1, file)
    documents2 = yaml.dump(dict_file2, file)
    documents3 = yaml.dump(dict_file3, file)

print('Created raw dir and raw.yaml in {}'.format(raw_dir))
print('Check skeleton in raw.yaml!!')

#%%
