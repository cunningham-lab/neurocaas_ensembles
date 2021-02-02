"""
2021-01-25: extract labels from dataset to create new project
upon visualizing these labels, we can tell the quality of the labels

"""
#%%
import os
from pathlib import Path
import numpy as np
import h5py
from moviepy.editor import VideoFileClip
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
# read manual labels

base_dir = Path('/data/libraries/dgpp/apply_dlc/iblright-kelly-2021-01-03')
labeled_data_dir = base_dir/'labeled-data'
videos_dir =  base_dir /'videos'
full_dir = '/data/libraries/dgpp/apply_dlc'
raw_dir = Path(full_dir) / 'raw_data'
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
#%%
datasets = os.listdir(labeled_data_dir)
print(datasets)
#%%
# filter those with short
datasets_short= [dat_ for dat_ in datasets if '_labeled' not in dat_]
print(datasets_short)
check_labels = True
#%%
raw_datasets = []
for dataname in datasets_short :
    #%%

    data_files = os.listdir(str(Path(labeled_data_dir) / dataname))
    data_h5 = [dat_ for dat_ in data_files if (('.h5' in dat_) and not ('.backup' in dat_))]
    data_csv = [dat_ for dat_ in data_files if (('.csv' in dat_) and not ('.backup' in dat_))]
    if not(len(data_h5) == 1):
        #import pdb; pdb.set_trace()
        #pass
        continue
    assert len(data_h5) == 1
    raw_datasets.append(dataname)
    data_h5 = str(Path(labeled_data_dir) / dataname / data_h5[0])
    data_csv = str(Path(labeled_data_dir) / dataname / data_csv[0])
    data_video = str(Path(videos_dir) / (dataname +'.mp4'))
    print(data_h5, data_csv)
    df = pd.read_csv(data_csv)
    df = df.values
    import pdb; pdb.set_trace()
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
             'body_parts': body_parts}

    dict2 = {'keep': list(frame_names), 'discard':[]}
    #%%
    raw_dir_data = raw_dir / dataname
    if not os.path.exists(raw_dir_data):
        os.makedirs(raw_dir_data)
    np.save('{}.npy'.format(str(raw_dir_data /dataname)), dict1)
    np.save('{}.npy'.format(str(raw_dir_data /'dlc_labels_check')), dict1)
    copyfile(data_video, str(raw_dir_data /  (dataname +'.mp4')))

    #%%
    # check labels
    if check_labels:
        # make folder called images
        raw_img_data =  raw_dir_data / 'labeled-images'
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
            #plt.plot()
    clip.close()

    #%%
#%%
#   Make yaml file
raw_yaml = raw_dir / 'raw.yaml'
#%%

import yaml

dict_file1 = {'video_sets' : raw_datasets}
dict_file2 = {'bodyparts' : body_parts}
dict_file3 = {'skeleton': body_parts}

with open(raw_yaml, 'w') as file:
    #documents0 =yaml.dump("Project configuration",file)
    documents1 = yaml.dump(dict_file1, file)
    documents2 = yaml.dump(dict_file2, file)
    documents3 = yaml.dump(dict_file3, file)

print('Created raw dir and raw.yaml in {}'.format(raw_dir))
print('Update skeleton in raw.yaml!!')