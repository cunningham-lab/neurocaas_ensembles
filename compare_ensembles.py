"""
Plot traces for ensembles and compare the outputs
Fix dgp_plot so that it actually does what it is supposed to do
Actually use ensembles for comparison
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ['DLClight'] = 'True'
from deepgraphpose.models.eval import load_pose_from_dlc_to_dict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from moviepy.editor import VideoFileClip
from matplotlib import animation
import sys
#video_name = 'left_movieflip'
#video_name = 'right_movie'
video_name = str(sys.argv[1])

task="iblfingers"
full_dir = "/share/home/ekb2154/data/datasets/tracki/{}/model_data/".format(task)
scorer="kelly"
snapshot ='snapshot-step3-final--0'

# new dates for ensemble
ranges = range(1,5)
dates= ["2030-01-0{}".format(ii) for ii in ranges]
run_names= ["run {}".format(ii) for ii in ranges]
frame_range = np.arange(0, 1000).astype('int')
n_frames = frame_range.size
moviefile = "/home/ekb2154/data/libraries/dgp_paninski/etc/ensembles/iblvideos/" + video_name + '.mp4'
outname = '{}{}'.format(task,video_name)
figname = 'comparison_{}.mp4'.format(outname)

#%% make video comparing these 4 traces:
xrs,yrs = [], []
for date_ in dates:
    project_dir = Path(full_dir)/ "{}-{}-{}".format(task,scorer, date_)
    print("\n{}".format(project_dir))
    video_files = os.listdir(project_dir / "videos_pred" )
    video_files = [video_file_ for video_file_ in video_files if '.csv' in video_file_]
    video_files = [video_file_ for video_file_ in video_files if video_name in video_file_]
    video_files = [video_file_ for video_file_ in video_files if snapshot in video_file_]
    #import pdb; pdb.set_trace()
    assert  len(video_files) == 1
    #print(video_files)
    label_file = str(project_dir / "videos_pred" / video_files[0])
    #label_file = project_dir / "videos_pred" / ("{}_labeled{}.csv".format(video_name, snapshot))
    # load labels
    print('using ', label_file)
    labels = load_pose_from_dlc_to_dict(label_file)
    xr = labels['x']  # T x D
    yr = labels['y']  # T x D

    xr = xr[frame_range, :]
    yr = yr[frame_range, :]
    xrs.append(xr)
    yrs.append(yr)

#%%
nj = xr.shape[1]
num_traces = len(xrs)
bodyparts = ['part {}'.format(ii) for ii in range(nj)] # replace w real name
#%%
print('Init video')
video_clip = VideoFileClip(str(moviefile))
fps = video_clip.fps
n_frames = min(n_frames, np.ceil(video_clip.fps * video_clip.duration).astype('int'))
frame_init = frame_range[0] / fps
frame_stop = frame_range[-1] / fps
video_clip = video_clip.subclip(0, (n_frames / fps)) # it didnt take into account the range
#video_clip = video_clip.subclip(frame_init, frame_stop) # it didnt take into account the range

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({  # 'font.sans-serif' : 'Helveltica',
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.titlesize': 11})

#%%
# coment out y 
num_coordinates = 1 # x and y or just 2
fig2 = plt.figure(constrained_layout=False, figsize=(11, 10))
widths = [2, 7]
heights = np.ones(num_coordinates * nj)
num_cols = 2
num_rows = num_coordinates*nj
spec2 = fig2.add_gridspec(ncols=num_cols, nrows=num_rows, width_ratios=widths,
                          height_ratios=heights)

framerows = num_rows// num_traces
#%% Plot
markersize = 2
markersize2 = 5
from matplotlib.lines import Line2D
color_class = plt.cm.ScalarMappable(cmap="cool")
colors = color_class.to_rgba(np.linspace(0, 1, nj))
markers = Line2D.filled_markers[:num_traces]

#%%
frame_0name = 0
frame = video_clip.get_frame(0)
tidx = 0
colors_traces= ['C{}'.format(ii) for ii in range(num_traces)]

imagesfig = [[]]*num_traces
imagesmakers = [[]]*num_traces*nj
xtracesmarkers = [[]]*num_traces*nj
ytracesmarkers = [[]]*num_traces*nj
frametitles = [[]]*num_traces

count= 0
count1 = 0
for ss_idx in range(num_traces):
    #import pdb; pdb.set_trace()
    # for the image
    f2_ax1 = fig2.add_subplot(spec2[ss_idx*framerows:ss_idx*framerows+ framerows, 0])
    if ss_idx == 0:
        frametitles[ss_idx] = f2_ax1.set_title('Frame {} \n {} '.format(frame_0name, run_names[ss_idx]),
                                               color=colors_traces[ss_idx], weight='bold')
    else:
        frametitles[ss_idx] = f2_ax1.set_title('{}'.format(run_names[ss_idx]),
                                               color=colors_traces[ss_idx], weight='bold')

    imagesfig[ss_idx] = f2_ax1.imshow(frame)

    for njj in range(nj):
        dxs_dgp = xrs[ss_idx][tidx]
        dys_dgp = yrs[ss_idx][tidx]
        dgp_frame, = f2_ax1.plot(dxs_dgp[njj], dys_dgp[njj], c=colors[njj],
            marker=markers[ss_idx], ms=markersize, )

        imagesmakers[count] = dgp_frame
        count+=1

    # x traces
    for njj in range(nj):
        f2_ax2 = fig2.add_subplot(spec2[num_coordinates * njj, 1:])
        f2_ax2.set_yticks([])
        f2_ax2.set_xticks([])
        
        # plot trace
        f2_ax2.plot(np.arange(xrs[0].shape[0]), xrs[ss_idx][:, njj], c=colors_traces[ss_idx], linewidth=1.5, linestyle='--')

        dgp_mtx, = f2_ax2.plot(tidx, dxs_dgp[njj], c=colors_traces[ss_idx], marker=markers[ss_idx], markersize=markersize2,
                               markerfacecolor=None)
        f2_ax2.set_ylabel('x')
        f2_ax2.set_xlim([0, n_frames])
        f2_ax2.set_title('{}'.format(bodyparts[njj]), color=colors[njj], weight='bold')
        xtracesmarkers[count1]= dgp_mtx
        count1+=1

allmyparts = imagesfig + imagesmakers + xtracesmarkers
    
plt.tight_layout()

def init():
    return allmyparts

def animate(tidx):
    frame = video_clip.get_frame(tidx * 1.0 / fps)
    for ss_idx in range(num_traces):
        if (ss_idx == 0) or (ss_idx == num_traces-1):
            pass
            #print('ss', ss_idx)
        dxs_dlc = xrs[ss_idx][tidx,:]
        dys_dlc = yrs[ss_idx][tidx,:]
        if ss_idx == 0:
            frametitles[ss_idx].set_text("Frame {} \n {} ".format(tidx, run_names[ss_idx]))
        imagesfig[ss_idx].set_array(frame)

    count = 0
    count1 = 0
    for ss_idx in range(num_traces):
        for nji in range(nj):
            dxs_dlc = xrs[ss_idx][tidx, nji]
            dys_dlc = yrs[ss_idx][tidx, nji]
            imagesmakers[count].set_data(dxs_dlc, dys_dlc)
            xtracesmarkers[count1].set_data(tidx, dxs_dlc)
            count +=1
            count1 +=1

    return allmyparts

print('NFRAMES +{}'.format(n_frames))
anim = animation.FuncAnimation(fig2, animate,  init_func=init,
                               frames=n_frames, interval=20, blit=True,
                              repeat=False)

#%%
import time
start = time.time()

anim.save(figname)

print(time.time() - start)
plt.close(fig2)
#%%
video_clip.close()
