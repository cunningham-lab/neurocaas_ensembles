## Test per-model code. 
import numpy as np
import pytest
import matplotlib.pyplot as plt
import dgp_ensembletools.models
import os

loc = os.path.abspath(os.path.dirname(__file__))

class Test_Ensemble():
    def test_init(self):
        topdir = os.path.join(loc,"../","data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
    def test_get_video_clip(self):
        topdir = os.path.join(loc,"../","data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        clip = ens.get_video_clip("ibl1_labeled.mp4",(0,1000))
    def test_get_poses(self):
        topdir = os.path.join(loc,"../","data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        print(ens.get_poses("ibl1_labeled.mp4"))
    def test_make_exampleframe(self):    
        topdir = os.path.join(loc,"../","data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        fig = ens.make_exampleframe(75,4,"ibl1_labeled.mp4",range(0,1000))
        fig.savefig("./test{}frame.png".format(75))
    def test_get_median_pose(self):
        topdir = os.path.join(loc,"../","data")
        modeldirs = ["{}".format(i+1) for i in range(1)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        med = ens.get_median_pose("ibl1_labeled.mp4",range(0,4))


class Test_TrainedModel():
    def test_init(self):
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        assert tm.pred_videos == ["ibl1_labeled.mp4"]
        assert tm.label_files == ["ibl1_labeled.csv"]
        
    def test_get_poses_raw(self):    
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        poses = tm.get_poses_raw("ibl1_labeled.mp4")
        for pk in poses.keys():
            assert pk in ["x","y","likelihoods"]
            assert type(poses[pk]) == np.ndarray

    def test_get_video_clip(self):
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip("ibl1_labeled.mp4",range(0,1000))
    def test_get_video_clip_fail(self):
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        with pytest.raises(AssertionError):
            clip = tm.get_video_clip("ibl1_labeled.mp4",(0,1000))
    def test_get_poses_array(self):
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        labelarray = tm.get_poses_array("ibl1_labeled.mp4")
        assert type(labelarray) == np.ndarray
        assert labelarray.shape[1:] == (2,4)
    def test_get_poses_and_heatmap_range(self):    
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        all_outs = tm.get_poses_and_heatmap_range("ibl1_labeled.mp4",frame_range = range(0,2))
    def test_get_poses_and_heatmap_cache(self):    
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        all_outs = tm.get_poses_and_heatmap_cache("ibl1_labeled.mp4",frame_range = range(0,2))

    @pytest.mark.xfail    
    def test_get_poses_and_heatmap(self):    
        relpath = "data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        all_outs = tm.get_poses_and_heatmap("ibl1_labeled.mp4",framenb = 0)
 
    def test_get_groundtruth(self):
        relpath = "data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"
        t = 305 

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip(videopath)
        out = tm.get_groundtruth(datapath)
        
        poses = tm.get_poses_array(videopath)
        frame = clip.get_frame(t/clip.fps)
        plt.imshow(frame)
        plt.plot(*poses[t,:,:],label = "auto",color = "blue")
        plt.plot(*out[t,:,:],label = "gt",color="orange")
        plt.plot(*poses[t,:,0],"o",markersize = 5,color = "blue")
        plt.plot(*out[t,:,0],"o",markersize = 5,color = "orange")
        plt.legend()
        plt.savefig(f"test_get_groundtruth_frame{t}.png")

    def test_get_groundtruth_perm(self):
        relpath = "data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"
        t = 305 

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip(videopath)
        out = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        
        poses = tm.get_poses_array(videopath)
        frame = clip.get_frame(t/clip.fps)
        plt.imshow(frame)
        plt.plot(*poses[t,:,:],label = "auto",color = "blue")
        plt.plot(*out[t,:,:],label = "gt",color="orange")
        plt.plot(*poses[t,:,0],"o",markersize = 5,color = "blue")
        plt.plot(*out[t,:,0],"o",markersize = 5,color = "orange")
        plt.legend()
        plt.savefig(f"test_get_groundtruth_frame{t}_perm.png")

    def test_compare_groundtruth(self):    
        relpath = "data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        out = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2])
        assert out < 41

    def test_marker_epsilon_distance(self):    
        relpath = "data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")

        gt = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        poses = tm.get_poses_array(videopath)

        output = tm.marker_epsilon_distance(poses[:-1],gt)
        


