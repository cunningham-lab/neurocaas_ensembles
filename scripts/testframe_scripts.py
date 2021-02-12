## Script to save out example frames. 
import dgp_ensembletools.models
import os

loc = os.path.abspath(os.path.dirname(__file__))
saveloc = os.path.join(loc,"script_mats")

ensembledata = {
        "../data":{"nb_models":4,"video":"ibl1_labeled.mp4","frames":[73,74,75]},
        "../fishdata":{"nb_models":5,"video":"male1_labeled.mp4","frames":[340,420,430]}
        }

if __name__ == "__main__":
    for ed,edict in ensembledata.items():
        ensemble = dgp_ensembletools.models.Ensemble(os.path.join(loc,ed),[str(i+1) for i in range(edict["nb_models"])],ext = "mp4")
        for frame in edict["frames"]:
            fig = ensemble.make_exampleframe(frame,4,edict["video"],range(0,1000))
            fig.savefig(os.path.join(saveloc,"example_frame_{video}_{frame}.png".format(video = edict["video"],frame=frame)))
            fig.clear()


