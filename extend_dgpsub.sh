#!/usr/bin/env bash

# run ensembles by copying directory
# read dgp snapshot

call_read_label () {
    echo "Read label $1"
    #sbatch ./project_init.sh "$1"
    python read_manual_labels.py $1
}


call_proj_init () {
    echo "Project init $1"
    #sbatch ./project_init.sh "$1"
    python project_init.py $1
}

call_run_dgp() {
    echo "Run dgp $1"
    sbatch ./extend_dgp.sh "$1"
    #python run_dpg_demo.py $1
}

call_extend_dgp() {
    echo "Run dgp $1"
    sbatch ./extend_dgp.sh "$1"
    #python extend_dgp.py $1
}



declare -A scorers=( ["fish"]="claire" ["flyballm"]="matt"  ["ibl1"]="kelly"  ["reach"]="mac" ["twomice"]="erica" ["paw2"]="mic") # "${animals[moo]}"
#declare -A videos=( ["fish"]="male1.avi" ["flyballm"]="2019_08_08_fly1.avi"  ["ibl1"]="ibl1.mp4"  ["reach"]="reach.avi" ["twomice"]="B29_post_side_15.avi" ["paw2"]="cortexlab_KS004_2019-09-25_001__iblrig_leftCamera.paws.short.mp4") # "${animals[moo]}"


# options
# fish
# flyball
# ibl1
# reach
#twomice
#paw2
# directory where the dlc project is
#base_dir="/data/datasets/tracki/iblright-kelly-2021-01-03"
# directory where we should run dlc
#basepath="/datahd2a/datasets/tracki/iblright/"

#base_dir="/share/home/ekb2154/data/libraries/dgp_paninski/data/iblfingers-kelly-2021-01-03"
basepath="/share/home/ekb2154/data/datasets/tracki/iblfingers/"
#base_dir="/share/home/ekb2154/data/libraries/dgp_paninski/data/iblright-kelly-2021-01-03"
#basepath="/share/home/ekb2154/data/datasets/tracki/iblright/"
# directory where we run dgp
task="iblfingers"
scorer="kelly"
# new dates for ensemble
date="2030-01-0"
batch_size=3

# Read labels
#call_read_label "$base_dir $basepath"

# Project Init
for mday in {3..4}
    do 
    newdate="$date$mday"
    #call_proj_init "$task $scorer $newdate $basepath"
    done

# Run DGP
for mday in {1..4}
    do
    #call_pyfunction "$data ${scorers[$data]} $date"
    dlcpath="$basepath""model_data/$task-$scorer-$date$mday"
    #echo "$newdate"
    #echo "$dlcpath"
    call_extend_dgp "--dlcpath $dlcpath --batch_size $batch_size"
    done





