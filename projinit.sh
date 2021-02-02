#!/usr/bin/env bash

# plots all environemnts
# this will make videos comparing step 1 and step 2
#source activate /home/ekellbuch/anaconda3/envs/dgp

call_pyfunction () {
    #echo "$1"
    #sbatch ./project_init.sh "$1"
    python project_init.py $1
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

task="iblright"
scorer="kelly"
date="2030-01-0"
basepath="/datahd2a/datasets/tracki/iblright/"
for mday in {1..2}
    do 
    #call_pyfunction "$data ${scorers[$data]} $date"
    newdate="$date$mday"
    #echo "$newdate"
    call_pyfunction "$task $scorer $newdate $basepath"
    done

