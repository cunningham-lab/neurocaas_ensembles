#!/usr/bin/env bash

#name the job pybench33 and place it's output in a file named slurm-<jobid>.out
# allow 40 minutes to run (it should not take 40 minutes however)
# set partition to 'all' so it runs on any available node on the cluster

#SBATCH -J 'slurm_ekb'
#SBATCH -o slurm_ekb-%j.out
#SBATCH -t 02-11:00:00
#SBATCH --mem 16gb
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -p ctn
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekb2154@columbia.edu         # Where to send mail (e.g. uni123@columbia.edu)
. activate dgptest


echo "extend_dgp.py $1"
python /home/ekb2154/data/libraries/dgp_paninski/etc/ensembles/extend_dgp.py $1
echo "Ran extend_dgp.py $1"
