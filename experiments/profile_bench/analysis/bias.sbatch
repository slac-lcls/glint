#!/bin/bash
#SBATCH -A m5149 -C cpu -q shared -c 2 --mem=12G -t 00:15:00 -J bias45 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/bias_%j.out
export PYTHONNOUSERSITE=1
/pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/bias.py
