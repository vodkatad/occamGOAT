#!/bin/bash
#occam-run -n node40 -v /archive/home/egrassi/cuda/:/archive/home/egrassi/cuda/ egrassi/occamsnakes/cuda12_jupy:1 /archive/home/egrassi/cuda/saver_6000/launch_snake.sh
#source /etc/profile && snakemake -p -f --cores 1 all &> run_all.slog
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source /etc/profile && snakemake -k -p -f --cores 1 all_graphsage_512  &> run_all_sage512.slog
