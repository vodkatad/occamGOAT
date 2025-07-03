#!/bin/bash
#occam-run -n node40 -v /archive/home/egrassi/cuda/:/archive/home/egrassi/cuda/ egrassi/occamsnakes/cuda12_jupy:1 /archive/home/egrassi/cuda/saver_6000/launch_snake_sage_k.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source /etc/profile && snakemake -k -p -f --cores 1 allk  &> run_allk.slog
