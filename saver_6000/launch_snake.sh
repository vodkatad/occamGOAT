#!/bin/bash
#occam-run -n node40 -v /archive/home/egrassi/cuda/:/archive/home/egrassi/cuda/  egrassi/occamsnakes/cuda12_jupy:1 /archive/home/egrassi/cuda/small_GNN_filtered_saver/launch_snake.sh
source /etc/profile &&  snakemake -p -f --cores 1 all
