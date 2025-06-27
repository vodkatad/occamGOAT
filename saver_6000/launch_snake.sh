#!/bin/bash
#occam-run -n node40 -v /archive/home/egrassi/cuda/:/archive/home/egrassi/cuda/ egrassi/occamsnakes/cuda12_jupy:1 /archive/home/egrassi/cuda/saver_6000/launch_snake.sh
#source /etc/profile && snakemake -p -f --cores 1 all &> run_all.slog
source /etc/profile && snakemake -p -f --cores 1 data/transformer_2t_2l_b16-l5e-06-h16-d0.6.done  &> run_test.slog
