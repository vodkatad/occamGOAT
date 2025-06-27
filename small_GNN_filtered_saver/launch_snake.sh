#!/bin/bash
source /etc/profile &&  time snakemake -p -f --cores 1 data/edges.png
