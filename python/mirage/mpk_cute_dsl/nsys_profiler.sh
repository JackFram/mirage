#!/bin/bash
CUDA_VISIBLE_DEVICES=6,7 nsys profile -o results/mpk_intranode_moe.nsys-rep --stats=true --trace cuda,nvtx,osrt python mpk_intranode_moe.py
