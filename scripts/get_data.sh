#!/bin/bash

mkdir -p ./data
git clone https://huggingface.co/datasets/baohao/sage_train ./data/train
git clone https://huggingface.co/datasets/baohao/sage_validation ./data/validation