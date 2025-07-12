#!/bin/bash
echo Prepare raw data
python3 prepare_s3dis.py
echo Prepare superpoints
python3 prepare_superpoints.py
echo Prepare normals
python3 prepare_normals.py