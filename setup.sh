#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate surfel_splatting

# I am not sure if I need all of these but it worked with them
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9
export CMAKE_CUDA_COMPILER=/home/mo50kizi/disk/miniconda3/envs/hierarchical_3d_gaussians/bin/nvcc
export CUDA_HOME=/home/mo50kizi/disk/miniconda3/envs/hierarchical_3d_gaussians/
export PATH=$CUDA_HOME/bin:$PATH

# this should not be called every time but just install the diff-surfel-rasterization package with 'pip install -e'
# conda env update --file environment.yml
# this might be slow because it first uninstalls the package. but it should be done once like this in the beginning:
# pip install -e submodules/diff-surfel-rasterization --no-cache-dir --force-reinstall  # (i don't know if the no-cache-dir and force-reinstall are needed for the initial install but if the initial install didn't work, this should fix it)
# pip install -e submodules/simple-knn --no-cache-dir --force-reinstall
# also make sure to install ninja to accelerate the build process:
# pip install ninja
# finally, we can just update the package using this command:
cd submodules/diff-surfel-rasterization
python setup.py build_ext --inplace
cd -
