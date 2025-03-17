#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh

mamba env update -n aria --file conda-environment_py38_cu11_ubuntu.yml 
echo "Installing other requirements"
pip install -r requirements38.txt
pip install Cython==0.29
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
echo "Installing GDL"
pip install -e . 
echo "Installation finished"
