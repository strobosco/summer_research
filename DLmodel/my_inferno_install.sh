#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh

mamba env update -n aria --file conda-environment_py38_cu11.yaml

pip install -r requirements38.txt
pip install Cython==0.29

pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2

pip install -e . 

# Insightface has problems with downloading some of their models
echo -e "\nDownloading insightface models..."
mkdir -p ~/.insightface/models/
if [ ! -d ~/.insightface/models/antelopev2 ]; then
  wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
  unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi
if [ ! -d ~/.insightface/models/buffalo_l ]; then
  wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1"
  unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi

echo "Installation finished"