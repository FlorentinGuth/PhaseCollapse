# Phase Collapse in Neural Networks
This repository contains the code to reproduce experiments in the paper "Phase Collapse in Neural Networks", accepted at ICLR 2022.

## Requirements
### Python packages
Our code is designed to run on GPU using [PyTorch](https://pytorch.org/). In order to run our experiments, you will need the following packages: `numpy`, `scipy`, `torch` (1.8), and `torchvision`. See `requirements.txt` for the precise (but not minimal) environment which was used to run the experiments in the paper, it can be installed with `pip install -r requirements.txt`.

### Datasets
The ImageNet dataset must be downloaded from http://www.image-net.org/challenges/LSVRC/2012/downloads (registration required).
Then move validation images to labeled subfolders, using [the PyTorch shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

CIFAR-10 is automatically downloaded by Pytorch.

## Usage
To train a model, run `main_block.py` with the desired arguments. Running `run.py` will train all models whose results are reported in Tables 1 and 2, if provided with the path to the ImageNet dataset.

