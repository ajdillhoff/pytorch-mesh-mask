# pytorch-mesh-mask

Computes a mask indicating the visible vertices of a mesh. This repository contains a minimal working example which demonstrates an issue referenced [here](https://discuss.pytorch.org/t/signficant-delay-when-accessing-tensor-returned-from-cuda-extension/42214).

# Dependencies

- PyTorch 1.0
- CUDA 10.0
- numpy

# Installation

To install the PyTorch extension, run `python setup.py install` under the `extension/` folder.

# Running

`python run.py`
