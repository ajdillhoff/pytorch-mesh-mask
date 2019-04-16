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

# Output

The output demonstrates the issue of accessing tensors from GPU.

    **CPU**
    Compute time: 0.03938102722167969s
    Access time: 0.026835918426513672s
    **CUDA**
    Compute time: 0.0003864765167236328s
    Access time: 1.0503544807434082s

Why is the access time for the CUDA version so much longer?
