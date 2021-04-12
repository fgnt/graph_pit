# Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers

This repository contains a PyTorch implementation of the Graph-PIT objective proposed in the paper "Graph-PIT:
Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers", submitted to
INTERSPEECH 2021.

**:warning: This repository is under construction!**

## Installation

Make sure to have PyTorch installed. You can install this package from GitHub:

```shell
pip install git+https://github.com/fgnt/graph_pit.git
```

Or in editable mode if you want to make modifications:

```shell
git clone https://github.com/fgnt/graph_pit.git
pip install -e graph_pit
```

## Usage

```python
import torch
from graph_pit import graph_pit_loss

# Create three target utterances and two estimated signals
targets = [torch.rand(100), torch.rand(200), torch.rand(150)]
segment_boundaries = [(0, 100), (150, 350), (300, 450)]
estimate = torch.rand(2, 500)

# Compute loss
loss = graph_pit_loss(
    estimate, targets, segment_boundaries,
    torch.nn.functional.mse_loss
)
```

## Cite this work

If you use this code, please cite the paper:

**TODO: Insert bibtex citation when paper is accepted**

```
Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers
```
