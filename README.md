# Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers

![GitHub Actions](https://github.com/fgnt/graph_pit/actions/workflows/pytest.yml/badge.svg)

This repository contains a PyTorch implementation of the Graph-PIT objective proposed in the paper ["Graph-PIT:
Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers"](https://arxiv.org/abs/2107.14446), submitted to
INTERSPEECH 2021 and the optimized variant from the paper ["Speeding up permutation invariant training for source separation"](https://arxiv.org/abs/2107.14445), submitted to the 14th ITG conference on Speech Communication 2021.

## Installation

You can install this package from GitHub:

```shell
pip install git+https://github.com/fgnt/graph_pit.git
```

Or in editable mode if you want to make modifications:

```shell
git clone https://github.com/fgnt/graph_pit.git
cd graph_pit
pip install -e .
```

This will install the basic dependencies of the package. 
If you want to run the example or the tests, install their requirements with

```shell
git clone https://github.com/fgnt/graph_pit.git
cd graph_pit
pip install -e '.[example]' # Installs example requirements
pip install -e '.[test]'    # Installs test requirements
pip install -e '.[all]'     # Installs all requirements
```
## Usage

The default Graph-PIT loss can be used as follows:

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

There is also an optimized version in `graph_pit.loss.optimized`.

## Advanced Usage

Each loss variant has three interfaces:
 - function: A simple functional interface as used above
 - class: A (data)class that computes the loss for one pair of estimates and 
   targets and exposes all intermediate states (e.g., the intermediate signals,
   the best coloring, ...). This makes testing (you can test for intermediate 
   signals, mock things, ...) and extension (you can easily sub-class and 
   overwrite parts of the computation) easier.
 - `torch.nn.Module`: A module wrapper around the class interface that allows 
   usage as a Module so that `loss_fn` can be a trainable module and the loss
   shows up in the print representation.

This is an example of the class interface `GraphPITLoss` to get access to the 
best coloring and target sum signals:

```python
import torch
from graph_pit import GraphPITLoss

# Create three target utterances and two estimated signals
targets = [torch.rand(100), torch.rand(200), torch.rand(150)]
segment_boundaries = [(0, 100), (150, 350), (300, 450)]
estimate = torch.rand(2, 500)

# Compute loss
loss = GraphPITLoss(
    estimate, targets, segment_boundaries,
    torch.nn.functional.mse_loss
)
print(loss.loss)
print(loss.best_coloring)   # This is the coloring that minimizes the loss
print(loss.best_target_sum) # This is the target sum signal (\tilde{s})
```

This is an example of the `torch.nn.Module` variant:

```python
import torch
from graph_pit.loss import GraphPITLossModule, ThresholdedSDRLoss


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = GraphPITLossModule(
            loss_fn=ThresholdedSDRLoss(max_sdr=20, epsilon=1e-6)
        )
```

## Optimized variant

There is an optimized variant for a modified SDR in `graph_pit.loss.optimized`.
This loss sums the energies of the target and error signals across estimated 
and target signals before computing the SDR.
This loss variant showed a slight improvement in separation performance compared
to the standard SDR.
It can be factorized and its computation can be optimized (a lot).

The optimized variant works by computing a similarity matrix and using a 
constrained permutation solving algorithm on this similarity matrix.

## Cite this work

If you use this code, please cite the papers:

**TODO: Insert bibtex citation when papers are accepted / published**


- "Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers": https://arxiv.org/abs/2107.14446
- "Speeding up permutation invariant training for source separation": https://arxiv.org/abs/2107.14445

