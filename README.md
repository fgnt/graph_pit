# Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers

![GitHub Actions](https://github.com/fgnt/graph_pit/actions/workflows/pytest.yml/badge.svg)

This repository contains a PyTorch implementation of the Graph-PIT objective proposed in the paper ["Graph-PIT:
Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers"](https://arxiv.org/abs/2107.14446), submitted to
INTERSPEECH 2021 and the optimized variant from the paper ["Speeding up permutation invariant training for source separation"](https://arxiv.org/abs/2107.14445), submitted to the 14th ITG conference on Speech Communication 2021.

The optimized uPIT code used in [2] can be found in [padertorch](https://github.com/fgnt/padertorch) and the example noteook [`runtimes.ipynb`](graph_pit/examples/runtimes.ipynb).

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

The Graph-PIT losses in this repository require a list of utterance signals and segment boundaries (tuples of start and end times).
There are two different implementations:

- `graph_pit.loss.unoptimized` contains the original Graph-PIT loss as proposed in [1], and
- `graph_pit.loss.optimized` contains the optimized Graph-PIT loss variants from [2].

The default (unoptimized) Graph-PIT loss from [1] can be used as follows:

```python
import torch
from graph_pit import graph_pit_loss

# Create three target utterances and two estimated signals
targets = [torch.rand(100), torch.rand(200), torch.rand(150)]   # List of target utterance signals
segment_boundaries = [(0, 100), (150, 350), (300, 450)]     # One start and end time for each utterance
estimate = torch.rand(2, 500)   # The estimated separated streams

# Compute loss with the unoptimized loss function, here mse for example
loss = graph_pit_loss(
    estimate, targets, segment_boundaries,
    torch.nn.functional.mse_loss
)

# Example for using the optimized sa-SDR loss from [2]
from graph_pit.loss.optimized import optimized_graph_pit_source_aggregated_sdr_loss
loss = optimized_graph_pit_source_aggregated_sdr_loss(
   estimate, targets, segment_boundaries,
   # assignent_solver can be one of:
   #  - 'optimal_brute_force'
   #  - 'optimal_branch_and_bound'
   #  - 'optimal_dynamic_programming' <- fastest
   #  - 'dfs'
   #  - 'greedy_cop' 
   assignment_solver='optimal_dynamic_programming'
)
```

This unoptimized loss variant works with any loss function `loss_fn`, but it is in may cases quite slow (see [2]).
The optimized version from [2] can be found in `graph_pit.loss.optimized` for the source-aggregated SDR.
You can define your own optimized Graph-PIT losses by subclassing 
`graph_pit.loss.optimized.OptimizedGraphPITLoss` and defining the property 
`similarity_matrix` and the method `compute_f`.

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

## Examples

There are two examples in `graph_pit.examples`:
 - [`tasnet`](graph_pit/examples/tasnet): An example training script for a DPRNN-based TasNet model trained with Graph-PIT using [padertorch](https://github.com/fgnt/padertorch)
 - [`runtimes.ipynb`](graph_pit/examples/runtimes.ipynb): A Jupyter notebook comparing the runtimes of different uPIT and Graph-PIT variants. This notebook creates plots similar to [2].

## Cite this work / References

If you use this code, please cite the papers:

- [1] The first paper: "Graph-PIT: Generalized permutation invariant training for continuous separation of arbitrary numbers of speakers": https://arxiv.org/abs/2107.14446
```bibtex
@inproceedings{vonneumann21_GraphPIT,
  author={Thilo von Neumann and Keisuke Kinoshita and Christoph Boeddeker and Marc Delcroix and Reinhold Haeb-Umbach},
  title={{Graph-PIT: Generalized Permutation Invariant Training for Continuous Separation of Arbitrary Numbers of Speakers}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={3490--3494},
  doi={10.21437/Interspeech.2021-1177}
}
```

- [2] The speed optimizations: "Speeding up permutation invariant training for source separation": https://arxiv.org/abs/2107.14445
```bibtex
@inproceedings{vonneumann21_SpeedingUp,
  author={Thilo von Neumann and Christoph Boeddeker and Keisuke Kinoshita and Marc Delcroix and Reinhold Haeb-Umbach},
  booktitle={Speech Communication; 14th ITG Conference}, 
  title={Speeding Up Permutation Invariant Training for Source Separation}, 
  year={2021},
  volume={},
  number={},
  pages={1-5},
  doi={}
}
```
