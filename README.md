# NTarp
A Scikit-Learn implementation of the NTarp clustering algorithm, originally introduced [here](https://arxiv.org/pdf/1806.05297.pdf) and developed further [here](https://arxiv.org/abs/2008.09579). 

The objective function of NTarp is the minimum normalized within-ss, and so this also necessarily includes an efficient way of computing this. The method used here is comparable to [CKmeans.1d.dp](https://github.com/AldenMB/Ckmeans.1d.dp) in terms of algorithmic efficiency, but roughly twice as fast in practice (at least in my testing).

# Installation

```
pip install ntarp
```

# Usage

The main interface is the `ntarp.NTarp` object, which uses the sklearn cluster interface, through the `fit` and `predict` methods. In addition the withinss function is available as `ntarp.separability.w`.

