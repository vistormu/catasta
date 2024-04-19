# Catasta: Streamlined Model Experimentation

<div align="center">
<img style="width: 30%" src="assets/catasta.svg">
    
[![pypi version](https://img.shields.io/pypi/v/catasta?logo=pypi)](https://pypi.org/project/catasta/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
<!-- [![docs](https://badgen.net/badge/readthedocs/documentation/blue)](https://catasta.readthedocs.io/en/latest/) -->
</div>

_Catasta_ is a Python library designed to simplify the process of Machine Learning model experimentation. It encapsulates the complexities of model training, evaluation, and inference in a very simple API.

> [!WARNING]
> :construction: _Catasta_ is in early development :construction:
> 
> Expect breaking changes on every release until `v1.0.0` is reached.
> 
> Also, The documentation and examples for the library are under development.

---

_Catasta_ is a very simple and easy to use package.

### The `models` module

_Catasta_ offers a variety of pre-implemente Machine Learning models. All models are **single-scripted**, so feel free to copy and paste them anywhere.

For regression:

- Approximate Gaussian Process
- Transformer
- Transformer with FFT
- Mamba
- Mamba with FFT
- FeedForward Neural Network

For classification:

- Convolutional Neural Network
- Transformer
- Transformer with FFT
- Mamba
- Mamba with FFT
- FeedForward Neural Network

### The `datasets` module 

Provides an easy way to import the data contained in directories.

### The `transformations` module 

Let's you apply transformations to the data when its loaded to a dataset, such as window sliding, normalization...

### The `scaffolds` module 

Scaffolds are where models and datasets are integrated for training, handling both training and evaluation. 

_Catasta_ supports and plans to support the following Machine Learning tasks:

- [x] SISO Regression
- [x] MISO Regression
- [x] Image Classification
- [ ] Signal Classification
- [ ] Binary Classification
- [ ] Probabilistic Regression and Classification

### The `archways` module 

Takes a trained model and handles the inference task.

## Installation

### Install via pip

Catasta is available as a PyPi package:

```sh
pip install catasta
```

### Install from source

Clone the repository

```sh
git clone https://github.com/vistormu/catasta
```

and install the dependencies

```sh
pip install -r requirements.txt
```
