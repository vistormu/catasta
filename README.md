# Catasta: Streamlined Model Experimentation

<p align="center">
    <img style="width: 40%" src="assets/catasta.svg">
</p>

[![pypi version](https://img.shields.io/pypi/v/catasta?logo=pypi)](https://pypi.org/project/catasta/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
<!-- [![docs](https://badgen.net/badge/readthedocs/documentation/blue)](https://catasta.readthedocs.io/en/latest/) -->

Catasta is a Python library designed to simplify and accelerate the process of machine learning model experimentation. It encapsulates the complexities of model training and evaluation, offering researchers and developers a straightforward pipeline for rapid model assessment with minimal setup required.

> Note: Catasta only supports regression at the moment. Other techniques such as classification or prediction are being developed.

> Important: Catasta is subject of change until a major version is launched.

## Key features

### Models
The `models` module in Catasta houses a variety of machine learning models. Users can easily select from a range of pre-implemented models suited for different tasks and requirements.

### Datasets
Within the `datasets` module, Catasta provides an easy way to import datasets contained in directories, being able to modify the data shape in an easy way.

### Transformations
The `transformations` module lets you apply transformations to the data when its loaded to a dataset.

### Scaffolds
The `scaffolds` component is the core of the Catasta library, where the integration of models and datasets occurs. Scaffolds handle the intricacies of training, evaluation, and any additional processing required to transform raw data into actionable insights. This automation empowers users to focus on the conceptual aspects of their models rather than the operational details.

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

## Documentation

Work in progress
