# Catasta: Streamlined Model Experimentation

<p align="center">
    <img style="width: 40%" src="assets/catasta.svg">
</p>

[![pypi version](https://img.shields.io/pypi/v/catasta?logo=pypi)](https://pypi.org/project/catasta/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![docs](https://badgen.net/badge/readthedocs/documentation/blue)](https://catasta.readthedocs.io/en/latest/)

Catasta is a Python library designed to simplify and accelerate the process of machine learning model experimentation. It encapsulates the complexities of model training and evaluation, offering researchers and developers a straightforward pipeline for rapid model assessment with minimal setup required.

Catasta's philosophy is centered on ease-of-use and efficiency. By simplifying the selection of models and datasets, and by automating the training and evaluation process, Catasta allows users to focus on rapid prototyping and iterative experimentation. This library is ideal for those who wish to test multiple models quickly to determine the best fit for their problem statement.

## Key components

### Models
The `models` module in Catasta houses a variety of machine learning models. Users can easily select from a range of pre-implemented models suited for different tasks and requirements.

### Datasets
Within the `datasets` module, Catasta provides an easy way to import datasets contained in directories, being able to modify the data shape in an easy way.

### Scaffolds
The `scaffolds` component is the core of the Catasta library, where the integration of models and datasets occurs. Scaffolds handle the intricacies of training, evaluation, and any additional processing required to transform raw data into actionable insights. This automation empowers users to focus on the conceptual aspects of their models rather than the operational details.

## Getting started

To begin using Catasta, install the library using pip:

```sh
pip install catasta
```

## Examples
