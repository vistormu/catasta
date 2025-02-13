# catasta: straightforward machine learning model experimentation

<div align="center">
<img style="width: 40%" src="assets/catasta_logo.svg">
    
[![pypi version](https://img.shields.io/pypi/v/catasta?logo=pypi)](https://pypi.org/project/catasta/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
<!-- [![docs](https://badgen.net/badge/readthedocs/documentation/blue)](https://catasta.readthedocs.io/en/latest/) -->
</div>

_catasta_ is a python library designed to simplify the process of machine learning model experimentation. optimization, training, evaluation and inference all in one place!

> [!WARNING]
> :construction: _catasta_ is in early development :construction:
> 
> expect breaking changes on every release until `v1.0.0` is reached.
> 
> the documentation is under development.

---

With _catasta_, you can build a model like an _archway_... let me explain:

### optimization

first, set the foundations of the model with the `Foundation` class. this class uses the popular and supercool [optuna](https://github.com/optuna/optuna) library to optimize a model given a hyperparameter space and an objective function.

```python
hp_space = {
    "n_patches": (2, 7),
    "d_model": (8, 16),
    "n_layers": (1, 2),
    "n_heads": (1, 2),
    "feedforward_dim": (8, 16),
    "head_dim": (4, 8),
    "dropout": (0.0, 0.5),
    "layer_norm": (True, False),
}

foundation = Foundation(
    hyperparameter_space=hp_space,
    objective_function=objective,
    sampler="bogp",
    n_trials=100,
    direction="maximize",
    use_secretary=True,
    catch_exceptions=True,
)

optimization_info = foundation.optimize()
```

### training

set the scaffolds of your model with the `Scaffold` class. this class integrates a model and a dataset for training and evaluation.

```python
model = FeedforwardRegressor(
    n_inputs=32,
    n_outputs=1,
    hidden_dims=[8, 16, 8],
    dropout=0.0,
    use_layer_norm=True,
    activation="relu",
)

dataset = CatastaDataset(
    root="path/to/dataset/",
    task="regression",
    input_name="input",
    output_name="output",
)

scaffold = Scaffold(
    model=model,
    dataset=dataset,
    optimizer="adamw",
    loss_function="mse",
)

scaffold.train(
    epochs=100,
    batch_size=256,
    lr=1e-3,
)

info = scaffold.evaluate()
```

### inference

your archway is finished with the `Archway` class. this class runs the inference of the model given its saved path

```python
archway = Archway(
    path= "path/to/saved/model.pt",
)

example_input = np.random.rand(1, 4).astype(np.float32)
output = archway.predict(example_input)
```

the archway uses the `onnxruntime` library if a `.onnx` file is provided, but you must install manually `onnx` and `onnxruntime` to use this feature

```python
archway = Archway(
    path= "path/to/saved/model.onnx",
)

example_input = np.random.rand(1, 4).astype(np.float32)
output = archway.predict(example_input)
```

finally, the archway can also serve a model as a REST API using the `FastAPI` library. to use this feature, you must install `fastapi`, `pydantic`, and `uvicorn` manually

```python
archway = Archway(
    path= "path/to/saved/model.pt",
)

class Data(BaseModel):
    s0: float
    s1: float
    s2: float
    s3: float

archway.serve(
    host="145.94.127.212",
    port=8080,
    pydantic_model=Data,
)
```

### other modules

_catasta_ also has different modules that facilitate model experimentation

* `catasta.models` offers a variety of pre-implemented Machine Learning models. All models are **single-scripted**, so feel free to copy and paste them anywhere.

* `catasta.transformations` let's you apply transformations to the data when its loaded to a dataset, such as window sliding, normalization...

* `catasta.utils` has several functions that are useful for model optimization and training.

## installation

### Install via pip

_catasta_ is available as a PyPi package:

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

## documentation

the documentation is under development, but you can check out some examples in the `examples` folder!
