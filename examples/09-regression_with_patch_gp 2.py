from torch.nn import Sequential, Linear
from einops.layers.torch import Rearrange, Reduce

from catasta import Scaffold, Dataset
from catasta.models import GPHeadRegressor
from catasta.dataclasses import TrainInfo


def single_output() -> None:
    n_inputs: int = 1
    n_patches: int = 1
    patch_size: int = n_inputs // n_patches
    d_model: int = 16
    pooling: str = "concat"
    pre_model_output_dim = d_model * n_patches if pooling == "concat" else d_model

    pre_model = Sequential(
        Rearrange('b (n p) -> b n p', p=patch_size),
        Linear(patch_size, d_model),
        Rearrange('b n d -> b (n d)') if pooling == "concat" else Reduce('b n d -> b d', pooling)
    )

    model = GPHeadRegressor(
        pre_model=pre_model,
        pre_model_output_dim=pre_model_output_dim,
        n_inputs=1,
        n_outputs=1,
        n_inducing_points=16,
        kernel="rq",
        mean="constant",
        likelihood="gaussian",
    )

    dataset_root: str = "data/incomplete/"
    dataset = Dataset(
        dataset_root,
        task="regression",
        input_name="input",
        output_name="output",
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=1000,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


def multi_output() -> None:
    n_inputs: int = 4
    n_patches: int = 1
    patch_size: int = n_inputs // n_patches
    d_model: int = 16
    pooling: str = "concat"
    pre_model_output_dim = d_model * n_patches if pooling == "concat" else d_model

    pre_model = Sequential(
        Rearrange('b (n p) -> b n p', p=patch_size),
        Linear(patch_size, d_model),
        Rearrange('b n d -> b (n d)') if pooling == "concat" else Reduce('b n d -> b d', pooling)
    )

    model = GPHeadRegressor(
        pre_model=pre_model,
        pre_model_output_dim=pre_model_output_dim,
        n_inputs=4,
        n_outputs=3,
        n_inducing_points=16,
        kernel="rq",
        mean="constant",
        likelihood="gaussian",
    )

    # dataset_root: str = "data/tactile/"
    dataset_root: str = "data/squishy/"
    dataset = Dataset(
        dataset_root,
        task="regression",
        input_name=["s0", "s1", "s2", "s3"],
        output_name=["x", "y", "fz"],
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=5000,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    single_output()
    multi_output()
