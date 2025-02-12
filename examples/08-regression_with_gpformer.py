from torch.nn import Identity
from catasta import Scaffold, Dataset
from catasta.models import GPHeadRegressor, TransformerRegressor
from catasta.dataclasses import TrainInfo


def single_output() -> None:
    pre_model = TransformerRegressor(
        n_inputs=1,
        n_outputs=1,
        n_patches=1,
        d_model=16,
        n_heads=1,
        n_layers=1,
        feedforward_dim=16,
        head_dim=16,
        dropout=0.1,
    )
    pre_model.linear_head = Identity()  # type: ignore

    model = GPHeadRegressor(
        pre_model=pre_model,
        pre_model_output_dim=16,
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
    pre_model = TransformerRegressor(
        n_inputs=4,
        n_outputs=3,
        n_patches=1,
        d_model=16,
        n_heads=1,
        n_layers=1,
        feedforward_dim=16,
        head_dim=16,
        dropout=0.1,
    )
    pre_model.linear_head = Identity()  # type: ignore

    model = GPHeadRegressor(
        pre_model=pre_model,
        pre_model_output_dim=16,
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
    # single_output()
    multi_output()
