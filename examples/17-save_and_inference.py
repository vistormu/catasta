import time
import numpy as np
from catasta import Scaffold, Dataset, Archway
from catasta.models import FeedforwardRegressor
from catasta.dataclasses import TrainInfo


def train() -> None:
    model = FeedforwardRegressor(
        n_inputs=4,
        n_outputs=3,
        hidden_dims=[16, 16, 16],
        dropout=0.1,
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
        loss_function="mse",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=5000,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate(prediction_function=lambda model, x: model(x).cpu().numpy())
    print(info)

    scaffold.save("models/fnn.pt")
    scaffold.save("models/fnn.onnx")


def inference_with_torch() -> None:
    archway = Archway(
        path="models/fnn.pt",
    )

    example_input = np.random.rand(1, 4).astype(np.float32)
    start = time.time()
    output = archway.predict(example_input)
    print(f"Output: {output}")
    print(f"Elapsed time: {(time.time() - start) * 1000:.2f} ms")


def inference_with_onnx() -> None:
    # onnxruntime must be installed
    archway = Archway(
        path="models/fnn.onnx",
    )

    example_input = np.random.rand(1, 4).astype(np.float32)
    start = time.time()
    output = archway.predict(example_input)
    print(f"Output: {output}")
    print(f"Elapsed time: {(time.time() - start) * 1000:.2f} ms")


if __name__ == '__main__':
    train()
    inference_with_torch()
    inference_with_onnx()
