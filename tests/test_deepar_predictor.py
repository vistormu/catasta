import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catasta.entities import ModelData
from catasta.scaffolds import PredictorScaffold

# from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator


def get_data(dir: str) -> list[ModelData]:
    data: list[ModelData] = []
    for filename in os.listdir(dir):
        if not filename.endswith(".csv"):
            continue

        data_frame: pd.DataFrame = pd.read_csv(dir + filename)
        data.append(ModelData(
            input=data_frame['input'].to_numpy(),
            output=data_frame['output'].to_numpy(),
        ))

    return data


def escalonate(data: ModelData) -> list[ModelData]:
    return [ModelData(
        input=data.input[:i],
        output=data.output[:i],
    ) for i in range(1, len(data.input))]


def main() -> None:
    # data: list[ModelData] = get_data("data/steps/")
    data: list[ModelData] = get_data("tests/data/nylon_strain/")

    # Train
    prediction_length: int = 12
    epochs: int = 100
    context: int = int(len(data[0].input)*0.15)

    # model = DeepAREstimator(
    #     prediction_length=prediction_length,
    #     num_layers=4,
    #     hidden_size=64,
    #     freq="D",
    #     num_feat_dynamic_real=1,
    # )

    model = PatchTSTEstimator(
        prediction_length=prediction_length,
        patch_len=prediction_length,
        d_model=32,
        nhead=4,
        dim_feedforward=128,
    )

    scaffold = PredictorScaffold(model)

    scaffold.train(dataset=data[:-1],
                   epochs=epochs,
                   lr=0.001,
                   train_ratio=0.8,
                   )

    test_data: list[ModelData] = escalonate(data[-1])[context:]
    inputs: list[np.ndarray] = [d.input for d in test_data]
    outputs: list[np.ndarray] = [d.output for d in test_data]

    # WITH REAL OUTPUT
    predicted_output: np.ndarray = outputs[0]
    q: np.ndarray = outputs[0][-prediction_length:]
    for input, output in zip(inputs, outputs):
        assert len(q) == prediction_length
        assert len(input) == len(output)

        d = ModelData(input, output)
        forecasts, _ = scaffold.predict(d)

        # p_hat = forecasts[0].mean
        # overlapping_mean: np.ndarray = np.mean([q[1:], p_hat[:-1]], axis=0)
        # q = np.append(overlapping_mean, p_hat[-1])
        # predicted_output = np.append(predicted_output, q[0])

        predicted_output = np.append(predicted_output, np.mean(forecasts[0].mean))

    plt.figure(figsize=(30, 20))
    plt.plot(data[-1].output, label="Real")
    plt.plot(predicted_output, label="Predicted")
    plt.legend()
    plt.show()
    plt.close()

    # WITHOUT REAL OUTPUT
    predicted_output: np.ndarray = outputs[0]
    q: np.ndarray = outputs[0][-prediction_length:]
    for input in inputs:
        assert len(q) == prediction_length
        assert len(input) == len(predicted_output)

        d = ModelData(input, predicted_output)

        forecasts, _ = scaffold.predict(d)

        # p_hat = forecasts[0].mean
        # overlapping_mean: np.ndarray = np.mean([q[1:], p_hat[:-1]], axis=0)
        # q = np.append(overlapping_mean, p_hat[-1])
        # predicted_output = np.append(predicted_output, q[0])
        predicted_output = np.append(predicted_output, np.mean(forecasts[0].mean))

    plt.figure(figsize=(30, 20))
    plt.plot(data[-1].output, label="Real")
    plt.plot(predicted_output, label="Predicted")
    plt.plot(data[-1].output[:context], color="green")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")
