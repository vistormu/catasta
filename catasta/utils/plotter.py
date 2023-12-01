import matplotlib.pyplot as plt

from ..dataclasses import RegressionEvalInfo, RegressionTrainInfo


class Plotter:
    def __init__(self, *, train_info: RegressionTrainInfo, eval_info: RegressionEvalInfo) -> None:
        self.train_info: RegressionTrainInfo = train_info
        self.eval_info: RegressionEvalInfo = eval_info

    def plot_train_loss(self) -> None:
        plt.figure(figsize=(30, 20))
        plt.plot(self.train_info.train_loss, label="train loss", color="red")
        if self.train_info.eval_loss is not None:
            plt.plot(self.train_info.eval_loss, label="eval loss", color="blue")
        plt.legend()
        plt.show()

    def plot_predictions(self) -> None:
        fig, ax = plt.subplots(2, 1, figsize=(30, 20))
        ax[0].plot(self.eval_info.true_input, label="input", color="black")
        ax[1].plot(self.eval_info.predicted_output, label="predictions", color="red")
        ax[1].plot(self.eval_info.true_output, label="real", color="black")
        if self.eval_info.stds is not None:
            ax[1].fill_between(
                range(len(self.eval_info.predicted_output)),
                self.eval_info.predicted_output-2 * self.eval_info.stds,
                self.eval_info.predicted_output+2*self.eval_info.stds,
                color="red",
                alpha=0.2,
            )
        plt.legend()
        plt.show()

    def plot_input_output_relation(self) -> None:
        plt.figure(figsize=(30, 20))
        plt.plot(self.eval_info.true_input, self.eval_info.predicted_output, label="predictions", color="red")
        plt.plot(self.eval_info.true_input, self.eval_info.true_output, label="real", color="black")
        plt.legend()
        plt.show()

    def plot_output_relation(self) -> None:
        plt.figure(figsize=(30, 20))
        plt.plot(self.eval_info.true_output, self.eval_info.true_output, label="real", color="black")
        plt.plot(self.eval_info.true_output, self.eval_info.predicted_output, "o", label="predictions", color="red")
        plt.legend()
        plt.show()

    def plot_all(self) -> None:
        self.plot_train_loss()
        self.plot_predictions()
        self.plot_input_output_relation()
        self.plot_output_relation()
