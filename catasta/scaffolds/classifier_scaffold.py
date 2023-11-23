import time
from copy import deepcopy

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import AdamW

from vclog import Logger

from ..dataclasses import TrainInfo
from ..datasets import ClassifierDataset


@torch.no_grad()
def estimate_loss(model: Module, train_dataset: Dataset, val_dataset: Dataset, batch_size: int, eval_iters: int) -> TrainInfo:
    model.eval()

    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0

    for dataset in [train_dataset, val_dataset]:
        losses: Tensor = torch.zeros(eval_iters)
        predictions_list: list[int] = []
        labels_list: list[int] = []

        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # TMP
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        dtype: torch.dtype = torch.float32

        for i in range(eval_iters):
            input_batch, label_batch = next(iter(data_loader))
            input_batch = input_batch.to(device, dtype=dtype)
            label_batch = label_batch.to(device, dtype=torch.long)

            logits, loss = model(input_batch, label_batch)

            losses[i] = loss.item()
            predictions: Tensor = logits.argmax(dim=1)
            predictions_list.append(predictions.cpu().numpy())
            labels_list.append(label_batch.cpu().numpy())

        predictions_array: np.ndarray = np.concatenate(predictions_list, axis=0)
        labels: np.ndarray = np.concatenate(labels_list, axis=0)

        accuracy: float = np.sum(predictions_array == labels) / len(labels)

        if dataset == train_dataset:
            train_loss: float = losses.mean().item()
            train_accuracy: float = accuracy
        else:
            val_loss: float = losses.mean().item()
            val_accuracy: float = accuracy

    model.train()

    return TrainInfo(train_loss, val_loss, train_accuracy, val_accuracy)


class ClassifierScaffold:
    def __init__(self, model: Module, dataset: ClassifierDataset):
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.dataset: ClassifierDataset = dataset

        self.logger: Logger = Logger("catasta")
        self.logger.info(f'Using device: {self.device}')

    def train(self, *,
              epochs: int,
              batch_size: int,
              train_split: float = 0.8,
              learning_rate: float = 6e-4,
              log_interval: int = 1,
              eval_iters: int = 5,
              ) -> list[TrainInfo]:
        self.model.train()

        # Variables
        train_info_list: list[TrainInfo] = []
        best_val_loss: float = float('inf')
        best_model_parameters: dict = {}
        start_time: float = time.time()

        # Initialize optimizer
        optimizer: AdamW = AdamW(lr=learning_rate,
                                 params=self.model.parameters(),
                                 weight_decay=0.1,
                                 betas=(0.9, 0.95),
                                 )

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_split, (1.0-train_split)/2, (1.0-train_split)/2])

        for epoch in range(1, epochs + 1):
            data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            if best_model_parameters and (epoch % epochs//5) == 0:
                self.model.load_state_dict(best_model_parameters)

            for batch, (input_batch, label_batch) in enumerate(data_loader):
                # Forward
                input_batch = input_batch.to(self.device, dtype=self.dtype)
                label_batch = label_batch.to(self.device, dtype=torch.long)

                optimizer.zero_grad()

                _, loss = self.model(input_batch, label_batch)

                loss.backward()

                optimizer.step()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # type: ignore

                # Logging info
                if batch % log_interval == 0:
                    # Log info
                    ms_per_batch: float = (time.time() - start_time) * 1000 / log_interval

                    train_info = estimate_loss(self.model, train_dataset, val_dataset, batch_size, eval_iters)

                    epoch_log: str = f'epoch {epoch}/{epochs}'
                    percentage_log: str = f'{batch/(len(train_dataset)//batch_size)*100:.2f}%'
                    percentage_log = '0'+percentage_log if len(percentage_log.split('.')[0]) == 1 else percentage_log
                    percentage_log = percentage_log[:-1] if len(percentage_log.split('.')[0]) == 3 else percentage_log
                    train_loss_log: str = f'train({train_info.train_loss:.3f}, {train_info.train_acc:.3f})'
                    val_loss_log: str = f'val({train_info.val_loss:.3f}, {train_info.val_acc:.3f})'
                    best_val_loss_log: str = f'{best_val_loss:.3f}'
                    time_log: str = f'{ms_per_batch:.2f}ms/batch'

                    self.logger.info(f'{epoch_log} ({percentage_log}) | (loss, acc) -> {train_loss_log}, {val_loss_log} ({best_val_loss_log}) | {time_log}', flush=True)

                    # Save data
                    train_info_list.append(train_info)

                    # Update model parameters
                    if train_info.val_loss < best_val_loss:
                        best_val_loss = train_info.val_loss
                        best_model_parameters = deepcopy(self.model.state_dict())

                    # Reset timer
                    start_time = time.time()

        self.logger.info(f'{epoch_log} ({percentage_log}) | (loss, acc) -> {train_loss_log}, {val_loss_log} ({best_val_loss_log}) | {time_log}')  # type: ignore

        # Load best model parameters
        self.model.load_state_dict(best_model_parameters)

        self._calculate_accuracy(test_dataset)

        return train_info_list

    def _calculate_accuracy(self, test_dataset: Dataset) -> None:
        # Variables
        predictions_list: list[list[int]] = []
        labels_list: list[int] = []

        data_loader: DataLoader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # type: ignore

        for _ in range(5):
            for input_batch, label_batch in data_loader:
                result: list[int] = self.predict(input_batch).tolist()

                predictions_list.append(result)
                labels_list.append(label_batch.cpu().numpy())

            predictions: np.ndarray = np.concatenate(predictions_list, axis=0)
            labels: np.ndarray = np.concatenate(labels_list, axis=0)

        accuracy: float = np.sum(predictions == labels) / len(labels)  # type: ignore

        Logger.info(f'acc: {accuracy:.5f}')

    @torch.no_grad()
    def predict(self, input: Tensor) -> Tensor:
        self.model.eval()

        logits, _ = self.model(input)

        return logits.argmax(dim=1)
