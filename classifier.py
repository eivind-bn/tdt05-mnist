from typing import *
from typing import List
import torch
from mnist_encoder import *

class Classifier(MnistEncoder[int]):

    def __init__(self, 
                 train_size: float,
                 device: Device|torch.device, 
                 latent_size: int, 
                 lr: float) -> None:
        
        self._classifier = torch.nn.Linear(latent_size,10)

        super().__init__(
            train_size=train_size,
            device=device,
            lr=lr,
            output_layers=self._classifier
        )

        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self._classifier_optimizer = torch.optim.Adam(
            params=self._classifier.parameters(),
            lr=lr
        )

    def fit(self, x: Tensor, y: Tensor, optimize_encoder: bool) -> float:
        optimizer = self._optimizer if optimize_encoder else self._classifier_optimizer
        optimizer.zero_grad()
        y_hat = self.forward(x, use_grad=True)
        loss: Tensor = self._cross_entropy_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        return float(loss)
    
    def get_accuracy(self) -> float:
        y_hat = self.forward(self._test_data)
        predictions = y_hat.argmax(dim=1)
        labels = self._test_target
        cmp = torch.count_nonzero(predictions == labels).item()
        return float(cmp/predictions.nelement())  
    
    def train(self, 
              epochs: int, 
              batch: float, 
              plot_stats: bool = False,
              optimize_encoder: bool = False) -> None:
        progress_bar = tqdm(range(epochs))
        losses: List[float] = []
        accuracies: List[float] = []
        for epoch in progress_bar:
            x,y = self.minibatch(batch)
            loss = self.fit(
                x=x,
                y=y,
                optimize_encoder=optimize_encoder
            )
            accuracy = self.get_accuracy()
            progress_bar.set_description(f"{loss=:.8f}, {accuracy=:.4f}")
            losses.append(loss)
            accuracies.append(accuracy)
    
        if plot_stats:
            plt.plot(torch.arange(len(losses)), losses, label="Loss")
            plt.plot(torch.arange(len(accuracies)), accuracies, label="Accuracy")
            plt.xlabel("Epoch")
            plt.legend()
            plt.show()
    
    def _output_transform(self, output: Tensor) -> int:
        return int(output[0].argmax().item())

    def set_encoder_params(self, params: Iterable[torch.nn.Parameter]) -> None:
        with torch.no_grad():
            for self_param, other_param in zip(self.encoder.parameters(), params, strict=True):
                self_param[:] = other_param

    @staticmethod
    def load(path: str) -> "Classifier":
        return MnistEncoder.load(path, Classifier)