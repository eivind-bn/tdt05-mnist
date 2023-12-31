from typing import *
import torch
import random
from torchvision.transforms import ToPILImage, PILToTensor
from PIL.Image import Image
from torch import Tensor
from torch._tensor import Tensor
from classifier import Classifier
from mnist_encoder import *
from matplotlib.axes import Axes

class AutoEncoder(MnistEncoder[Image]):

    def __init__(self, 
                 train_size: float,
                 device: Device|torch.device, 
                 latent_size: int, 
                 lr: float) -> None:
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_size, out_features=7*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=7*28, out_features=14*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=14*28,out_features=28*28),
            torch.nn.ReLU(),
        )

        super().__init__(
            train_size=train_size,
            device=device,
            lr=lr,
            output_layers=self.decoder
        )

        self._mse_loss = torch.nn.MSELoss()

    def fit(self, x: Tensor, _: Tensor) -> float:
        self._optimizer.zero_grad()
        y_hat = self.forward(x, use_grad=True).reshape((-1,28,28))
        loss: Tensor = self._mse_loss(y_hat, x)
        loss.backward()
        self._optimizer.step()
        return float(loss)
    
    def _input_transform(self, input: Image) -> Tensor:
        return self.pil_to_tensor(input)
    
    def _output_transform(self, output: Tensor) -> Image:
        return self.tensor_to_pil(torch.squeeze(output, 0).reshape((28,28)) / output.max())
    
    def to_classifier(self, train_size: float, lr: float|None = None) -> Classifier:
        classifier = Classifier(
            train_size=train_size,
            device=self.device,
            latent_size=self.latent_size,
            lr=self.lr if lr is None else lr
        )

        train_size = int(self.mnist_size*train_size)

        # Train can only shrink in size, but not grow in the new model.
        classifier._train_data = self._train_data[:train_size]
        classifier._train_target = self._train_target[:train_size]
        classifier._train_target_one_hot = self._train_target_one_hot[:train_size]

        # Train data can never become test-data. We simply transfer the old test.
        classifier._test_data = self._test_data
        classifier._test_target = self._test_target
        classifier._test_target_one_hot = self._test_target_one_hot

        classifier.set_encoder_params(self.encoder.parameters())

        return classifier
    
    def plot_predictions(self) -> None:
        fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(10,4))
        for i in range(6):
            rand = random.randint(0,self._test_data.shape[0])
            sample: Image = self._output_transform(self._test_data[rand]) 
            recon = self.predict(sample)
            ax0: Axes = axes[0,i]
            ax1: Axes = axes[1,i]
            ax0.imshow(sample, cmap="gray")
            ax0.axis("off")
            ax1.imshow(recon, cmap="gray")
            ax1.axis("off")

        plt.show()

    @staticmethod
    def load(path: str) -> "AutoEncoder":
        return MnistEncoder.load(path, AutoEncoder)