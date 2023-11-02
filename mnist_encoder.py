from typing import *
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch import Tensor
from PIL.Image import Image
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.datasets import MNIST
import torch
import matplotlib.pyplot as plt

Device = Literal["cpu", "cuda:0", "cuda:1"]

T = TypeVar("T")
C = TypeVar("C", bound="MnistEncoder")

class MiniBatch(NamedTuple):
    data: Tensor
    labels: Tensor

class MnistEncoder(ABC, Generic[T]):

    def __init__(self, 
                 train_size: float,
                 device: Device|torch.device,
                 lr: float,
                 output_layers: torch.nn.Module) -> None:

        mnist = MNIST(
            root=".", 
            download=True
        )

        self.mnist_size = mnist.data.shape[0]

        data = mnist.data.float() / 255

        assert 0 < train_size <= 1.0
        self.train_size = int(self.mnist_size*train_size)

        self._train_data = data[:self.train_size]
        self._train_target = mnist.targets[:self.train_size]
        self._train_target_one_hot = torch.nn.functional.one_hot(self._train_target)\
            .float()

        self._test_data = data[self.train_size:]
        self._test_target = mnist.targets[self.train_size:]
        self._test_target_one_hot = torch.nn.functional.one_hot(self._test_target)\
            .float()

        self.lr = lr
        self.device = torch.device(device)

        self.latent_size = 0
        for layer in output_layers.parameters():
            self.latent_size = layer.shape[1]
            break

        assert self.latent_size > 0

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28*28, out_features=14*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=14*28, out_features=7*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=7*28,out_features=self.latent_size),
            torch.nn.Sigmoid()
        )

        self.output_layers = output_layers
        
        self._logits = torch.nn.Sequential(
            self.encoder,
            self.output_layers
        ).to(device=self.device)

        for param in self._logits.parameters():
            self.dtype = param.dtype
            break

        self._optimizer = torch.optim.Adam(
            params=self._logits.parameters(),
            lr=lr
        )

        self.tensor_to_pil = ToPILImage()
        self.pil_to_tensor = PILToTensor()

    def _input_transform(self, input: Image) -> Tensor:
        return self.pil_to_tensor(input)

    @abstractmethod
    def _output_transform(self, output: Tensor) -> T:
        pass

    def predict(self, input: Image) -> T:
        return self._output_transform(self.forward(self._input_transform(input)))

    def forward(self, tensor: Tensor, use_grad: bool = False) -> Tensor:
        src_device = tensor.device
        tensor = tensor.to(device=self.device, dtype=self.dtype) / tensor.max()
        if use_grad:
            y_hat = cast(Tensor, self._logits.forward(tensor))
        else:
            with torch.no_grad():
                y_hat = cast(Tensor, self._logits.forward(tensor))

        return y_hat.to(device=src_device)

    def minibatch(self, percent: float) -> MiniBatch:
        assert 0 < percent <= 1.0
        size = int(self._train_data.shape[0])
        randperm = torch.randperm(size)[:int(size*percent)]
        return MiniBatch(self._train_data[randperm], self._train_target_one_hot[randperm])
    
    def get_image(self, index: int) -> Image:
        return self.tensor_to_pil(self._test_data[index])

    @abstractmethod
    def fit(self, x: Tensor, y: Tensor) -> float:
        pass

    def train(self, epochs: int, batch: float, plot_stats: bool = False) -> None:
        progress_bar = tqdm(range(epochs))
        losses: List[float] = []
        for epoch in progress_bar:
            loss = self.fit(*self.minibatch(batch))
            progress_bar.set_description(f"{loss=:.8f}")
            losses.append(loss)

        if plot_stats:
            plt.plot(torch.arange(len(losses)), losses, label="Loss curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def save(self, path: str, suffix: str|None = None) -> None:
        if suffix is None:
            suffix = self.__class__.__name__.lower()
        torch.save(obj=self, f=f"{path}.{suffix}")

    @staticmethod
    def load(path: str, type: Type[C]) -> C:
        obj = torch.load(f=path)
        assert isinstance(obj, type)
        return obj