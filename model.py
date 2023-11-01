from typing import *
import torch
import torchvision
from alive_progress import alive_bar

Device = Literal["cpu", "cuda:0", "cuda:1"]
class AutoEncoder:

    def __init__(self, 
                 device: Device, 
                 latent_size: int, 
                 lr: float) -> None:
        super().__init__()

        self._data = torchvision.datasets.MNIST(
            root=".", 
            download=True
        ).data.float() / 255

        self.device = torch.device(device)
        
        self._encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28*28, out_features=14*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=14*28,out_features=7*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=7*28,out_features=3*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=3*28,out_features=latent_size),
            torch.nn.ReLU(),
        ).to(device=self.device)

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_size, out_features=3*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=3*28,out_features=7*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=7*28,out_features=14*28),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=14*28,out_features=28*28),
            torch.nn.ReLU(),
        ).to(device=self.device)

        self._optimizer = torch.optim.Adam(
            params=tuple(self._encoder.parameters()) + tuple(self._decoder.parameters()),
            lr=lr
        )

    def forward(self, tensor: torch.FloatTensor, use_grad: bool = False) -> torch.FloatTensor:
        src_device = tensor.device
        tensor = tensor.to(device=self.device)
        if use_grad:
            encoding: torch.FloatTensor = self._encoder.forward(tensor)
            decoding: torch.FloatTensor = self._decoder.forward(encoding)
        else:
            with torch.no_grad():
                encoding: torch.FloatTensor = self._encoder.forward(tensor)
                decoding = self._decoder.forward(encoding)
        return decoding.reshape((-1,28,28)).to(device=src_device)

    def minibatch(self, percent: float) -> torch.FloatTensor:
        assert 0 < percent < 1.0
        size = int(self._data.shape[0])
        randperm = torch.randperm(size)[:int(size*percent)]
        return self._data[randperm]

    def fit_supervised(self, x: torch.FloatTensor, y: int) -> None:
        pass

    def fit_semisupervised(self, x: torch.FloatTensor) -> float:
        self._optimizer.zero_grad()
        y_hat = self.forward(x, use_grad=True)
        loss = torch.mean((y_hat - x)**2)
        loss.backward()
        self._optimizer.step()
        return float(loss)

    def train(self, epochs: int, batch: float) -> None:
        with alive_bar(epochs, bar='halloween') as bar:
            for epoch in range(epochs):
                loss = self.fit_semisupervised(self.minibatch(batch))
                bar.text(f"{loss=:.4f}")
                bar()

    def save(self, path: str) -> None:
        torch.save(obj=self, f=path)

    @staticmethod
    def load(path: str) -> "AutoEncoder":
        obj = torch.load(f=path)
        assert isinstance(obj, AutoEncoder)
        return obj