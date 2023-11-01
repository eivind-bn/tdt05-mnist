from typing import *
import torch
import torchvision
from alive_progress import alive_bar

Device = Literal["cpu", "cuda:0", "cuda:1"]
class Classifier:

    def __init__(self, 
                 device: Device|torch.device, 
                 latent_size: int, 
                 lr: float) -> None:
        super().__init__()

        mnist = torchvision.datasets.MNIST(
            root=".", 
            download=True
        )
        
        data = mnist.data.float() / 255
        test_size = int(data.shape[0]*0.8)

        self._train_data = data[:test_size]
        self._train_target = mnist.targets[:test_size]

        self._test_data = data[test_size:]
        self._test_target = mnist.targets[test_size:]

        self.latent_size = latent_size
        self.lr = lr
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

        self._classifier = torch.nn.Linear(in_features=latent_size, out_features=10)\
            .to(self.device)

        self._optimizer = torch.optim.Adam(
            params=tuple(self._encoder.parameters()) + tuple(self._classifier.parameters()),
            lr=lr
        )

    def forward(self, tensor: torch.FloatTensor, use_grad: bool = False) -> torch.FloatTensor:
        src_device = tensor.device
        tensor = tensor.to(device=self.device)
        if use_grad:
            encoding: torch.FloatTensor = self._encoder.forward(tensor)
            classification: torch.FloatTensor = self._classifier.forward(encoding)
        else:
            with torch.no_grad():
                encoding: torch.FloatTensor = self._encoder.forward(tensor)
                classification = self._classifier.forward(encoding)
        return classification.to(src_device)

    def minibatch(self, percent: float, train: bool) -> Tuple[torch.FloatTensor,torch.IntTensor]:
        assert 0 < percent < 1.0
        if train:
            data = self._train_data
            targets = self._train_target
        else:
            data = self._test_data
            targets = self._test_target

        size = int(data.shape[0])
        randperm = torch.randperm(size)[:int(size*percent)]
        one_hot = torch.zeros((size,10))
        one_hot[:] = targets[randperm]
        return data[randperm], targets[randperm]

    def fit(self, x: torch.FloatTensor, y: torch.IntTensor) -> float:
        self._optimizer.zero_grad()
        y_hat = self.forward(x, use_grad=True)
        loss = torch.mean((y_hat - y)**2)
        loss.backward()
        self._optimizer.step()
        return float(loss)

    def train(self, epochs: int, batch: float) -> None:
        with alive_bar(epochs, bar='halloween') as bar:
            for epoch in range(epochs):
                data,label = self.minibatch(batch, train=True)
                loss = self.fit(x=data, y=label)
                bar.text(f"{loss=:.4f}")
                bar()

    def set_encoder_params(self, params: Iterable[torch.nn.Parameter]) -> None:
        with torch.no_grad():
            for self_param, other_param in zip(self._encoder.parameters(), params):
                self_param[:] = other_param

    def save(self, path: str) -> None:
        torch.save(obj=self, f=path)

    @staticmethod
    def load(path: str) -> "Classifier":
        obj = torch.load(f=path)
        assert isinstance(obj, Classifier)
        return obj