from typing import *
from autoencoder import AutoEncoder

def main() -> None:
    model = AutoEncoder(
        device="cuda:0",
        lr=0.001,
        latent_size=28
    )

    model.train(
        epochs=1000,
        batch=0.2
    )

    model.save("auto_encoder")

if __name__ == "__main__":
    main()
