from typing import *
from classifier import Classifier

def main() -> None:
    model = Classifier(
        device="cuda:0",
        lr=0.001,
        latent_size=28
    )

    model.train(
        epochs=1000,
        batch=0.2
    )

    model.save("classifier")

if __name__ == "__main__":
    main()
