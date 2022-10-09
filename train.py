import typer

from src.dataset.opssat import TrainDataset
from src.train_test import Trainer

app = typer.Typer()


@app.command()
def train(batch_size: int = 32, model_weights: str = None,
          epochs: int = 1000, steps_per_epoch: int = 100, verbose: int = 0):

    input_shape = (200, 200, 3)
    trainer = Trainer(input_shape)
    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=batch_size)

    if model_weights:
        trainer.load_model(model_weights)
    else:
        trainer.create_model()
    trainer.train(dataset.iterator, dataset.class_weights, batch_size=batch_size, epochs=epochs,
                  steps_per_epoch=steps_per_epoch, verbose=verbose)
    trainer.evaluate()


if __name__ == "__main__":
    app()
