import os

import typer

from src.dataset.opssat import TrainDataset
from src.train_test import Trainer

app = typer.Typer()


@app.command()
def train(batch_size: int = 32, model_weights: str = None,
          epochs: int = 1000, steps_per_epoch: int = 100, verbose: int = 0,
          validation_dataset_path: str = 'datasets/opssat/val/',
          test_dataset_path='datasets/opssat/val/', run_on_gpu: bool = True):
    if not run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    input_shape = (200, 200, 3)
    trainer = Trainer(input_shape)
    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=batch_size)

    if model_weights:
        trainer.load_model(model_weights)
    else:
        trainer.create_model()
    trainer.train(dataset.iterator, dataset.class_weights, batch_size=batch_size, epochs=epochs,
                  steps_per_epoch=steps_per_epoch, verbose=verbose, validation_dataset_path=validation_dataset_path)
    trainer.evaluate(test_dataset_path=test_dataset_path)


if __name__ == "__main__":
    app()
