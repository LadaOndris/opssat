import os

import typer

from src.dataset.aid import AID
from src.dataset.opssat import get_images_from_path, TrainDataset
from src.train_test import Trainer

app = typer.Typer()
input_shape = (200, 200, 3)


@app.command()
def train(batch_size: int = 32, model_weights: str = None,
          epochs: int = 1000, steps_per_epoch: int = 100, verbose: int = 1,
          validation_dataset_path: str = 'datasets/opssat/val/',
          test_dataset_path='datasets/opssat/val/', run_on_gpu: bool = True):
    if not run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    trainer = Trainer(input_shape, num_classes=8)
    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=batch_size)

    if model_weights:
        trainer.load_model(model_weights)
    else:
        trainer.create_model()

    validation_data = get_images_from_path(validation_dataset_path, input_shape)
    trainer.train(dataset.iterator, dataset.class_weights, batch_size=batch_size, epochs=epochs,
                  steps_per_epoch=steps_per_epoch, verbose=verbose, validation_data=validation_data)
    trainer.evaluate(test_dataset_path=test_dataset_path)


@app.command()
def pretrain(batch_size: int = 32, model_weights: str = None,
             epochs: int = 1000, steps_per_epoch: int = 100, verbose: int = 1,
             dataset_path: str = 'datasets/AID/',
             validation_split: float = 0.1,
             run_on_gpu: bool = True):
    if not run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    trainer = Trainer(input_shape, num_classes=30)
    dataset = AID(dataset_path, batch_size=batch_size, image_size=input_shape[:2],
                  validation_split=validation_split)

    if model_weights:
        trainer.load_model(model_weights)
    else:
        trainer.create_model()
    trainer.train(dataset.train_iterator, dataset.class_weights, batch_size=batch_size, epochs=epochs,
                  steps_per_epoch=steps_per_epoch, verbose=verbose, validation_data=dataset.validation_iterator)


if __name__ == "__main__":
    app()
