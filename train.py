import os

import typer

from src.dataset.aid import AID
from src.dataset.opssat import get_eval_dataset, TrainDataset
from src.train_test import Trainer

app = typer.Typer()
input_shape = (200, 200, 3)


@app.command()
def train(batch_size: int = 32, model_weights: str = None, pretrain_classes: int = None,
          epochs: int = 1000, steps_per_epoch: int = 100, verbose: int = 2,
          validation_dataset_path: str = 'datasets/opssat/val/',
          test_dataset_path='datasets/opssat/val/', run_on_gpu: bool = True):
    if not run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    trainer = Trainer(input_shape, num_classes=8)
    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=batch_size)

    if model_weights:
        trainer.load_model(model_weights, pretrain_classes=pretrain_classes)
    else:
        trainer.create_model()

    validation_dataset = get_eval_dataset(validation_dataset_path, input_shape, batch_size=batch_size)
    validation_steps = validation_dataset.cardinality().numpy()

    trainer.train(dataset.iterator, dataset.class_weights, batch_size=batch_size, epochs=epochs,
                  steps_per_epoch=steps_per_epoch, verbose=verbose, validation_data=validation_dataset,
                  validation_steps=validation_steps)


@app.command()
def eval(batch_size: int = 32, model_weights: str = None, verbose: int = 2, run_on_gpu: bool = True,
         validation_dataset_path: str = 'datasets/opssat/val/'):
    if not run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    trainer = Trainer(input_shape, num_classes=8)

    if model_weights:
        trainer.load_model(model_weights)
    else:
        trainer.create_model()

    validation_dataset = get_eval_dataset(validation_dataset_path, input_shape, batch_size=batch_size)
    trainer.evaluate(dataset_iterator=validation_dataset, batch_size=batch_size, verbose=verbose)


@app.command()
def pretrain(batch_size: int = 32, model_weights: str = None,
             epochs: int = 1000, verbose: int = 2,
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
                  steps_per_epoch=dataset.train_steps, verbose=verbose, validation_data=dataset.validation_iterator,
                  validation_steps=dataset.validation_steps)


if __name__ == "__main__":
    app()
