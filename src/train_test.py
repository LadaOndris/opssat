import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
from tensorflow import keras

import src.logging as logs_utils
from src.dataset.opssat import get_images_from_path
from src.efficientnet_lite import EfficientNetLiteB0


class Trainer:

    def __init__(self, input_shape, num_classes):
        self.model = None
        self.num_classes = num_classes
        self.input_shape = input_shape  # input_shape is (height, width, number of channels) for images

    def create_model(self):
        self.model = EfficientNetLiteB0(classes=self.num_classes, weights=None, input_shape=self.input_shape,
                                        classifier_activation=None)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[keras.metrics.SparseCategoricalAccuracy()])

    def load_model(self, weights_path: str):
        self.model = EfficientNetLiteB0(classes=self.num_classes, weights=None, input_shape=self.input_shape,
                                        classifier_activation=None)
        self.model.load_weights(weights_path)

    def train(self, dataset_iterator, class_weights, batch_size: int, epochs: int, steps_per_epoch: int,
              verbose: int, validation_data, validation_steps: int = 0):
        log_dir = logs_utils.make_log_dir('logs')
        checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
        monitor_loss = 'val_loss'
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch'),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True,
                                               save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor=monitor_loss, patience=20, restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau()
        ]

        history = self.model.fit(dataset_iterator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                 verbose=verbose, batch_size=batch_size, callbacks=callbacks,
                                 class_weight=class_weights,
                                 validation_data=validation_data, validation_steps=validation_steps)

    def evaluate(self, test_dataset_path: str):
        x, y = get_images_from_path(test_dataset_path, self.input_shape)
        self.evaluate_copen_kappa(x, y)

    def evaluate_copen_kappa(self, x, y):
        predictions = np.zeros(len(y), dtype=np.int8)

        # inference loop
        for e, (image, target) in enumerate(zip(x, y)):
            image = np.expand_dims(np.array(image), axis=0)
            output = self.model.predict(image)
            predictions[e] = np.squeeze(output).argmax()

        # Keras model score
        score_keras = 1 - cohen_kappa_score(y.numpy(), predictions)
        print("Score:", score_keras)
