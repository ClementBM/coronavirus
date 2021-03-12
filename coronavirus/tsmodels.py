import datetime
from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import History
import os

from coronavirus import tswindow

MAX_EPOCHS = 10


class TsBaseModel(tf.keras.Model):
    history_losses: List
    history_val_losses: List
    history: History

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs, training=False):
        if self.label_index is None:
            return self._pipeline(inputs)

        inputs = inputs[:, :, self.label_index]
        inputs = inputs[:, :, tf.newaxis]
        return self._pipeline(inputs)

    def _pipeline(self, inputs):
        pass

    def compile_and_fit(
        self,
        window: tswindow.WindowGenerator,
        patience=2,
        checkpoint=False,
        tensorboard=True,
    ):
        self.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()],
        )

        callbacks = []
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )
        callbacks.append(early_stopping)

        if tensorboard:
            log_dir = (
                f"logs/{self.__class__.__name__}/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            tensorboard_callback = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
            )
            callbacks.append(tensorboard_callback)

        if checkpoint:
            # Directory where the checkpoints will be saved
            checkpoint_dir = "./training_checkpoints"
            # Name of the checkpoint files
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix, save_weights_only=True
            )
            callbacks.append(checkpoint_callback)

        history = self.fit(
            window.train,
            epochs=MAX_EPOCHS,
            validation_data=window.validation,
            callbacks=callbacks,
            shuffle=True,
        )

        return history

    def evaluate_on_validation(self, window: tswindow.WindowGenerator):
        """
        loss value & metrics values
        """
        return self.evaluate(window.validation)

    def evaluate_on_test(self, window: tswindow.WindowGenerator):
        """
        loss value & metrics values
        """
        return self.evaluate(window.test)

    # def accuracies(self):
    #     train_accuracy = metrics.accuracy_score(self.y_train, self.y_train_predicted)
    #     test_accuracy = metrics.accuracy_score(self.y_test, self.y_test_predicted)

    #     return train_accuracy, test_accuracy
