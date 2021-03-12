from coronavirus import tsmodels
import tensorflow as tf


class MultipleStepRNN(tsmodels.TsBaseModel):
    def __init__(self, output_steps, label_index=None):
        super().__init__(label_index)

        self.output_steps = output_steps
        self.feature_count = 1

        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=False)

        # Shape => [batch, out_steps*features]
        self.dense = tf.keras.layers.Dense(
            units=self.output_steps * self.feature_count,
            kernel_initializer=tf.initializers.zeros,
        )

        # Shape => [batch, out_steps, features]
        self.reshape_out = tf.keras.layers.Reshape(
            [self.output_steps, self.feature_count]
        )

    def _pipeline(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        x = self.reshape_out(x)
        return x