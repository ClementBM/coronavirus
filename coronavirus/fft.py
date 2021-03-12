import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_fft(y):
    fft = tf.signal.rfft(y)
    f_per_dataset = np.arange(0, len(fft))

    plt.figure(figsize=(12, 8))
    plt.step(f_per_dataset, np.abs(fft))
    plt.xscale("log")
    _ = plt.xlabel("Frequency (log scale)")
