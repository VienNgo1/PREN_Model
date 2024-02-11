import os
import time
import shutil
import random
import pickle
import datetime
import itertools
import random as rn
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import warnings


def plot_csv_file(test_df, head,title, xlabel, ylabel):
    plt.figure(figsize=(16, 20))
    ax = sns.barplot(x=test_df[head].value_counts().index,
                     y=test_df[head].value_counts().values, ci=False, palette='Accent',
                     orient='h')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.margins(0.1, 0.01)

    for p in ax.patches:
        width = p.get_width()
        plt.text(4 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
                 '{:1.1f}%'.format(width),
                 ha='center', va='center')
    plt.show()


def view_images(csv_file, indices, case, size):
    test_df = pd.read_csv(csv_file, encoding='latin-1')
    paths = test_df.image_paths.values[indices]
    labels = test_df.ground_truths.values[indices]
    predictions = test_df.predicts.values[indices]

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle('\n{} Images'.format(case), y=0.95)
    for i in range(size):
        ax = fig.add_subplot(5, size//5 + (1 if size % 5 else 0), i + 1, xticks=[], yticks=[])
        paths[i] = paths[i].replace('\\', '/')

        # gray scale image
        img_string = tf.io.read_file(paths[i])
        img = tf.image.decode_image(img_string, channels=1, dtype=tf.dtypes.float32, expand_animations=False)
        img = tf.image.resize(img, [64, 256])
        img = tf.image.grayscale_to_rgb(img)

        # original images
        # img = tf.image.decode_image(img_string, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
        # img = tf.image.resize(img, [64, 256])

        original = labels[i]
        predicted = predictions[i]
        parts = paths[i].split('/')
        desired_path = '/'.join(parts[-1:])
        ax.imshow(img)
        ax.set_title(f"Original: {original}, Predicted: {predicted}\nDirectory: {desired_path}")

    plt.axis('off')
    plt.show()
