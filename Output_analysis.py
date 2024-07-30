from Nets.EfficientNet import EfficientNet
from Nets.Aggregation import GCN, WeightAggregate, PoolAggregate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import shutil
import random
import pickle
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, TerminateOnNaN
from tqdm import tqdm
import datetime
import warnings

import torch
import torch.nn as nn
batch_size = 64
img_height = 64
img_width = 256
max_label_length = 25  # L
num_primitive_representations = 5  # n
alphabets = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_classes = len(alphabets)+3

def plot_feature_map(image_original, pooling_aggregator_f3_output, pooling_aggregator_f5_output, pooling_aggregator_f7_output):
    # Visualize the feature maps vertically
    plt.figure()  # Adjust figsize based on your preference
    # Plot the original image
    plt.subplot(4, 1, 1)
    plt.imshow(image_original.numpy().squeeze())  # Assuming 'img' contains the original image
    plt.title('Original Image')
    plt.axis('off')
    # Plot the feature map f3
    plt.subplot(4, 1, 2)
    plt.imshow(pooling_aggregator_f3_output.numpy().squeeze())
    plt.title('Feature Map f3')
    plt.axis('off')

    # Plot the feature map f5
    plt.subplot(4, 1, 3)
    plt.imshow(pooling_aggregator_f5_output.numpy().squeeze())
    plt.title('Feature Map f5')
    plt.axis('off')

    # Plot the feature map f7
    plt.subplot(4, 1, 4)
    plt.imshow(pooling_aggregator_f7_output.numpy().squeeze())
    plt.title('Feature Map f7')
    plt.axis('off')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def plot_heat_map(image_original, weighted_aggregator_f3_output, weighted_aggregator_f5_output, weighted_aggregator_f7_output):
    # Visualize the feature maps vertically
    plt.figure()  # Adjust figsize based on your preference
    # Plot the original image
    plt.subplot(4, 1, 1)
    plt.imshow(image_original.numpy().squeeze())  # Assuming 'img' contains the original image
    plt.title('Original Image')
    plt.axis('off')
    # Plot the feature map f3
    plt.subplot(4, 1, 2)
    plt.imshow(weighted_aggregator_f3_output.numpy().squeeze())
    plt.title('Heat Map f3')
    plt.axis('off')

    # Plot the feature map f5
    plt.subplot(4, 1, 3)
    plt.imshow(weighted_aggregator_f5_output.numpy().squeeze())
    plt.title('Heat Map f5')
    plt.axis('off')

    # Plot the feature map f7
    plt.subplot(4, 1, 4)
    plt.imshow(weighted_aggregator_f7_output.numpy().squeeze())
    plt.title('Heat Map f7')
    plt.axis('off')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def plot_PREN_image(image_original, primitive_representation_output):
    plt.subplot(2, 1, 1)
    plt.imshow(image_original.numpy().squeeze())  # Assuming 'img' contains the original image
    plt.title('Original Image')
    plt.axis('off')

    # Plot the primitive representation output
    plt.subplot(2, 1, 2)
    plt.imshow(primitive_representation_output.numpy().squeeze())
    plt.title('Primitive Representation Output')
    plt.axis('off')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def customDataGenerator(paths, labels, batch_size, input_size, is_training=True):
    def process_data(path, label):
        img_string = tf.io.read_file(path)
        img = tf.image.decode_image(img_string, channels=1, dtype=tf.float32, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.image.grayscale_to_rgb(img)

        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(50000)

    dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size,
                                                                                   drop_remainder=True if is_training else False).prefetch(
        tf.data.AUTOTUNE)
    dataset = dataset.repeat()

    return dataset


class FeatureExtractionModule(tf.keras.Model):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()

        self.enb3 = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3)
        )

        for layer in self.enb3.layers:
            layer.trainable = False

        self.block3 = tf.keras.Model(self.enb3.input, self.enb3.get_layer('block3c_add').output, name='block3')
        self.block5 = tf.keras.Model(self.enb3.input, self.enb3.get_layer('block5e_add').output, name='block5')
        self.block7 = tf.keras.Model(self.enb3.input, self.enb3.get_layer('block7b_add').output, name='block7')

        for layer in self.block7.layers:
            layer.trainable = True

    @tf.function
    def call(self, input):

        block3_output = self.block3(input)
        block5_output = self.block5(input)
        block7_output = self.block7(input)

        return (block3_output, block5_output, block7_output)


class CustomActivation(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomActivation, self).__init__()
        self.swish = tf.nn.silu  # x * sigmoid(x)

    @tf.function
    def call(self, input):
        return self.swish(input)


class PoolingAggregatorSubblock(tf.keras.Model):
    def __init__(self, d0, d):
        super(PoolingAggregatorSubblock, self).__init__()

        self.pool_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(d0, 3, 2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001),
            CustomActivation(),
            tf.keras.layers.Conv2D(d, 3, 2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])

    @tf.function
    def call(self, input):
        return tf.expand_dims(self.pool_block(input), axis=1)


class PoolingAggregator(tf.keras.Model):
    def __init__(self, n, d0, d):
        super(PoolingAggregator, self).__init__()
        self.blocks = []
        self.d = d
        for i in range(n):
            self.blocks.append(PoolingAggregatorSubblock(d0, d))

    @tf.function
    def call(self, input):

        P = None

        for block in self.blocks:
            x = block(input)
            if (P is None):
                P = x
            else:
                P = tf.concat([P, x], axis=1)

        return P


class WeightedAggregator(tf.keras.Model):
    def __init__(self, n, d0, d):
        super(WeightedAggregator, self).__init__()

        self.conv_d = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4 * d0, 3, 1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001),
            CustomActivation(),
            tf.keras.layers.Conv2D(d, 1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001)
        ])

        self.conv_n = tf.keras.Sequential([
            tf.keras.layers.Conv2D(d0, 3, 1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001),
            CustomActivation(),
            tf.keras.layers.Conv2D(n, 1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.01, epsilon=0.001),
            tf.keras.layers.Activation('sigmoid')
        ])

    @tf.function
    def call(self, input):
        Z = self.conv_d(input)
        H = self.conv_n(input)

        Z = tf.transpose(Z, perm=[0, 3, 1, 2])
        H = tf.transpose(H, perm=[0, 3, 1, 2])

        Z = tf.transpose(tf.reshape(Z, [-1, Z.shape[1], Z.shape[2] * Z.shape[3]]), perm=[0, 2, 1])
        H = tf.reshape(H, [-1, H.shape[1], H.shape[2] * H.shape[3]])

        P = tf.matmul(H, Z)

        return P


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, L):
        super(GCNLayer, self).__init__()
        self.L = L

    def build(self, input_shape):
        self.B = self.add_weight('B', shape=[self.L, input_shape[1]], trainable=True)

        self.W = self.add_weight('W', shape=[input_shape[2], input_shape[2]], trainable=True)

    @tf.function
    def call(self, P):
        return tf.matmul(tf.matmul(self.B, P), self.W)


class PrimitiveRepresentationLearningModule(tf.keras.Model):
    def __init__(self, L, n, d03, d05, d07, d):
        super(PrimitiveRepresentationLearningModule, self).__init__()

        self.pa1 = PoolingAggregator(n, d03, d // 3);
        self.pa2 = PoolingAggregator(n, d05, d // 3);
        self.pa3 = PoolingAggregator(n, d07, d // 3);

        self.wa1 = WeightedAggregator(n, d03, d // 3);
        self.wa2 = WeightedAggregator(n, d05, d // 3);
        self.wa3 = WeightedAggregator(n, d07, d // 3);

        self.concat1 = tf.keras.layers.concatenate
        self.concat2 = tf.keras.layers.concatenate

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.gcn1 = GCNLayer(L)
        self.gcn2 = GCNLayer(L)

        self.activation = CustomActivation()

    @tf.function
    def call(self, input):
        pa1_output = self.pa1(input[0])  # Primitive Representation 1 (nXd/3)
        pa2_output = self.pa2(input[1])  # Primitive Representation 2 (nXd/3)
        pa3_output = self.pa3(input[2])  # Primitive Representation 3 (nXd/3)

        wa1_output = self.wa1(input[0])  # Primitive Representation 1 (nXd/3)
        wa2_output = self.wa2(input[1])  # Primitive Representation 2 (nXd/3)
        wa3_output = self.wa3(input[2])  # Primitive Representation 3 (nXd/3)

        P1 = self.concat1([pa1_output, pa2_output, pa3_output],
                          axis=-1)  # Concatenated Primitive Representation 1 (nXd)
        P2 = self.concat1([wa1_output, wa2_output, wa3_output],
                          axis=-1)  # Concatenated Primitive Representation 2 (nXd)

        Y1 = self.activation(self.dropout(self.gcn1(P1)))  # Visual Text Representation 1 (LXd)
        Y2 = self.activation(self.dropout(self.gcn2(P2)))  # Visual Text Representation 2 (LXd)

        return tf.add(Y1, Y2) / 2.  # Fused Visual Text Representation (Lxd)


class PREN(tf.keras.Model):
    def __init__(self, L, n, num_classes):
        super(PREN, self).__init__()

        self.featureExtractor = FeatureExtractionModule()

        self.d03 = self.featureExtractor.get_layer('block3').output.shape[-1]
        self.d05 = self.featureExtractor.get_layer('block5').output.shape[-1]
        self.d07 = self.featureExtractor.get_layer('block7').output.shape[-1]
        self.d = self.featureExtractor.get_layer('block7').output.shape[-1]

        self.primitiveRepresentationLearner = PrimitiveRepresentationLearningModule(L, n, self.d03, self.d05, self.d07,
                                                                                    self.d)

        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    @tf.function
    def call(self, input):
        f3, f5, f7 = self.featureExtractor(input)
        Y = self.primitiveRepresentationLearner((f3, f5, f7))
        logits = self.fc(Y)  # (LXnum_classes)

        return logits

# 1. Prepare the image
image_path = "D:\Thac_si\Quiz\Thac_si_quiz\PycharmProject\Tri_tue_nhan_tao_nang_cao\PREN_Model_project\samples\word_1.png"
image = tf.io.read_file(image_path)
image_original = tf.image.decode_png(image, channels=3)
image = tf.image.decode_png(image, channels=3)  # Adjust channels based on your image format
image = tf.image.resize(image, (img_height, img_width))
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# 2. Create an instance of the PREN model
pren_model = PREN(L=max_label_length, n=num_primitive_representations, num_classes=num_classes)

# 3. Pass the image through the model to get the feature maps
f3, f5, f7 = pren_model.featureExtractor(image)

# 4. Extract the output of the WeightedAggregator layer and visualize it
weighted_aggregator_f3_output = pren_model.primitiveRepresentationLearner.wa1(f3)
weighted_aggregator_f5_output = pren_model.primitiveRepresentationLearner.wa2(f5)  # Use any layer you want to visualize
weighted_aggregator_f7_output = pren_model.primitiveRepresentationLearner.wa3(f7)

pooling_aggregator_f3_output = pren_model.primitiveRepresentationLearner.pa1(f3)
pooling_aggregator_f5_output = pren_model.primitiveRepresentationLearner.pa2(f5)  # Use any layer you want to visualize
pooling_aggregator_f7_output = pren_model.primitiveRepresentationLearner.pa3(f7)

# 5. Extract the output of the PrimitiveRepresentationLearningModule and visualize it
primitive_representation_output = pren_model.primitiveRepresentationLearner((f3, f5, f7))

# plot_feature_map(image_original, pooling_aggregator_f3_output, pooling_aggregator_f5_output, pooling_aggregator_f7_output)
# plot_heat_map(image_original, weighted_aggregator_f3_output, weighted_aggregator_f5_output, weighted_aggregator_f7_output)
plot_PREN_image(image_original, primitive_representation_output)
