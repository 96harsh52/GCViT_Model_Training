import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import cv2  # Import OpenCV for image reading
import tqdm
import logging
import sys

# Ensure only GPU 1 is used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # Use GPU 1 (0-indexed)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)

# Patch Embedding Layer
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size)

    def call(self, inputs):
        x = self.projection(inputs)
        x = tf.reshape(x, [tf.shape(inputs)[0], -1, self.embed_dim])
        return x

# Multi-Head Attention Layer
def multi_head_self_attention(query, key, value, num_heads):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=query.shape[-1])(query, key, value)
    return attention_output

# Transformer Block
def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1):
    norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = multi_head_self_attention(norm1, norm1, norm1, num_heads)
    x = layers.Add()([x, attention_output])

    norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp = keras.Sequential([
        layers.Dense(mlp_dim, activation=tf.nn.gelu),
        layers.Dense(embed_dim),
        layers.Dropout(dropout)
    ])
    mlp_output = mlp(norm2)
    x = layers.Add()([x, mlp_output])

    return x

# GCVIT Model for 320x384x3 input
def create_gcvit_model(input_shape, num_patches, embed_dim, num_heads, mlp_dim, num_blocks, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Patch Embedding
    patches = PatchEmbedding(patch_size=16, embed_dim=embed_dim)(inputs)

    # Transformer Blocks
    for _ in range(num_blocks):
        patches = transformer_block(patches, embed_dim, num_heads, mlp_dim)

    # Global Average Pooling
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    # Classification Head
    outputs = layers.Dense(num_classes, activation='softmax')(representation)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load the model once (renamed to avoid conflict)
def load_gcvit_model(ckpt_path):
    input_shape = (354, 316, 3)
    num_patches = (354 // 16) * (316 // 16)  # Patch size of 16x16
    embed_dim = 128  # Embedding dimension
    num_heads = 8  # Number of attention heads
    mlp_dim = 256  # MLP dimensions
    num_blocks = 10  # Transformer blocks
    num_classes = 2  # Change as per your classification task

    # Instantiate the model
    model = create_gcvit_model(input_shape, num_patches, embed_dim, num_heads, mlp_dim, num_blocks, num_classes)

    # Build the model by passing a dummy input
    dummy_input = tf.random.normal([1, 354, 316, 3])  # Batch size 1, image size 384x320, 3 channels
    model(dummy_input)  # This builds the model

    # Load the model weights
    model.load_weights(ckpt_path)
    print("Model weights loaded successfully!")

    return model



if __name__ == "__main__":
    # Load the model weights
    ckpt_path = "/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/GCViT_V5_ORG354X316_29112024_RESUME.keras"
    model = load_gcvit_model(ckpt_path)

    # Now convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    tflite_model_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/TFLITE/GCViT_V5_ORG354X316_29112024_RESUME.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model converted to TensorFlow Lite and saved at {tflite_model_path}")




###############################################################


# if __name__ == "__main__":
#     # Load the model weights
#     ckpt_path = "/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/GCViT_V5_ORG354X316_08102024.keras"
    
#     if not os.path.exists(ckpt_path):
#         logging.error(f"Checkpoint path {ckpt_path} does not exist.")
#         sys.exit(1)

#     try:
#         model = load_gcvit_model(ckpt_path)
#         model.summary()  # Print model summary
#     except Exception as e:
#         logging.error(f"Error loading model: {str(e)}")
#         sys.exit(1)

#     # Now convert the model to TensorFlow Lite format
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional optimization
#     tflite_model = converter.convert()

#     # Save the TensorFlow Lite model to a file
#     tflite_model_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/TFLITE/GCViT_V5_ORG354X316_08102024.tflite'
#     try:
#         with open(tflite_model_path, 'wb') as f:
#             f.write(tflite_model)
#         print(f"Model converted to TensorFlow Lite and saved at {tflite_model_path}")
#     except Exception as e:
#         logging.error(f"Error saving TensorFlow Lite model: {str(e)}")