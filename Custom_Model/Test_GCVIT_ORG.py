import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import cv2  # Import OpenCV for image reading
import tqdm
import logging
import sys

# Set TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)  # Change to ERROR to suppress INFO and WARNING logs

# Set the GPU device to 2 (if applicable)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set only GPU 2 as visible
        tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[2], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
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

# Preprocess the image
def preprocess_image(image_path, target_size=(316, 354)):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0

    # Expand dimensions to match the model input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image

# Test the model on a single image
def test_model(model, image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Temporarily suppress TensorFlow output during predictions
    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass

    original_stdout = sys.stdout  # Save original stdout
    sys.stdout = DummyFile()  # Redirect stdout to suppress output

    predictions = model.predict(image)  # Make predictions

    sys.stdout = original_stdout  # Restore original stdout

    return predictions

if __name__ == "__main__":
    # Load the model weights
    ckpt_path = "/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/Ckpt_office_05122024/GCViT_V6_For_office_ORG354X316_05122024_.keras"
    model = load_gcvit_model(ckpt_path)

    # Specify the CSV file path that contains image paths
    csv_input_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/DATA_354X316/0512_ALL_ORIGNAL_4495.csv'

    # Load the image paths from the CSV file
    image_df = pd.read_csv(csv_input_path)

    # Ensure the first column contains the image paths
    image_df['PLiveT'] = 0.0  # Initialize the column for Live probabilities
    image_df['PSpoofT'] = 0.0  # Initialize the column for Spoof probabilities

    # Process each image path and make predictions
    for index, row in tqdm.tqdm(image_df.iterrows(), total=image_df.shape[0]):
        img_path = row[0]  # Get the image path from the first column

        # Test the model on the current image
        predictions = test_model(model, img_path)

        # Extract live and spoof probabilities
        plive_t = predictions[0][0]  # Probability for Live
        pspoof_t = predictions[0][1]  # Probability for Spoof

        # Store the results in the DataFrame
        image_df.at[index, 'PLiveT'] = plive_t
        image_df.at[index, 'PSpoofT'] = pspoof_t

    # Save the updated DataFrame to a new CSV file
    output_csv_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/Result/result_09_12/Pred_GCVit_ORG_0512_ALL_ORIGNAL_4495_keras.csv'
    image_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")



# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import pandas as pd
# import cv2  # Import OpenCV for image reading
# import tqdm
# import logging
# import sys
# from concurrent.futures import ThreadPoolExecutor, as_completed  # Import ThreadPoolExecutor

# # Set TensorFlow logging level
# tf.get_logger().setLevel(logging.ERROR)  # Change to ERROR to suppress INFO and WARNING logs

# # Set the GPU device to 2 (if applicable)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Set only GPU 2 as visible
#         tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[2], True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         print(e)

# # Patch Embedding Layer
# class PatchEmbedding(layers.Layer):
#     def __init__(self, patch_size, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.projection = layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size)

#     def call(self, inputs):
#         x = self.projection(inputs)
#         x = tf.reshape(x, [tf.shape(inputs)[0], -1, self.embed_dim])
#         return x

# # Multi-Head Attention Layer
# def multi_head_self_attention(query, key, value, num_heads):
#     attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=query.shape[-1])(query, key, value)
#     return attention_output

# # Transformer Block
# def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1):
#     norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
#     attention_output = multi_head_self_attention(norm1, norm1, norm1, num_heads)
#     x = layers.Add()([x, attention_output])

#     norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
#     mlp = keras.Sequential([
#         layers.Dense(mlp_dim, activation=tf.nn.gelu),
#         layers.Dense(embed_dim),
#         layers.Dropout(dropout)
#     ])
#     mlp_output = mlp(norm2)
#     x = layers.Add()([x, mlp_output])

#     return x

# # GCVIT Model for 320x384x3 input
# def create_gcvit_model(input_shape, num_patches, embed_dim, num_heads, mlp_dim, num_blocks, num_classes):
#     inputs = keras.Input(shape=input_shape)

#     # Patch Embedding
#     patches = PatchEmbedding(patch_size=16, embed_dim=embed_dim)(inputs)

#     # Transformer Blocks
#     for _ in range(num_blocks):
#         patches = transformer_block(patches, embed_dim, num_heads, mlp_dim)

#     # Global Average Pooling
#     representation = layers.LayerNormalization(epsilon=1e-6)(patches)
#     representation = layers.GlobalAveragePooling1D()(representation)

#     # Classification Head
#     outputs = layers.Dense(num_classes, activation='softmax')(representation)

#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model

# # Load the model once (renamed to avoid conflict)
# def load_gcvit_model(ckpt_path):
#     input_shape = (354, 316, 3)
#     num_patches = (354 // 16) * (316 // 16)  # Patch size of 16x16
#     embed_dim = 128  # Embedding dimension
#     num_heads = 8  # Number of attention heads
#     mlp_dim = 256  # MLP dimensions
#     num_blocks = 10  # Transformer blocks
#     num_classes = 2  # Change as per your classification task

#     # Instantiate the model
#     model = create_gcvit_model(input_shape, num_patches, embed_dim, num_heads, mlp_dim, num_blocks, num_classes)

#     # Build the model by passing a dummy input
#     dummy_input = tf.random.normal([1, 354, 316, 3])  # Batch size 1, image size 384x320, 3 channels
#     model(dummy_input)  # This builds the model

#     # Load the model weights
#     model.load_weights(ckpt_path)
#     print("Model weights loaded successfully!")

#     return model

# # Preprocess the image
# def preprocess_image(image_path, target_size=(316, 354)):
#     # Load the image
#     image = cv2.imread(image_path)
#     # Resize the image to the target size
#     image = cv2.resize(image, target_size)
#     # Convert image to float32 and normalize
#     image = image.astype(np.float32) / 255.0
#     # Expand dimensions to match the model input shape (batch size, height, width, channels)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Test the model on a single image
# def test_model(model, image_path):
#     # Preprocess the image
#     image = preprocess_image(image_path)

#     # Temporarily suppress TensorFlow output during predictions
#     class DummyFile(object):
#         def write(self, x): pass
#         def flush(self): pass

#     original_stdout = sys.stdout  # Save original stdout
#     sys.stdout = DummyFile()  # Redirect stdout to suppress output

#     predictions = model.predict(image)  # Make predictions

#     sys.stdout = original_stdout  # Restore original stdout

#     return predictions

# # Function to process a single image and return the results
# def process_image(model, image_path, index):
#     # Test the model on the current image
#     predictions = test_model(model, image_path)

#     # Extract live and spoof probabilities
#     plive_t = predictions[0][0]  # Probability for Live
#     pspoof_t = predictions[0][1]  # Probability for Spoof

#     # Return the results along with the index
#     return index, plive_t, pspoof_t

# if __name__ == "__main__":
#     # Load the model weights
#     ckpt_path = "/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/GCViT_V5_ORG354X316_13102024_RESUME.keras"
#     model = load_gcvit_model(ckpt_path)

#     # Specify the CSV file path that contains image paths
#     csv_input_path = '/data/DATA/FINGERPRINT_DATA/DATA_VER5.0_JUNE2024/CSV_FILE/Full_image_CSV/1310_FINAL_VAL_EX_MELO31_ISSUE_39000.csv'  # Update this with your actual CSV file path

#     # Load the image paths from the CSV file
#     image_df = pd.read_csv(csv_input_path)

#     # Ensure the first column contains the image paths
#     image_df['PLiveT'] = 0.0  # Initialize the column for Live probabilities
#     image_df['PSpoofT'] = 0.0  # Initialize the column for Spoof probabilities

#     # Use ThreadPoolExecutor for parallel processing
#     max_workers = min(8, os.cpu_count())
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit tasks to the executor
#         futures = [
#             executor.submit(process_image, model, row[0], index)
#             for index, row in image_df.iterrows()
#         ]

#         # Collect results as tasks complete
#         for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
#             index, plive_t, pspoof_t = future.result()

#             # Store the results in the DataFrame
#             image_df.at[index, 'PLiveT'] = plive_t
#             image_df.at[index, 'PSpoofT'] = pspoof_t

#     # Save the updated DataFrame to a new CSV file
#     output_csv_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/Result/Pred_GCVIT_ORG_1310_Master_CSV_VAL_keras.csv'
#     image_df.to_csv(output_csv_path, index=False)

#     print(f"Results saved to {output_csv_path}")
