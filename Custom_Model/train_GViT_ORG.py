import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import cv2  # Import OpenCV for image reading

# Check TensorFlow version
print(tf.__version__)

# Set the GPU device to 2 (if applicable)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set only GPU 2 as visible
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

# Custom data generator for image files with shape (316, 354, 3) using a CSV file
class BinDataGeneratorFromCSV(Sequence):
    def __init__(self, csv_file, batch_size=128, target_size=(354, 316, 3), class_mode='binary', shuffle=True):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.filepaths, self.labels = self._load_filepaths_and_labels()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_filepaths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self._generate_data(batch_filepaths, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.filepaths, self.labels))
            np.random.shuffle(temp)
            self.filepaths, self.labels = zip(*temp)

    def _load_filepaths_and_labels(self):
        # Load data from CSV
        data = pd.read_csv(self.csv_file)
        filepaths = data['Image_path'].values
        labels = data['Fake'].values
        return np.array(filepaths), np.array(labels)

    def _generate_data(self, batch_filepaths, batch_labels):
        X = np.empty((self.batch_size, *self.target_size))
        y = np.empty((self.batch_size), dtype=int)

        for i, (filepath, label) in enumerate(zip(batch_filepaths, batch_labels)):
            # Read the image using cv2
            img = cv2.imread(filepath)

            # Check if the image was loaded correctly
            if img is None:
                print(f"Warning: Unable to read image at {filepath}. Skipping this image.")
                continue  # Skip this image if it could not be read

            # Resize the image to the target size
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))  # (width, height)
            
            # Normalize the image (optional)
            img = img / 255.0  # Rescale pixel values to [0, 1]

            X[i,] = img
            y[i] = label

        return X, y

# Patch Embedding Layer
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size)
        # (1,56,56,128)
    def call(self, inputs):
        x = self.projection(inputs) 
        x = tf.reshape(x, [tf.shape(inputs)[0], -1, self.embed_dim])
        return x

# (1,45,45,64)
# (none,418,128)



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

# GCVIT Model for 316x354x3 input
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

# Model Configuration for input size (316, 354, 3)
input_shape = (354, 316, 3)
num_patches = (354 // 16) * (316 // 16)  # Patch size of 16x16
embed_dim = 128  # Embedding dimension
num_heads = 8  # Number of attention heads
mlp_dim = 256  # MLP dimensions
num_blocks = 10  # Transformer blocks
num_classes = 2  # Change as per your classification task

# Create GCVIT model
model = create_gcvit_model(input_shape, num_patches, embed_dim, num_heads, mlp_dim, num_blocks, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# CSV file paths (Update these paths according to your actual data)
Folder_path = "/data/WORKSPACE_HS/Fingerprint_Dataset/Data_office_collect_05122024"
train_csv = os.path.join(Folder_path , "0512_TRAIN_47500.csv")
test_csv = os.path.join(Folder_path , "0512_TEST_13573.csv")
val_csv = os.path.join(Folder_path , "0512_VAL_6797.csv")
# Create data generators
train_generator = BinDataGeneratorFromCSV(csv_file=train_csv, batch_size=64, target_size=(354, 316, 3))
val_generator = BinDataGeneratorFromCSV(csv_file=val_csv, batch_size=64, target_size=(354, 316, 3))
test_generator = BinDataGeneratorFromCSV(csv_file=test_csv, batch_size=64, target_size=(354, 316, 3))

# Checkpoint directory to save the best model
ckpt_ = "/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/Ckpt_office_05122024"
os.makedirs(ckpt_, exist_ok=True)

# Best model checkpoint path
best_model_ckpt = os.path.join(ckpt_, "GCViT_V6_For_office_ORG354X316_05122024_test.keras")

# Define callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint(best_model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)

# Train the model
hist = model.fit(train_generator, validation_data=val_generator, epochs=100, 
                  callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model weights
model.load_weights(best_model_ckpt)

# Evaluate the model on train, validation, and test sets
train_loss, train_acc = model.evaluate(train_generator, verbose=2)
val_loss, val_acc = model.evaluate(val_generator, verbose=2)
test_loss, test_acc = model.evaluate(test_generator, verbose=2)

print("Train Acc: {}, Train Loss: {}".format(train_acc, train_loss))
print("Val Acc: {}, Val Loss: {}".format(val_acc, val_loss))
print("Test Acc: {}, Test Loss: {}".format(test_acc, test_loss))
