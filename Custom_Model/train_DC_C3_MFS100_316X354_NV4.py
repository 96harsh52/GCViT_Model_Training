import tensorflow as tf

# Set the GPU device to 2 (if applicable)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set only GPU 2 as visible
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)



# Optimized CNN model for 640X224X1 input with Depthwise Separable Convolution
def create_improved_CNN_DPC(input_shape=(316, 354, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    # Rescaling (normalization) layer
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2 (Deeper with 5x5 kernel)
    x = tf.keras.layers.DepthwiseConv2D((5, 5), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 3 (Deeper with 7x7 kernel)
    x = tf.keras.layers.DepthwiseConv2D((7, 7), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual Block 4
    residual = x

    # Adjust residual to match the number of channels in 'x' (256)
    residual = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(residual)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Now add 'x' and 'residual', which have matching shapes
    x = tf.keras.layers.add([x, residual])

    # Global Max Pooling + Global Average Pooling
    x_avg = tf.keras.layers.GlobalAveragePooling2D()(x)
    x_max = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([x_avg, x_max])

    # Dense Layers with Dropout
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Increased dropout for better generalization

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Model creation
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
# Create and print model summary
model = create_improved_CNN_DPC(input_shape=(316, 354, 3))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),  # Lowered learning rate for better convergence
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

# Print the model summary
model.summary()



###########################################################
# Training  part 
###################################################################

import pandas as pd
import numpy as np
import os 
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



# Custom data generator for image files with shape (316, 354, 3) using a CSV file
class BinDataGeneratorFromCSV(Sequence):
    def __init__(self, csv_file, batch_size=128, target_size=(316, 354, 3), class_mode='binary', shuffle=True):
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
            # Resize the image to the target size
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))  # (width, height)
            
            # Normalize the image (optional)
            img = img / 255.0  # Rescale pixel values to [0, 1]

            X[i,] = img
            y[i] = label

        return X, y



# CSV file paths (Update these paths according to your actual data)
train_csv = "/data/DATA/FINGERPRINT_DATA/DATA_VER5.0_JUNE2024/CSV_FILE/Full_image_CSV/DATA_WITH_FP_FN/DATA_INCLUDE_HO_1909_FP_FN/1610_TRAIN_EX_MELO31_ALL_FP_FN_279313.csv"
test_csv = "/data/DATA/FINGERPRINT_DATA/DATA_VER5.0_JUNE2024/CSV_FILE/Full_image_CSV/DATA_WITH_FP_FN/DATA_INCLUDE_HO_1909_FP_FN/1610_TEST_EX_MELO31_ALL_FP_FN__77904.csv"
val_csv = "/data/DATA/FINGERPRINT_DATA/DATA_VER5.0_JUNE2024/CSV_FILE/Full_image_CSV/DATA_WITH_FP_FN/DATA_INCLUDE_HO_1909_FP_FN/1610_VAL_EX_MELO31_ALL_FP_FN_39000.csv"


# Create data generators
train_generator = BinDataGeneratorFromCSV(csv_file=train_csv, batch_size=128, target_size=(316, 354, 3))
val_generator = BinDataGeneratorFromCSV(csv_file=val_csv, batch_size=128, target_size=(316, 354, 3))
test_generator = BinDataGeneratorFromCSV(csv_file=test_csv, batch_size=128, target_size=(316, 354, 3))

# Checkpoint directory to save the best model
ckpt_ = "CKPT_316X354/DC_C3_CKPT_ORG316X354"
os.makedirs(ckpt_, exist_ok=True)

# Best model checkpoint path
best_model_ckpt = os.path.join(ckpt_, "DC_C3_CKPT_ORG316X354_17102024.keras")

# Define callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint(best_model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)

# Train the model
hist = model.fit(train_generator, validation_data=val_generator, epochs=50, 
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
