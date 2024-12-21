import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def Tflite_infer(interpreter, input_details, output_details, input_image):
    input_tensor = preprocess_image(input_image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    probabilities = softmax(output_tensor)

    # sensor_probability = probabilities[0][1]
    # fgm_probability = probabilities[0][0]
    sensor_probability = probabilities[0][0]
    fgm_probability = probabilities[0][1]

    return sensor_probability, fgm_probability

def preprocess_image(image_path, target_size=(316, 354)):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    # Expand dimensions to match the model input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    tflite_model_path = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/CKPT_354X316/GCVIT_CKPT_ORG354X316/TFLITE/GCViT_V5_ORG354X316_21102024_RESUME_Optimize.tflite'
    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)

    input_csv = '/data/DATA/FINGERPRINT_DATA/DATA_VER5.0_JUNE2024/CSV_FILE/Full_image_CSV/DATA_WITH_FP_FN/DATA_INCLUDE_GCVIT_RESUME_1808_FP_FN/2110_TEST_EX_MELO31_ALL_FP_FN_77814.csv'
    output_csv = '/data/WORKSPACE_HS/MFS100_custome_model/MFS100_CUSTOM_MODEL/MODEL_ORG_354X316/Result/Result_13_11/pred_2110_TEST_EX_MELO31_ALL_FP_FN_77814_TFLITE_optmize.csv'

    csv_data = pd.read_csv(input_csv)
    csv_data['PLiveT'] = 0.0
    csv_data['PSpoofT'] = 0.0

    total_images = len(csv_data)
    print("Total images to process: {}".format(total_images))

    # Use tqdm to wrap the DataFrame iteration for progress tracking
    for idx, row in tqdm(csv_data.iterrows(), total=total_images, desc="Processing images"):
        prob_Live, prob_Fake = Tflite_infer(interpreter, input_details, output_details, row['Image_path'])
        csv_data.at[idx, 'PSpoofT'] = prob_Fake
        csv_data.at[idx, 'PLiveT'] = prob_Live

    csv_data.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
