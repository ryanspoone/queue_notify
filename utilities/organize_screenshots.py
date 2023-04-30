import os
import glob
import time
from PIL import Image
import tensorflow as tf


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def classify_image(model, image_path):
    image = Image.open(image_path)
    resized_image = image.resize((224, 224), Image.ANTIALIAS)
    input_data = tf.keras.preprocessing.image.img_to_array(resized_image)
    input_data = input_data / 255.0
    input_data = input_data.reshape((1, *input_data.shape))
    prediction = model.predict(input_data)
    return prediction


def organize_screenshots(screenshot_folder, model):
    screenshot_files = glob.glob(f"{screenshot_folder}/*.jpg")

    for screenshot_file in screenshot_files:
        prediction = classify_image(model, screenshot_file)

        if (
            prediction[0][0] > 0.5
        ):  # Adjust this threshold according to your model's output
            label = "queue_pop"
        else:
            label = "not_queue_pop"

        label_folder = os.path.join(screenshot_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        file_name = os.path.join(
            label_folder, os.path.basename(screenshot_file)
        )
        os.rename(screenshot_file, file_name)
        print(f"Moved {screenshot_file} to {file_name}")


def main():
    screenshot_folder = "screenshots"
    model_path = "queue_pop_detector.h5"

    if not os.path.exists(screenshot_folder):
        print(
            f"{screenshot_folder} does not exist. Please provide a valid folder path."
        )
        return

    model = load_model(model_path)
    organize_screenshots(screenshot_folder, model)


if __name__ == "__main__":
    main()
