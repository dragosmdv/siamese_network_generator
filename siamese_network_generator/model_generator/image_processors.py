import os
import tensorflow as tf


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    target_shape = (200, 200)
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def prepare_image(filepath):
    image_p = preprocess_image(filepath)
    image_p = image_p[None]
    return image_p


def get_file_map(data_set_folder):
    arr = os.listdir(data_set_folder)
    file_map = {}
    for name in arr:
        file_map[name] = os.listdir(data_set_folder + '\\' + name)

        for i in range(len(file_map[name])):
            file_map[name][i] = data_set_folder + '\\' + name + '\\' + file_map[name][i]
    return file_map

