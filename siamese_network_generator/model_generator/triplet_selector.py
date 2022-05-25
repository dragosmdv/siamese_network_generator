import copy
from itertools import combinations
import random
from keras.applications import resnet

from siamese_network_generator.model_generator.image_processors import get_file_map, prepare_image
import tensorflow as tf


def get_random_triplets(type):

    file_map = get_file_map(type)
    triplets = []
    for key in file_map.keys():
        anchor_positives = list(combinations(file_map[key], 2))
        for a_p in anchor_positives:
            elem = a_p[0]
            random_positive = a_p[1]
            new_key_list = [el for el in file_map.keys() if el != key]
            random_key = random.choice(new_key_list)
            random_negative = random.choice(file_map[random_key])
            triplets.append((elem, random_positive, random_negative))
    anchor_images = [triplet[0] for triplet in triplets]
    positive_images = [triplet[1] for triplet in triplets]
    negative_images = [triplet[2] for triplet in triplets]

    return anchor_images, positive_images, negative_images


_encoding_map = {}


def _get_encoding(image, embedding):
    if image not in _encoding_map:
        img = prepare_image(image)
        rez = embedding(resnet.preprocess_input(img))
        _encoding_map[image] = rez
    return _encoding_map[image]


def get_close_distance_triplets(type, embedding):
    # chose triplets so that d(A,P) ~ d(A, N)
    # return get_random_triplets(type)
    file_map = get_file_map(type)
    triplets = []
    for key in file_map.keys():
        print("Fetching triplet for {} ----".format(key))
        anchor_positives = list(combinations(file_map[key], 2))
        for a_p in anchor_positives:
            elem = a_p[0]
            random_positive = a_p[1]
            # calculate d(A, P)
            dp = tf.reduce_sum(tf.square(_get_encoding(elem, embedding) - _get_encoding(random_positive, embedding)), -1)
            new_key_list = [el for el in file_map.keys() if el != key]
            best_dn = 999999
            chosen_negative = None
            # find the closest dn
            for new_key in new_key_list:
                for new_el in file_map[new_key]:
                    dn = tf.reduce_sum(
                        tf.square(_get_encoding(elem, embedding) - _get_encoding(new_el, embedding)), -1)
                    if abs(dp - dn) < best_dn:
                        best_dn = abs(dp - dn)
                        chosen_negative = copy.deepcopy(new_el)

            triplets.append((elem, random_positive, chosen_negative))
    anchor_images = [triplet[0] for triplet in triplets]
    positive_images = [triplet[1] for triplet in triplets]
    negative_images = [triplet[2] for triplet in triplets]

    return anchor_images, positive_images, negative_images


def get_most_similar(em, embedding, train_dataset_path):
    file_map = get_file_map(train_dataset_path)
    maxi = 999999
    image_map = {}
    image_name = ""
    encoding_map = {}
    for key in file_map.keys():
        for image in file_map[key]:
            if image not in encoding_map:
                img = prepare_image(image)
                rez = embedding(resnet.preprocess_input(img))
                encoding_map[image] = rez
            css = tf.reduce_sum(tf.square(em - encoding_map[image]), -1)
            image_map[image] = css
            if css < maxi:
                maxi = css
                image_name = key
    return image_name, maxi