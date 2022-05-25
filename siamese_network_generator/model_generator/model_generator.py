import tensorflow as tf
from tensorflow.keras import optimizers

from siamese_network_generator.model_generator.image_processors import preprocess_image
from siamese_network_generator.model_definition.model_importer import load_untrained_model, get_siamese_network_and_embedding
from siamese_network_generator.model_generator.SiameseModel import SiameseModel
from siamese_network_generator.model_generator.triplet_selector import get_close_distance_triplets


def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


def prepare_triplets(type):
    """
    :param type: the type of the dataset
    :return:
    """
    embedding = load_untrained_model()
    anchor_images, positive_images, negative_images = get_close_distance_triplets(type, embedding)
    print("{} {} images".format(len(anchor_images), type))
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)
    dataset = dataset.batch(32, drop_remainder=False)
    return dataset


def prepare_dataset(test_file, train_file):
    train_dataset = prepare_triplets(train_file)
    val_dataset = prepare_triplets(test_file)
    # train_dataset = prepare_triplets("datasets/train_images_augumented")
    # val_dataset = prepare_triplets("datasets/test_images_augumented")
    # train_dataset = prepare_triplets("datasets/train_images_cropped")
    # val_dataset = prepare_triplets("datasets/test_images_cropped")
    return train_dataset, val_dataset


def train(train_dataset, val_dataset, save_path, epochs):
    siamese_network, embedding = get_siamese_network_and_embedding()
    # embedding.summary()
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.000001))
    history = siamese_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    embedding.save_weights(save_path)
    return history
