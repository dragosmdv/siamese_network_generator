from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

from siamese_network_generator.model_definition.DistanceLayer import DistanceLayer

model = None


def get_siamese_network_and_embedding():
    target_shape = (200, 200)
    # load base resNet model
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")
    # freeze the weights of all the layers of the model up until the layer `conv5_block1_out`.
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    return Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    ), embedding


def load_untrained_model():
    global model
    model = None
    _, embedding = get_siamese_network_and_embedding()
    return embedding


def reset_model():
    global model
    model = None

def load_cached_model():
    global model
    if model is None:
        _, embedding = get_siamese_network_and_embedding()
        embedding.load_weights("production_model/myModel_v9")
        model = embedding
    return model
