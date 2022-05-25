from siamese_network_generator.model_generator.model_generator import prepare_dataset, train
from keras.applications import resnet

from siamese_network_generator.model_generator.image_processors import prepare_image, get_file_map
from siamese_network_generator.model_definition.model_importer import load_untrained_model
from siamese_network_generator.model_generator.triplet_selector import get_most_similar


def generate_model(test_file, train_file, save_path, epochs):
    train_dataset, val_dataset = prepare_dataset(test_file, train_file)
    train(train_dataset, val_dataset, save_path, epochs)


def test_prediction(image_path, model_path, train_dataset_path):
    embedding = load_untrained_model()
    embedding.load_weights(model_path)
    img = prepare_image(image_path)
    em = embedding(resnet.preprocess_input(img))
    image_name, distance = get_most_similar(em,embedding, train_dataset_path)
    return image_name, distance

