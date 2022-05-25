from siamese_network_generator.simese_network_actions import generate_model, test_prediction

if __name__ == "__main__":
    test_file = "datasets/test_images"
    train_file = "datasets/train_images"
    save_path = "generated_models/myModel_v16"
    epochs = 1
    generate_model(test_file, train_file, save_path, epochs)
    '''distance should be > 0  and the resulted tensor should be ('Opera_Națională_Română_din_Cluj-Napoca', 
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.150906], dtype=float32)>) 
    '''
    print(test_prediction("test_image.jpg", save_path, test_file))

    '''given a sample from the train dataset, distance should be =0  and the resulted tensor should be (
    'Biserica_evanghelică_din_Cluj-Napoca', <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>) 
    '''
    print(test_prediction(
        "datasets/test_images/Biserica_evanghelică_din_Cluj-Napoca/Biserica evanghelică din Cluj-Napoca_18.jpeg",
        save_path, test_file))
