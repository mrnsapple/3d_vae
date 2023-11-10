
@hydra.main(config_path="../config", config_name="main", version_base=None)
def eval_model(config: DictConfig):
    train_dataset, test_dataset = process.load_mnist_dataset_batch(
        config.data.train_size, config.data.batch_size, config.data.test_size
    )
    for test_batch in test_dataset.take(1):
        # test_sample = test_batch[0:num_examples_to_generate, :, :, :]
        test_sample = test_batch
    model = tf.keras.models.load_model(config.model.checkpoint)
    # save_image(test_sample, "{}eval_image_original.png".format(config.data.final_images))
    # import pdb;pdb.set_trace()
    generate_and_save_images(
        model, 0, test_sample, "{}eval_image.png".format(config.data.final_images)
    )

    mean, logvar = model.encode(test_sample)
    model.reparameterize(mean, logvar)
    plt.figure(figsize=(10, 10))
    plt.scatter(mean[:, 0], mean[:, 1], cmap="brg")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.colorbar()
    plt.show()

