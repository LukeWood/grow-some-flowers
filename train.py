import sys

import tensorflow_datasets as tfds
import tensorflow as tf
import visualization as visualiation_lib
from absl import app
from absl import flags
from model import DiffusionModel
from tensorflow import keras

flags.DEFINE_string("artifacts_dir", None, "artifact save dir")
flags.DEFINE_string("model_dir", None, "directory to save model to")
flags.DEFINE_string(
    "checkpoint_path",
    "artifacts/checkpoint/diffusion_model",
    "model checkpoint directory",
)
flags.DEFINE_float("percent", 100, "percentage of dataset to use")
flags.DEFINE_integer("epochs", 100, "epochs to train for")
flags.DEFINE_boolean(
    "force_download", False, "Whether or not to force download the dataset."
)
FLAGS = flags.FLAGS


dataset_name = "oxford_flowers102"
dataset_repetitions = 5
num_epochs = 50  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

def main(args):
    # load dataset
    artifacts_dir = FLAGS.artifacts_dir or "artifacts"
    train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")

    model = DiffusionModel(image_size, widths, block_depth)
    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
    )

    # save the best model based on the validation KID metric
    checkpoint_path = FLAGS.checkpoint_path
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        # validation_data=val_dataset,
        callbacks=[
            visualiation_lib.SaveVisualOfSameNoiseEveryEpoch(
                model=model, save_path=f"{artifacts_dir}/same-noise"
            ),
            visualiation_lib.SaveRandomNoiseImages(
                model=model, save_path=f"{artifacts_dir}/random"
            ),
            checkpoint_callback,
        ],
    )

if __name__ == "__main__":
    app.run(main)
