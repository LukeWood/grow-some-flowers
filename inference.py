import sys

import tensorflow as tf
import tensorflow_addons as tfa
import visualization as visualiation_lib
from absl import app
from absl import flags
from model import DiffusionModel
from tensorflow import keras

import goa_loader

image_size = 64

flags.DEFINE_string(
    "model_path",
    None,
    "model checkpoint directory",
)
flags.DEFINE_string(
    "artifacts_dir", None, "artifact location"
)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

widths = [32, 64, 96, 128]
block_depth = 2
ema = 0.999

learning_rate = 1e-3
weight_decay = 1e-4

def preprocess_image(image):
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)

model = DiffusionModel(image_size, widths, block_depth)
checkpoint_path = FLAGS.model_path
train_ds = goa_loader.load()
model.normalizer.adapt(train_ds)
model.generate(18)
model.compile(
	optimizer=tfa.optimizers.AdamW(
		learning_rate=learning_rate, weight_decay=weight_decay
	),
	loss=keras.losses.mean_absolute_error,
)
model.evaluate(train_ds.take(5))
model.load_weights(checkpoint_path)
