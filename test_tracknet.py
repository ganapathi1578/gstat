"""Inspect layer config of the working TrackNet model."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

base = tf.keras.models.load_model(
    "models/best_model_base_new_data_e28.keras", compile=False)

for i, layer in enumerate(base.layers):
    cfg = layer.get_config()
    data_fmt = cfg.get("data_format", "N/A")
    print(f"[{i:3d}] {layer.__class__.__name__:30s} {str(layer.output_shape):35s} data_format={data_fmt}")
