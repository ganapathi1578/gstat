import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

for path in [
    'models/best_model_V2_NF_RIO_1m_e8.keras',
    'models/best_model_V1_NF_RIO_10u_e17.keras',
    'models/best_model_base_new_data_e28.keras',
]:
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f'=== {path} ===')
        print(f'  Input shape : {m.input_shape}')
        print(f'  Output shape: {m.output_shape}')
        print(f'  Params      : {m.count_params():,}')
        print()
    except Exception as e:
        print(f'FAILED {path}: {e}')
