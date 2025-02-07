## Custom L1 Distance layer module
## WHY DO WE NEED THIS:- its needed to load the custom model


# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


## Custom L1 Distance Layer from Jupyter  (4.2)
class L1Dist(Layer):

    # init method -- inheritance
    def __init___(self, **kwargs):
        super().__init__()

    # Magic happens here -- similarity calculation
    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)  # calculating L1 distances