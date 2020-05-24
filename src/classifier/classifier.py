from tensorflow import ConfigProto, Session
from keras import Input, backend
from keras.layers import Dense, concatenate, LeakyReLU, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


class Classifier:

    def __init__(self):

        def leaky_relu(x, alpha=0.01):
            if x < 0:
                return x * alpha
            return x

        get_custom_objects().update(dict(leaky_relu=Activation(leaky_relu)))

        config = ConfigProto(device_count=dict(GPU=1, CPU=2))
        sess = Session(config=config)

        backend.set_session(sess)
        self.run()
        backend.clear_session()

    @classmethod
    def run(cls):

        input_x = Input(shape=(100,))
        input_y = Input(shape=(100,))

        x_axis = Dense(50, activation='leaky_relu')
        x_axis = Model(inputs=input_x, outputs=x_axis)

        y_axis = Dense(50, activation='leaky_relu')
        y_axis = Model(inputs=input_y, outputs=y_axis)

        combined = concatenate([x_axis.output, y_axis.output])

        predictions = Dense(4, activation="softmax")(combined)

        model = Model(inputs=[x_axis.input, y_axis.input], outputs=predictions)
