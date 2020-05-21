from tensorflow import ConfigProto, Session
from keras import Input, backend
from keras.layers import Dense, concatenate, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


class Classifier:

    def __init__(self):

        def swish(x, beta=1):
            return x * backend.sigmoid(beta * x)

        get_custom_objects().update(dict(swish=Activation(swish)))

        config = ConfigProto(device_count=dict(GPU=1, CPU=2))
        sess = Session(config=config)

        backend.set_session(sess)
        self.run()
        backend.clear_session()

    @classmethod
    def run(cls):

        input_x = Input(shape=(32,))
        input_y = Input(shape=(32,))

        x_axis = Dense(8, activation="swish")(input_x)
        x_axis = Dense(4, activation="swish")(x_axis)
        x_axis = Model(inputs=input_x, outputs=x_axis)

        y_axis = Dense(8, activation="swish")(input_y)
        y_axis = Dense(4, activation="swish")(y_axis)
        y_axis = Model(inputs=input_y, outputs=y_axis)

        combined = concatenate([x_axis.output, y_axis.output])

        predictions = Dense(10, activation="swish")(combined)
        predictions = Dense(5, activation="softmax")(predictions)

        model = Model(inputs=[x_axis.input, y_axis.input], outputs=predictions)
