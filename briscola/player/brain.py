import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
import numpy as np

values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}

from card import Card
from seed import Seed


class Conv2DSeq(Layer):

    def __init__(self, nfilter):
        super(Conv2DSeq, self).__init__(name="Conv2DSeq".lower())
        self.conv_layer = Conv2D(nfilter, kernel_size=(6, 4), padding='same')
        self.glb_avg = GlobalAveragePooling2D()

    def call(self, x, **kwargs):
        assert len(x.shape) == 4
        output = self.conv_layer(x)
        output = self.glb_avg(output)
        # print(output.shape)
        return output


def to_feature(c: Card) -> np.ndarray:
    if c is None:
        return np.zeros((6,))
    ohe_seed = Seed.ohe_repr(c.seed)
    points_card = values_points[c.value]
    value_card = c.value
    features = np.array([value_card, points_card])
    return np.concatenate([features, ohe_seed])


def root_brain(state_size):
    inputs = Input((state_size,))
    outputs = inputs
    for i in range(3):
        outputs = Dense(state_size)(outputs)
        outputs = ReLU()(outputs)
    return tf.keras.Model(inputs, outputs, name='root')


# def brain_section_discarded_cards():
#     inputs = Input((6, None, 1))
#     layer1 = Conv2DSeq(1000)
#     output = layer1(inputs)
#     output = BatchNormalization()(output)
#     output = ReLU()(output)
#     output = tf.reshape(output, (1, 1, 1000))
#     output = layer2(output)
#     return Model(inputs, output, name='brain_discarded')


def brain_v1(state_size):
    inputs = Input((state_size,))
    root = root_brain(state_size)
    middle = root(inputs)
    outputs = middle
    for i in range(3):
        outputs = Dense(state_size)(outputs)
        outputs = ReLU()(outputs)
    outputs = Dense(3, name='q_values')(outputs)
    return Model(inputs, outputs, name='Deep-q-network')

#
# def brain_v2(input_size):
#     brain = brain_v1(input_size)
#     brain_discarded = brain_section_discarded_cards()
#     inputs_1 = Input(input_size)
#     inputs_2 = Input((6, None, 1))
#     stop = brain.get_layer(name='recurrent')
#     output = inputs_1
#     for l in brain.layers:
#         if l == stop:
#             break
#         output = l(output)
#     output = tf.reshape(output, (1, 1, input_size))
#     output = stop(output)
#     output = tf.reshape(output, (1, input_size))
#     output = Concatenate()([output, brain_discarded(inputs_2)])
#     exp_reward = Dense(1)(output)
#     output = Dense(3, name='logit_action')(output)
#     output = Activation('softmax')(output)
#     return Model([inputs_1, inputs_2], [output, exp_reward], name='brain_v2')


__all__ = ['brain_v1', 'brain_section_discarded_cards', 'root_brain', 'to_feature']
