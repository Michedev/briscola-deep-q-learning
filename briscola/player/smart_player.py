from typing import List

from base_player import BasePlayer
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
from card import Card
from seed import Seed
from brain import *
from random import randint, random

values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}
GLOBAL_COUNTER = 0
from path import Path


def build_state_array(public_state, hand: List[Card], pname: str) -> np.ndarray:
    x = np.zeros((34,))
    for i, c in enumerate(hand):
        range_i = slice(i * 6, (i + 1) * 6)
        x[range_i] = to_feature(c)
    offset = 18
    for c in public_state.table:
        x[offset:offset + 6] = to_feature(c)
    offset = 24
    x[offset:offset + 2] = public_state.points
    offset = 26
    x[offset:offset + 6] = to_feature(public_state.briscola)
    offset = 32
    x[offset] = public_state.order[0] == pname
    offset = 33
    x[offset] = public_state.turn
    return x


def build_x_discarded(state: 'PublicState', hand: List[Card]):
    x = np.zeros(shape=(1, 6, 4 * len(state.discarded), 1), dtype='float32')
    if len(state.discarded) == 0:
        x = np.zeros(shape=(1, 6, 4, 1), dtype='float32')
        return x
    for j, c in enumerate(state.discarded):
        for i in range(3):
            if i < len(hand):
                x[:, :, j + i] = to_feature(hand[i]).reshape((1, 6, 1))
            else:
                x[:, :, j + i] = to_feature(None).reshape((1, 6, 1))
        x[:, :, j + 3] = to_feature(c).reshape((1, 6, 1))
    return x



FOLDER = Path(__file__).parent
BRAINFILEINDEX = FOLDER / 'brain.tf.index'
BRAINFILE = FOLDER / 'brain.tf'


class QAgent:

    def __init__(self, input_size, discount_factor=0.8, experience_size=300_000, update_q_fut=1000,
                 sample_experience=128, update_freq=60, no_update_start=500):
        '''

        :param input_size:
        :param discount_factor:
        :param experience_size:
        :param update_q_fut:
        :param sample_experience: sample size drawn from the buffer
        :param update_freq: number of steps for a model update
        :param no_update_start: number of initial steps which the model doesn't update
        '''
        ### TODO implement in decide update_freq and no_update_start
        self.no_update_start = no_update_start
        self.update_freq = update_freq
        self.sample_experience = sample_experience
        self.update_q_fut = update_q_fut
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        self.input_size = list(input_size)
        self.brain = brain_v1(self.input_size)  # throw first, second, third card
        if BRAINFILEINDEX.exists():
            self.brain.load_weights(BRAINFILE)
        self.q_future = tf.keras.models.clone_model(self.brain)
        self._q_value_hat = 0
        self.opt = tfa.optimizers.RectifiedAdam(0.00025, 0.95, 0.95, 0.01)
        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.brain.summary()
        self.writer = tf.summary.create_file_writer('briscola_logs')
        self.writer.set_as_default()
        self.epsilon = 1.0
        self._i_experience = 0
        self._curiosity_values = None
        self.experience_max_size = experience_size
        self.experience_size = 0
        self.same_values = tf.zeros((4,))
        self.same_counter = 0
        self.destination_position = None

        self.experience_buffer = [np.zeros([self.experience_max_size] + self.input_size, dtype='bool'),  # s_t
                                  np.zeros([self.experience_max_size], dtype='int8'),  # action
                                  np.zeros([self.experience_max_size], dtype='float32'),  # reward
                                  np.zeros([self.experience_max_size] + self.input_size, dtype='bool')]  # s_t1

    def brain_section(self, section):
        return self.brain.get_layer(section).trainable_variables

    def decide(self, state):
        if self._i_experience == self.experience_max_size:
            self._i_experience = 0
        if self.no_update_start < self._i_experience and self._i_experience % self.update_freq == 0:
            self.experience_update(self.experience_buffer, self.discount_factor)
        self.experience_buffer[0][self._i_experience] = state
        state = tf.Variable(state.astype('float32'), trainable=False)
        state = tf.expand_dims(state, axis=0)
        q_values = self.brain(state)
        q_values = tf.squeeze(q_values)
        if random() > self.epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 2)
        self._q_value_hat = q_values[i]
        tf.summary.scalar('q value first card', q_values[0], self.step)
        tf.summary.scalar('q value second card', q_values[1], self.step)
        tf.summary.scalar('q value third card', q_values[2], self.step)
        tf.summary.scalar('expected reward', self._q_value_hat, self.step)
        tf.summary.scalar('action took', i, self.step)
        self.experience_buffer[1][self._i_experience] = i
        self.epsilon = max(0.02, self.epsilon - 0.0004)
        if self.step_episode % 1000 == 0:
            self.epsilon = 1.0
        if self.same_counter == 100:
            print('chosen the wrong path, going back to the random...')
            self.epsilon = 1.5
            self.same_counter = 0
        elif self.epsilon == 0.02 and np.all(tf.abs(self.same_values - q_values) < 0.00001):
            self.same_counter += 1
        else:
            self.same_counter = 0
            self.same_values = q_values
        return i

    def get_reward(self, next_state, reward):
        self.experience_buffer[2][self._i_experience] = reward
        self.experience_buffer[3][self._i_experience] = next_state

        if self.step % 1000 == 0 and self.step > 0:
            del self.q_future
            gc.collect()
            self.q_future = tf.keras.models.clone_model(self.brain)
        tf.summary.scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self._i_experience += 1
        if self.experience_max_size > self.experience_size:
            self.experience_size += 1

    def experience_update(self, data, discount_factor):
        index = np.random.randint(0, self.experience_size, (self.sample_experience,))
        batch_size = 32
        nbatch = np.ceil(len(index) / batch_size)
        nbatch = int(nbatch)
        for i_batch in range(nbatch):
            is_last = i_batch == nbatch - 1
            slice_batch = slice(i_batch * batch_size, ((i_batch + 1) * batch_size if not is_last else None))
            (s_t, a_t, r_t, s_t1) = [data[i][index[slice_batch]] for i in range(len(data))]
            a_t = tf.cast(a_t, tf.int32)
            s_t = tf.cast(s_t, tf.float32)
            s_t1 = tf.cast(s_t1, tf.float32)
            with tf.GradientTape() as gt:
                exp_rew_t = self.brain(s_t)
                exp_rew_t = exp_rew_t * tf.one_hot(a_t, depth=4)
                exp_rew_t = tf.reduce_max(exp_rew_t, axis=1)
                exp_rew_t1 = self.q_future(s_t1)
                exp_rew_t1 = tf.reduce_max(exp_rew_t1, axis=1)
                loss = tf.losses.mse(r_t + discount_factor * exp_rew_t1, exp_rew_t)
                del s_t, a_t, r_t, s_t1
                loss = tf.reduce_mean(loss, axis=0)
                loss = tf.reduce_sum(loss)
            if self.step % 10 == 0:
                tf.summary.scalar('loss', loss, self.step)
            gradient = gt.gradient(loss, self.brain.trainable_variables)
            if self.step % 100 == 0:
                for l, g in zip(self.brain.trainable_variables, gradient):
                    tf.summary.histogram('gradient ' + l.name, g, self.step)
                    tf.summary.histogram(l.name, l, self.step)
            self.opt.apply_gradients(zip(gradient, self.brain.trainable_variables))
            del gradient, exp_rew_t, exp_rew_t1

    def reset(self):
        self.epsilon = max(1.0 - 0.001 * self.episode, 0.1)
        self.step_episode = 0

    def on_win(self):
        tf.summary.scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1


class SmartPlayer(BasePlayer, QAgent):

    def __init__(self):
        QAgent.__init__(self, [34])
        BasePlayer.__init__(self)

        self.name = "intelligent_player"
        self._reward = 0

    def choose_card(self) -> int:
        state = build_state_array(self.get_public_state(), self.hand, self.name)
        if self.step_episode > 0:
            self.get_reward(state, self._reward)
        i = self.decide(state)
        if i >= len(self.hand):
          i = randint(0, len(self.hand)-1)
          self._reward = -30
        return i

    def notify_turn_winner(self, points: int):
        self._reward += points
