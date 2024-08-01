from agents.qmix_full.dq_network import DQN
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config
import tensorflow as tf

FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
minibatch_size = FLAGS.minibatch_size
target_update = FLAGS.target_update

if "battery" in FLAGS.scenario:
    sup_len = 3
elif "endless" in FLAGS.scenario:
    sup_len = 3
else:
    sup_len = 0

import numpy as np
import tensorflow as tf
from agents.replay_buffer import ReplayBuffer


class PartialQMIXAgent:
    def __init__(self, obs_space, act_space, sess, n_agents, name):
        self.obs_space = obs_space
        self.act_space = act_space
        self.n_agents = n_agents

        # Use a simpler DQN model
        self.dqn = DQN(sess, obs_space, 0, act_space, n_agents, name, False)
        # Smaller replay buffer capacity
        self.rb = ReplayBuffer(capacity=5000)  # Suboptimal small capacity
        self.train_cnt = 0

    def act_multi(self, obs, random):
        q_values = self.dqn.get_q_values(obs)
        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random + 1) % 2) * (q_values.argmax(axis=1)) + (random) * r_action
        return action_n

    def add_to_memory(self, exp):
        self.rb.add_to_memory(exp)

    def sync_target(self):
        self.dqn.training_target_qnet()

    def train(self):
        data = self.rb.sample_from_memory(32)  # Sample fewer experiences

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])
        coords = np.asarray([x[5] for x in data])
        next_coords = np.asarray([x[6] for x in data])

        not_done = (done + 1) % 2

        # Add noise to the rewards
        noisy_reward = reward + np.random.normal(0, 0.1, size=reward.shape)

        td, _ = self.dqn.training_qnet(coords, state, action, noisy_reward, not_done, next_coords, next_state)

        self.train_cnt += 1
        if self.train_cnt % 200 == 0:  # Less frequent target updates
            self.sync_target()

        return td

    def generate_indiv_dqn(self, s, a_id, trainable=True):
        obs = tf.reshape(s, (-1, 4, 4, 3))  # Assuming the observation shape

        conv1 = tf.compat.v1.layers.conv2d(obs, 16, (3, 3), activation=tf.nn.relu, 
                                           use_bias=True, trainable=trainable, name='conv1_'+str(a_id))

        conv1 = tf.compat.v1.layers.flatten(conv1)

        hidden = tf.compat.v1.layers.dense(conv1, 32, activation=tf.nn.relu,
                                           use_bias=True, trainable=trainable, name='dense_a1_'+str(a_id))

        q_values = tf.compat.v1.layers.dense(hidden, self.act_space, trainable=trainable, name='q_value_'+str(a_id))

        return q_values

    def generate_mixing_network(self, state, coords, q_values, trainable=True):
        if self.sup_state_dim > 0:
            sup_state = state[:,:,-1*self.sup_state_dim:]
            sup_state = tf.reshape(sup_state, (-1,self.sup_state_dim*self.n_agents))
            hyper_in = tf.concat([coords, sup_state], axis=1)
        else:
            hyper_in = coords

        w1 = tf.compat.v1.layers.dense(hyper_in, self.n_agents*8, 
            use_bias=True, trainable=trainable, name='dense_w1')

        w2 = tf.compat.v1.layers.dense(hyper_in, 8,
            use_bias=True, trainable=trainable, name='dense_w2')

        b1 = tf.compat.v1.layers.dense(hyper_in, 8, 
            use_bias=True, trainable=trainable, name='dense_b1')

        b2_h = tf.compat.v1.layers.dense(hyper_in, 8, activation=tf.nn.relu, 
            use_bias=True, trainable=trainable, name='dense_b2_h')

        b2 = tf.compat.v1.layers.dense(b2_h, 1, 
            use_bias=True, trainable=trainable, name='dense_b2')

        w1 = tf.reshape(tf.abs(w1), [-1, self.n_agents, 8])
        w2 = tf.reshape(tf.abs(w2), [-1, 8, 1])

        q_values = tf.reshape(q_values, [-1,1,self.n_agents])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,8]) + b1)
        q_hidden = tf.reshape(q_hidden, [-1,1,8])
        q_total = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

        return q_total

    def get_q_values(self, state_ph):
        state_ph = np.asarray(state_ph).reshape((-1, 20, 4, 4, 3))  # Reshape the state to the expected shape
        return self.sess.run(self.concat_dqns, feed_dict={self.state_ph: state_ph, self.n_in: len(state_ph)})

    def training_qnet(self, coords_ph, state_ph, action_ph, reward_ph, is_not_terminal_ph, next_coords_ph, next_state_ph, lr=0.001):
        return self.sess.run([self.td_errors, self.train_network], 
            feed_dict={
                self.coords_ph: coords_ph,
                self.state_ph: state_ph,
                self.next_coords_ph: next_coords_ph,
                self.next_state_ph: next_state_ph,
                self.action_ph: action_ph,
                self.reward_ph: reward_ph,
                self.is_not_terminal_ph: is_not_terminal_ph,
                self.n_in: len(coords_ph),
                self.lr: lr})

    def training_target_qnet(self):
        self.sess.run(self.update_slow_target_dqn)
