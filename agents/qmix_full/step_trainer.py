from agents.qmix_full.agent import PartialQMIXAgent
from agents.base_trainer import Trainer as BaseTrainer
from agents.base_trainer import stringify
from agents.simple_agent import RunningAgent as NonLearningAgent
import numpy as np
import tensorflow as tf
import config
from datetime import datetime

np.set_printoptions(precision=2)

import tensorflow as tf
import numpy as np
from datetime import datetime

FLAGS = config.flags.FLAGS
minibatch_size = FLAGS.minibatch_size
n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
test_interval = FLAGS.test_interval
train_interval = FLAGS.train_interval
map_size = FLAGS.map_size


class Trainer(BaseTrainer):
    def __init__(self, environment, logger):
        self.env = environment
        self.logger = logger
        self.n_agents = n_predator + n_prey

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        self._agent_profile = self.env.get_agent_profile()
        agent_precedence = self.env.agent_precedence

        # Initialize predator agent using PQMIX
        self.predator_singleton = PartialQMIXAgent(obs_space=(4,4,3),  # Adjusted for 4x4 grid and 3 channels (RGB)
                                             act_space=self._agent_profile["predator"]["act_spc"],
                                             sess=self.sess, n_agents=n_predator, 
                                             name="predator")

        # Initialize other agents (prey) using NonLearningAgent
        self.agents = []
        for i, atype in enumerate(agent_precedence):
            if atype == "predator":
                agent = self.predator_singleton
            else:
                agent = NonLearningAgent(self._agent_profile[atype]["act_spc"])

            self.agents.append(agent)

        # Initialize tf variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

        if FLAGS.load_nn:
            if FLAGS.nn_file == "":
                logger.error("No file for loading Neural Network parameter")
                exit()
            self.saver.restore(self.sess, FLAGS.nn_file)
        else:
            self.predator_singleton.sync_target()

    def learn(self, max_global_steps, max_step_per_ep):
        epsilon = 1.0
        epsilon_dec = 1.0 / FLAGS.explore
        epsilon_min = 0.1

        start_time = datetime.now()

        if max_global_steps % test_interval != 0:
            max_global_steps += test_interval - (max_global_steps % test_interval)

            steps_before_train = min(FLAGS.minibatch_size * 4, FLAGS.rb_capacity)

        tds = []
        ep = 0
        global_step = 0
        while (global_step < max_global_steps):
            ep += 1
            obs_n = self.env.reset()

            # Get initial 4x4 grid partial observations and coordinates
            coords = self.get_agent_coordinates(obs_n)
            partial_observations = self.get_partial_observations(obs_n)

            for step in range(max_step_per_ep):
                global_step += 1                

                # Get the action using epsilon-greedy policy
                act_n = self.get_actions(partial_observations, epsilon)

                # Do the action and update observation
                obs_n_next, reward_n, done_n, _ = self.env.step(act_n)
                done = done_n[:n_predator].all()

                # Get next coordinates and partial observations
                next_coords = self.get_agent_coordinates(obs_n_next)
                next_partial_observations = self.get_partial_observations(obs_n_next)

                # Prepare experience tuple for predator agent
                exp = [
                    partial_observations[:n_predator],    # Predator's partial observation
                    act_n[:n_predator],                   # Predator's action
                    reward_n[:n_predator].mean(),         # Average reward for predators
                    next_partial_observations[:n_predator],  # Next partial observations for predators
                    done,                                 # Done flag for episode
                    coords,                               # Current coordinates
                    next_coords                           # Next coordinates
                ]

                self.predator_singleton.add_to_memory(exp)

                # Update coordinates and observations
                coords = next_coords
                partial_observations = next_partial_observations

                if done:
                    break

                # Perform training step
                if global_step > steps_before_train and global_step % train_interval == 0:
                    td_error = self.predator_singleton.train()
                    tds.append(td_error)

                # Decay epsilon
                if epsilon > epsilon_min:
                    epsilon -= epsilon_dec

            if ep % test_interval == 0:
                mean_steps, mean_b_reward, mean_captured, success_rate, remaining_battery = self.test(FLAGS.test_episodes, max_step_per_ep)
                logger.info(f"Episode: {ep}, Global Step: {global_step}, Mean Steps: {mean_steps}, Mean Reward: {mean_b_reward.mean()}, Success Rate: {success_rate.mean()}, Remaining Battery: {remaining_battery.mean()}")

    def get_partial_observations(self, obs_n):
        """
        Get partial observations for all agents.
        
        Args:
        - obs_n: List of observations for each agent.

        Returns:
        - List of partial observations for each agent.
        """
        partial_observations = [self.get_partial_observation(agent_id, obs) for agent_id, obs in enumerate(obs_n)]
        return partial_observations

    def get_partial_observation(self, agent_id, obs):
        agent = self.env.agents[agent_id]
        x, y = agent._x, agent._y  # Get the current position of the agent

        # Assuming `obs` is a 2D grid centered around the agent, reshape it accordingly
        partial_observation = obs.reshape(4, 4, 3)  # Adjust as necessary
        return partial_observation

    def get_agent_coordinates(self, obs_n):
        """
        Get the current coordinates of all agents in the environment.
        
        Returns:
        - numpy array: Flattened coordinates of all agents normalized by map size.
        """
        # Retrieve agent positions directly from the environment's grid
        agent_positions = []
        for agent in self.env.agents:
            agent_positions.append((agent._x, agent._y))
        
        coords = np.asarray(agent_positions, dtype=np.float32).flatten() / (map_size - 1)
        return coords
