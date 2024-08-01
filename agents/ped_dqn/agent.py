from agents.ped_dqn.ddq_network import DQN 
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config
from envs.scenarios.battery_endless import Scenario


FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
mrb_capacity = FLAGS.mrb_capacity
minibatch_size = FLAGS.minibatch_size
target_update = FLAGS.target_update

discount_factor = FLAGS.gamma
mlr = FLAGS.mlr

if "battery" in FLAGS.scenario:
    sup_len = 3
elif "endless" in FLAGS.scenario:
    sup_len = 3
else:
    sup_len = 0


class Agent(object):
    def __init__(self, obs_space, act_space, sess, n_agents, name):
        self.act_space = act_space
        self.n_agents = n_agents
        self.scenario = Scenario()
        self.trust_matrix = self.scenario.get_trust()
        self.reputation_scores_list = []
        self.ped_dqn = DQN(sess, obs_space, sup_len, act_space, n_agents, name)

        self.action_rb = ReplayBuffer(capacity=rb_capacity)
        self.mission_rb = ReplayBuffer(capacity=mrb_capacity)
       
        self.train_cnt = 0
        self.mission_train_cnt = 0
        self.sns_q = None
        self.lamda = 0.6

    def reset(self):
        self.sns_q = None

    def act_multi(self, obs, random):        
        if self.sns_q is None:
            q_values = self.ped_dqn.get_aq_values([obs])[0]
        else:
            q_values = self.sns_q

        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random+1)%2)*(q_values.argmax(axis=1)) + (random)*r_action
        best_action= q_values.argmax(axis=1)
        
        return action_n, best_action
    
    
    def incentivize_multi(self, info):
        state, action, reward, next_state, done = info
        done = done.all()
            
        [ls_q, lns_q] = self.ped_dqn.get_aq_values([state, next_state])
        s_q = ls_q[range(self.n_agents), action]
        ns_q = discount_factor*lns_q.max(axis=1)*(not done) + reward

        td = ns_q - s_q    

        if done:
            self.sns_q = None

        return td
    
    def add_to_memory(self, exp):
        self.action_rb.add_to_memory(exp)
        self.mission_rb.add_to_memory(exp[:5])

    def sync_target(self):
        self.ped_dqn.training_target_qnet()

    def train_mission_dqn(self):
        # train mission DQN with recent and old data
        data = self.mission_rb.sample_from_memory(minibatch_size)
        
        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])

        not_done = (done+1)%2

        mtd,_ = self.ped_dqn.training_m_qnet(state, action, 
            base_reward, not_done, next_state, mlr)

        return mtd

    def train(self, use_rx):
        data = self.action_rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])

        not_done = (done+1)%2

        if use_rx:
            rx_inc = np.asarray([x[5] for x in data])
            reward = base_reward + rx_inc
            #print("reward", reward, "base_reward", base_reward, "rx_inc", rx_inc)
        else:
            reward = base_reward

        td_error,_ = self.ped_dqn.training_a_qnet(state, action, reward, not_done, next_state)

        self.train_cnt += 1
        
        if self.train_cnt % (target_update) == 0:
            self.ped_dqn.training_target_qnet()
            self.ped_dqn.training_peer_qnet()

        return td_error
    
    
    def get_reputation(self, peer_evaluations):
        reputation_scores = []
        trust_values = self.scenario.get_trust()

        for i in range(self.n_agents):
            average_trust = np.mean(trust_values[:, i])  # Average trust given to agent i
            if peer_evaluations[i] != 0:
                reputation = (1/peer_evaluations[i]) + average_trust  
            else:
                reputation = average_trust
            # Combine peer evaluation with average trust
            reputation_scores.append(reputation)

        # Scale reputation values between 0 and 1
        min_reputation = min(reputation_scores)
        max_reputation = max(reputation_scores)
        if max_reputation > min_reputation:
            reputation_scores = [(x - min_reputation) / (max_reputation - min_reputation) for x in reputation_scores]
        else:
            reputation_scores = [0 for _ in reputation_scores]
        
        self.reputation_scores_list.append(reputation_scores)
        
        if len(self.reputation_scores_list) < 3:
            return np.array(reputation_scores)
        else: # Smooth reputation scores
            reputation_scores = (1 - self.lamda) * np.array(reputation_scores) + self.lamda * np.array(self.reputation_scores_list[-2])
            return np.array(reputation_scores)

    