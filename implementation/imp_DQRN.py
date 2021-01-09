import os
import numpy as np
import tensorflow as tf
import time
import collections
import itertools as it
import random
import pdb

class DeepQNetwork(object):

    def __init__(self, lr, n_actions, name, fc1_dims=512, LSTM_DIM=256,
                 input_dims=(210, 160, 4), chkpt_dir="tmp/dqn"):
        self.lr = lr
        self.name = name
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "2,3"
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.LSTM_DIM = LSTM_DIM
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session(config=config)
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.checkpoint_file = os.path.join(chkpt_dir, "deepqnet.ckpt")
        self.saver = tf.train.Saver(max_to_keep=100)
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)
        self.write_op = tf.summary.merge_all()
        #dirname = os.path.dirname(__file__)
        #self.log = os.path.join(*[dirname, 'tmp', 'log_dir', self.name])
        #if os.path.isdir(self.log):
         #   print("output_results: ", str(self.log))
        #else:
            #os.mkdir(self.log)
        self.make_log_dir()
        self.writer = tf.summary.FileWriter(self.log, self.sess.graph)

    def make_log_dir(self):
        path = '/home/azlaans/aienvs'
        self.log = os.path.join(*[path, 'test', 'tmp', 'log_dir', 'single', self.name])
        if os.path.isdir(self.log):
            print('Log_Dir exists for tensorboard summary')
        else:
            os.mkdir(self.log)
            print('Lod_Dir created for tensorboard summary', self.log)
            

    def build_network(self):

        with tf.variable_scope(self.name):
            self.states = tf.placeholder(tf.float32, shape=[None, *self.input_dims],
                                        name='states') 
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                          name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None],
                                           name='q_value')
            self.seq_len = tf.placeholder(tf.int32, name='sequence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

            # Create placeholders to input the hidden state values
            c_in = tf.placeholder(tf.float32, [None, self.LSTM_DIM], name='cell_state')
            h_in = tf.placeholder(tf.float32, [None, self.LSTM_DIM], name='h_state')
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)

            #self._reward = tf.placeholder(tf.float32, shape=[], name='Reward/Time_step')
            #self.reward_sum = tf.summary.scalar('Reward/Time_step', self._reward)

            #self._waitingtime = tf.placeholder(tf.float32, shape=[], name='TotalWaitingTime/Time_step')
            #self.waitingtime_sum = tf.summary.scalar('TotalWaitingTime/Time_step', self._waitingtime)

            #self._delay = tf.placeholder(tf.float32, shape=[], name='TotalDelay/Time_step')
            #self.delay_sum = tf.summary.scalar('TotalDelay/Time_step', self._delay)

            conv1 = tf.layers.conv2d(inputs=self.states, filters=16,
                                     kernel_size=(8, 8), strides=(4,4), name='conv1', padding='VALID',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2),use_bias=True, bias_initializer=tf.constant_initializer(0.1))
            # TensorShape([Dimension(None), Dimension(44), Dimension(39), Dimension(32)])

            conv1_activated = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=32,
                                     kernel_size=(4, 4), strides=(2,2), name='conv2', padding='VALID',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2), use_bias=True, bias_initializer=tf.constant_initializer(0.1))
            # TensorShape([Dimension(None), Dimension(21), Dimension(18), Dimension(64)])

            conv2_activated = tf.nn.relu(conv2)

            #conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=64,
            #                         kernel_size=(3, 3), strides=1, name='conv3',
                                     #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            #conv3_activated = tf.nn.relu(conv3)

            n_input = conv2_activated.get_shape().as_list()[1]*conv2_activated.get_shape().as_list()[2]*conv2_activated.get_shape().as_list()[3]
            
            conv2_activated = tf.reshape(conv2_activated, [-1, n_input])
            
            conv2_activated = tf.reshape(conv2_activated, [self.batch_size, self.seq_len, n_input])
   
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.LSTM_DIM, initializer=tf.contrib.layers.xavier_initializer())
            outputs, self.cell_state = tf.nn.dynamic_rnn(lstm_cell, conv2_activated, initial_state=self.state_in, dtype=tf.float32, sequence_length=self.seq_len)

            var1 = tf.get_variable('weights', (self.LSTM_DIM, self.n_actions), initializer=tf.contrib.layers.xavier_initializer(), trainable=True, 
                                                    regularizer=tf.contrib.layers.l2_regularizer(0.01))
            var2 = tf.get_variable('biases', (self.n_actions,), trainable=True, initializer=tf.constant_initializer(0.1))

            h = outputs[:,-1,:] 

            self.Q_values = tf.matmul(h, var1) + var2
            tf.summary.histogram('Q_value', self.Q_values)

            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))
    
            self.loss_sum = tf.summary.scalar("Loss", self.loss)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            if self.name == 'q_eval':
                for var in tf.trainable_variables():
                    c = var.name[:-2]
                    with tf.name_scope(c):
                        self.variable_summaries(var)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.histogram('histogram', var)

    def load_checkpoint(self, filename):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, filename)

    def save_checkpoint(self, epi_num):
        print('... Saving Checkpoint ...')
        #self.epi_num = epi_num
        #dir_name = os.path.join(self.chkpt_dir, str(self.epi_num))
        #if os.path.isdir(dir_name):
         #   print("directory exists ", str(dirname))
        #else:
         #   os.mkdir(dir_name)
        #filename = "deepQnet_" + str(epi_num)
        self.saver.save(self.sess, self.checkpoint_file, global_step=epi_num)

class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, batch_size, num_agents, act_per_agent,
                 replace_target=3000, input_dims=(210, 160, 4), q_next_dir="tmp/q_next", q_eval_dir="tmp/q_eval", test= False):
        self.num_agents = num_agents
        self.act_per_agent = act_per_agent
        self.input_dims = input_dims
        self.n_actions = self.act_per_agent**(self.num_agents)
        self.action_space = [i for i in range(self.act_per_agent)]
        self.gamma = gamma
        self.seq_length = 10
        self.LSTM_DIM = 256
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target        
        
        self.q_eval = DeepQNetwork(alpha, self.n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=q_eval_dir)

        #self.q_next = DeepQNetwork(alpha, self.n_actions, input_dims=input_dims,
                                   #name='q_next', chkpt_dir=q_next_dir)

        if test==False:
            self.create_memory()
        else:
            self.test_initialiser()
    
        self.all_list = []
        for j in it.product(tuple(self.action_space), repeat = self.num_agents):
            self.all_list.append(j) 

    def create_memory(self):
        self.state_memory = np.zeros((self.mem_size, *self.input_dims)) 
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        c_init = np.zeros((1, self.LSTM_DIM), np.float32)
        h_init = np.zeros((1, self.LSTM_DIM), np.float32)
        self.state_out = (c_init, h_init)

    def action_hot_encoder(self, actions, all_list):
        action = np.zeros((self.n_actions))
        value_list = tuple(actions.values())
        for key, val in enumerate(all_list):
            if val == value_list:
                action[key] = 1.
                break
        return action

    def action_decoder(self, encoded_action, all_list):
        index = (list(np.where(encoded_action==1.))[0])[0]
        decoded_action = collections.OrderedDict()
        for i in range(len(encoded_action)):
            try:
                decoded_action[str(i)] = all_list[index][i]
            except:
                break
        return decoded_action

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.reward = reward
        self.action_memory[index] = self.action_hot_encoder(action, self.all_list)
        self.reward_memory[index] = reward['result']
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal
        if self.mem_cntr >= self.mem_size:
            self.epsilon = 0.01
   
    def upgrade(self):
        self.mem_cntr +=1
            
    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            value_list = []
            for i in range(self.num_agents):
                value_list.append(np.random.choice(self.action_space))
            value_list = tuple(value_list)
            action = np.zeros((self.n_actions))
            for key, val in enumerate(self.all_list):
                if val == value_list:
                    action[key] = 1.
                    break
            action = self.action_decoder(action, self.all_list)
        else:
            actions, lstm_state = self.q_eval.sess.run([self.q_eval.Q_values, self.q_eval.cell_state],
                                       feed_dict={self.q_eval.states: state,
                                                  self.q_eval.state_in: self.state_out,
                                                  self.q_eval.seq_len: 1,
                                                  self.q_eval.batch_size: 1})
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            action = np.argmax(actions)
            action_ht = np.zeros((self.n_actions))
            action_ht[action] = 1. 
            action = self.action_decoder(action_ht, self.all_list)
        return action

    def RandomSequenceSampling(self):
        batch_length = self.batch_size*self.seq_length
        state_batch = np.zeros((batch_length, *self.input_dims))
        next_state_batch = np.zeros((batch_length, *self.input_dims))
        reward_batch = []
        action_batch = []
        terminal_batch = []
        indices = np.arange(self.seq_length-1, self.mem_size)
        for b in np.arange(0, batch_length, self.seq_length):
            i = random.choice(indices)
            while (sum(self.terminal_memory[i+1-self.seq_length:i+1]) > 0 and self.terminal_memory[i] != 1):
                i = random.choice(indices)
            state_batch[b:b+self.seq_length] = self.get_sequence(i, self.state_memory)
            action_batch.append(self.action_memory[i])
            reward_batch.append(self.reward_memory[i])
            next_state_batch[b:b+self.seq_length] = self.get_sequence(i, self.new_state_memory)
            terminal_batch.append(self.terminal_memory[i])
        return state_batch, np.asarray(action_batch), np.asarray(reward_batch), next_state_batch, np.asarray(terminal_batch)

    def get_sequence(self, index, collection):
        stop = index + 1 
        start = stop - self.seq_length     
        if start < 0 and stop >= 0:
            try:
                seq = np.vstack((collection[start:], collection[:stop]))
            except ValueError:
                seq = np.append(collection[start:], collection[:stop])
        else:
            seq = collection[start:stop]

        if len(seq.shape) != len(collection.shape):
            seq = np.reshape(seq, (-1,))
        return seq

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.RandomSequenceSampling() 

        state = (np.zeros((self.batch_size, self.LSTM_DIM)),np.zeros((self.batch_size, self.LSTM_DIM)))

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                      feed_dict={self.q_eval.states: state_batch,
                                                 self.q_eval.state_in: state,
                                                 self.q_eval.seq_len: self.seq_length,
                                                 self.q_eval.batch_size: self.batch_size})

        q_eval_next = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.states: next_state_batch,
                                                      self.q_eval.state_in: state,
                                                      self.q_eval.seq_len: self.seq_length,
                                                      self.q_eval.batch_size: self.batch_size})

        index_best_action = np.argmax(q_eval_next, axis=1)

        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                      feed_dict={self.q_next.states: next_state_batch,
                                                 self.q_next.state_in: state,
                                                 self.q_next.seq_len: self.seq_length,
                                                 self.q_next.batch_size: self.batch_size})

        idx = np.arange(self.batch_size)
        q_target = reward_batch + \
            self.gamma*(q_next[idx, index_best_action])*(1 - terminal_batch)
 
        _ = self.q_eval.sess.run(self.q_eval.train_op,
                                 feed_dict={self.q_eval.states: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target,
                                            self.q_eval.seq_len: self.seq_length,
                                            self.q_eval.batch_size: self.batch_size,
                                            self.q_eval.state_in: state,
                                            self.q_eval._reward: self.reward['result'],
                                            self.q_eval._waitingtime: self.reward['total_waiting'],
                                            self.q_eval._delay: self.reward['total_delay']})

        if self.mem_cntr % 400==0:
            summary1, _ = self.q_eval.sess.run([self.q_eval.write_op, self.q_eval.train_op],
                                        feed_dict={self.q_eval.states: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target,
                                            self.q_eval.seq_len: self.seq_length,
                                            self.q_eval.batch_size: self.batch_size,
                                            self.q_eval.state_in: state,
                                            self.q_eval._reward: self.reward['result'],
                                            self.q_eval._waitingtime: self.reward['total_waiting'],
                                            self.q_eval._delay: self.reward['total_delay']})
            self.q_eval.writer.add_summary(summary1)
            self.q_eval.writer.flush()

    def test(self, state):
        actions, lstm_state = self.q_eval.sess.run([self.q_eval.Q_values, self.q_eval.cell_state],
                                       feed_dict={self.q_eval.states: state,
                                                  self.q_eval.state_in: self.state_out,
                                                  self.q_eval.seq_len: 1,
                                                  self.q_eval.batch_size: 1})
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        action = np.argmax(actions)
        action_ht = np.zeros((self.n_actions))
        action_ht[action] = 1.	
        return self.action_decoder(action_ht, self.all_list) 

    def get_qval(self, state):
        q_values, lstm_state = self.q_eval.sess.run([self.q_eval.Q_values, self.q_eval.cell_state],
                                       feed_dict={self.q_eval.states: state,
                                                  self.q_eval.state_in: self.state_out,
                                                  self.q_eval.seq_len: 1,
                                                  self.q_eval.batch_size: 1})
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        return q_values
    
    def test_initialiser(self):
        c_init = np.zeros((1, self.LSTM_DIM), np.float32)
        h_init = np.zeros((1, self.LSTM_DIM), np.float32)
        self.state_out = (c_init, h_init)

    def reset(self):
        self.state_out = (np.zeros(self.state_out[0].shape),np.zeros(self.state_out[1].shape))
       
    def save_models(self, episode_number):
        self.episode_number = episode_number
        self.q_eval.save_checkpoint(epi_num = self.episode_number)
        self.q_next.save_checkpoint(epi_num = self.episode_number)

    def load_models(self, filename):
        self.q_eval.load_checkpoint(filename)
        #self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t, e))
