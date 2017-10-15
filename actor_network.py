import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.python.client import timeline

# Hyper-Parameters
CONV_CH = 2
LAYER1_SIZE = 4 - CONV_CH
# LAYER2_SIZE = 256
RNN_SIZE = 4
# LAYER3_SIZE = 256

LEARNING_RATE = 1e-3
TAU = 0.

class ActorNetwork:
    """docstring for ActorNetwork"""

    # def __init__(self, sess, state_dim, action_dim, trace_length,temp_abstract,DIRECTORY):
    def __init__(self, sess, batch_size, state_dim, action_dim, temp_abstract,DIRECTORY):
        self.DIRECTORY = DIRECTORY

        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.temp_abstract = temp_abstract
        self.rnn_size= RNN_SIZE
        # self.rnn_layer= RNN_LAYER

        # create behaviour actor network
        self.xav = tf.contrib.layers.xavier_initializer()
        self.state_trace_input, self.init_hidden_cm, \
        self.action_trace_output, self.last_hidden_cm, \
        self.net = \
            self.create_network(state_dim, action_dim)


        # # create target actor network
        # self.var_count = len(self.net)
        # self.target_state_trace_input, self.target_init_hidden_cm, \
        # self.target_action_trace_output, self.target_last_hidden_cm,\
        # self.target_net, self.target_update_op, self.target_assign_op = \
        #     self.create_target_network(state_dim, action_dim, self.net)
        #
        # ###
        # self.norm_diff = self.diff(self.net, self.target_net)
        # ###

        # # define training rules
        self.create_training_method()
        self.sess.run(tf.global_variables_initializer())
        # update target net
        # self.update_target()
        # save & upload
        self.load_network()

        # # Graph...
        # self.summary_op = tf.summary.merge_all()#tf.summary.merge(inputs=[self.net[i].name for i in range(self.var_count)])#[var.name for var in tf.trainable_variables() if 'behaviour' in var.name])
        # self.writer = tf.summary.FileWriter(logdir=self.DIRECTORY+'/results/tf_ddpg', graph=sess.graph_def)
        # self.action_taken = 0

    def create_training_method(self):
        self.q_gradient_trace_input = tf.placeholder("float", [None, None, self.action_dim])
        self.update_length = tf.placeholder(tf.int32)
        self.parameters_gradients = tf.gradients(self.action_trace_output[:,-self.update_length:,:], self.net, -self.q_gradient_trace_input)
        self.grad_norm = tf.reduce_sum([tf.norm(grad) for grad in self.parameters_gradients])
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.net))

    def network(self, input_list, init=False):
        state_trace_agent, state_trace_ridar, (init_hidden1_c, init_hidden1_m) = input_list

        ###1)State embeddings part
        with tf.variable_scope('s_emb') as s_emb:
            W_conv1 = tf.get_variable("W_conv1", shape=[1, 4, 1, CONV_CH], initializer=self.xav)
            b_conv1 = tf.get_variable("b_conv1", shape=[CONV_CH], initializer=self.xav)
            conv1 = tf.nn.relu(tf.nn.bias_add(value=tf.nn.conv2d(input=tf.pad(tf.expand_dims(state_trace_ridar, axis=-1), [[0,0],[0,0],[2,2],[0,0]]),\
                                                                 filter=W_conv1, strides=[1,1,1,1], padding='VALID'),\
                                              bias=b_conv1))
            conv1_size = conv1.get_shape().as_list()
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,conv1_size[2]], strides=[1,conv1_size[2]], padding="valid")

            # W2_ridar = tf.get_variable("W2_ridar", shape=[1,1,CONV_CH, LAYER2_SIZE], initializer=self.xav)
            # b2_ridar = tf.get_variable("b2_ridar", shape=[LAYER2_SIZE], initializer=self.xav)
            # layer2_in1 = tf.nn.relu(tf.nn.bias_add(\
            #     tf.nn.conv2d(input=pool1, filter=W2_ridar,strides=[1,1,1,1], padding='VALID'),
            #     bias=b2_ridar))

            state_trace_agent_size = state_trace_agent.get_shape().as_list()
            W1_agent = tf.get_variable("W1_agent", shape=[1, state_trace_agent_size[2], 1, LAYER1_SIZE], initializer=self.xav)
            b1_agent = tf.get_variable("b1_agent", shape=[LAYER1_SIZE], initializer=self.xav)
            layer1_agent = tf.nn.relu(tf.nn.bias_add(\
                value=tf.nn.conv2d(tf.expand_dims(state_trace_agent,-1), filter=W1_agent, strides=[1,1,1,1], padding='VALID'), \
                bias=b1_agent))

            # W2_agent = tf.get_variable("W2_agent", shape=[1,1,LAYER1_SIZE, LAYER2_SIZE], initializer=self.xav)
            # b2_agent = tf.get_variable("b2_agent", shape=[LAYER2_SIZE], initializer=self.xav)
            # layer2_in2 = tf.nn.relu(tf.nn.bias_add(\
            #     tf.nn.conv2d(input=layer1_agent, filter=W2_agent,strides=[1,1,1,1], padding='VALID'),
            #     bias=b2_agent))

            # layer2 = tf.concat([layer2_in1, layer2_in2],axis=-1)
            layer2 = tf.concat([pool1, layer1_agent],axis=-1)
            timesteps_rep_list = tf.squeeze(input=layer2, axis=2)
        # timesteps_rep_list = tf.concat([state_trace_agent, state_trace_ridar],axis=-1) # Skip 1)

        ###3) Action-trace generation
        # ## -0: No rnn
        # Wf = tf.Variable(tf.truncated_normal([1, timesteps_rep_list.get_shape().as_list()[2], 1, self.action_dim], 0., 2e-1))
        # bf = tf.Variable(tf.random_uniform([self.action_dim], -3e-2, 3e-2))
        # finals = tf.nn.relu(tf.nn.bias_add( \
        #     tf.nn.conv2d(input=tf.expand_dims(timesteps_rep_list,-1), filter=Wf,strides=[1,1,1,1], padding='VALID'), \
        #     bias=bf))
        # rnn_hidden_cm = (init_hidden1_c, init_hidden1_m)
        # return tf.squeeze(finals,axis=2), rnn_hidden_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name] #

        # ## -1: 1st Rnn for action output
        with tf.name_scope('a_gen') as a_gen:
            cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            init_state1 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden1_c, init_hidden1_m)
            rnn_states, rnn_hidden_cm = tf.nn.dynamic_rnn(cell1, timesteps_rep_list, dtype=tf.float32, initial_state=init_state1)#!
        # return rnn_states, rnn_hidden_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name]

        # ## -2: 2nd Rnn for action output
        # cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        # init_state1 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden1_c, init_hidden1_m)
        # rnn_states, rnn_hidden_cm = tf.nn.dynamic_rnn(cell1, timesteps_rep_list, dtype=tf.float32, initial_state=init_state1)#!
        # cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.action_dim, state_is_tuple=True) #!
        # init_state2 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden2_c, init_hidden2_m)
        # cell_wrap = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2],state_is_tuple=True) #!
        # init_state_wrap = (init_state1, init_state2)
        # rnn_states, rnn_hidden_cm = tf.nn.dynamic_rnn(cell_wrap, timesteps_rep_list, dtype=tf.float32, initial_state=init_state_wrap)#!
        # return finals, rnn_hidden_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name] #!

        # ## -3: Take whole time-step into account from rnn to fnn
        #     W3 = tf.get_variable("W3", shape=[1, self.rnn_size, 1, LAYER3_SIZE], initializer=self.xav)
        #     b3 = tf.get_variable("b3", shape=[LAYER3_SIZE], initializer=self.xav)
        #     layer3 = tf.nn.elu(tf.nn.bias_add( \
        #         tf.nn.conv2d(input=tf.expand_dims(rnn_states,-1), filter=W3,strides=[1,1,1,1], padding='VALID'), \
        #         bias=b3))

            Wf = tf.get_variable("Wf", shape=[1, self.rnn_size, 1, self.action_dim], initializer=self.xav) # shape=[1, 1, LAYER3_SIZE, self.action_dim]
            bf = tf.get_variable("bf", shape=[self.action_dim], initializer=self.xav)
            finals = tf.nn.tanh(tf.nn.bias_add( \
                tf.nn.conv2d(input=tf.expand_dims(rnn_states,-1), filter=Wf,strides=[1,1,1,1], padding='VALID'), \
                bias=bf))#input=layer3
            action_trace = tf.squeeze(finals,axis=2)

        ####Final)Return action and network params
        net = [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name]
        return action_trace, rnn_hidden_cm, net #

        # return final, rnn_hidden_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name] #param_list#

    def create_network(self, state_dim, action_dim): #, trace_length):
        # state_trace_input = tf.placeholder("float", [None, trace_length, state_dim])
        state_trace_input = tf.placeholder("float", [None, None, state_dim])

        # state_trace_agent_hull, state_trace_agent_legs, state_trace_ridar = tf.split(avg_state_trace_input, [4, 10, 10],axis=2)
        state_trace_agent, state_trace_ridar = tf.split(state_trace_input, [14, 10],axis=2)

        init_hidden1_c = tf.placeholder("float", [None,self.rnn_size])
        init_hidden1_m = tf.placeholder("float", [None,self.rnn_size])
        # init_hidden2_c = tf.placeholder("float", [None,self.action_dim])
        # init_hidden2_m = tf.placeholder("float", [None,self.action_dim])
        init_hidden_cm = (init_hidden1_c, init_hidden1_m)#((init_hidden1_c, init_hidden1_m),(init_hidden2_c, init_hidden2_m))

        with tf.variable_scope('actor_behaviour') as ab:
            input_list = [state_trace_agent, state_trace_ridar, init_hidden_cm]
            action_trace_output, last_hidden_cm, net = self.network(input_list)
        return state_trace_input, init_hidden_cm, action_trace_output, last_hidden_cm, net

    # def create_target_network(self, state_dim, action_dim, net):#, trace_length):
    #     # state_trace_input = tf.placeholder("float", [None, trace_length, state_dim])
    #     state_trace_input = tf.placeholder("float", [None, None, state_dim])
    #
    #     # state_trace_agent_hull, state_trace_agent_legs, state_trace_ridar = tf.split(avg_state_trace_input, [4, 10, 10],axis=2)
    #     state_trace_agent, state_trace_ridar = tf.split(state_trace_input, [14, 10],axis=2)
    #
    #     init_hidden1_c = tf.placeholder("float", [None,self.rnn_size])
    #     init_hidden1_m = tf.placeholder("float", [None,self.rnn_size])
    #     # init_hidden2_c = tf.placeholder("float", [None,self.action_dim])
    #     # init_hidden2_m = tf.placeholder("float", [None,self.action_dim])
    #     init_hidden_cm = (init_hidden1_c, init_hidden1_m)#((init_hidden1_c, init_hidden1_m),(init_hidden2_c, init_hidden2_m))
    #
    #     ema = tf.train.ExponentialMovingAverage(decay=1. - TAU)
    #     target_update_op = ema.apply(net)
    #
    #     with tf.variable_scope('actor_target') as at:
    #         input_list = [state_trace_agent, state_trace_ridar, init_hidden_cm]
    #         action_trace_output, last_hidden_cm, target_net \
    #             = self.network(input_list)
    #
    #         target_assign_op = [[] for i in range(self.var_count)]
    #         for i in range(self.var_count):
    #             target_assign_op[i] = target_net[i].assign(ema.average(net[i]))
    #
    #     return state_trace_input, init_hidden_cm, action_trace_output, last_hidden_cm, target_net, target_update_op, target_assign_op,

    # def update_target(self):
    #     self.sess.run(self.target_update_op)
    #     for i in range(self.var_count):
    #         self.sess.run(self.target_assign_op[i].op)

    def train(self, q_gradient_trace_batch, update_length, state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch):
        try:
            _, grad_norm = self.sess.run([self.optimizer,self.grad_norm], feed_dict={
                self.q_gradient_trace_input: q_gradient_trace_batch,
                self.update_length: update_length,
                self.state_trace_input: state_trace_batch,
                self.action_trace_output: action_trace_batch,
                self.init_hidden_cm: init_temporal_hidden_cm_batch
            })
            return grad_norm
        except Exception as e:
            print('?')
            raise

    def action_trace(self, state_trace_batch, init_temporal_hidden_cm_batch):
        return self.sess.run([self.action_trace_output, self.last_hidden_cm],feed_dict={
            self.state_trace_input: state_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm_batch
        })

    def target_action(self, state_trace_batch, init_temporal_hidden_cm_batch, mode=0):
        action_trace, last_h = self.sess.run([self.action_trace_output, self.last_hidden_cm],feed_dict={
            self.state_trace_input: state_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm_batch
        })#target_s
        if mode == 0:
            return action_trace[:,-1,:]
        elif mode == 1:
            return last_h

    def action(self, state_trace, init_temporal_hidden_cm, mode=2):
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        action, last_h = self.sess.run([self.action_trace_output, self.last_hidden_cm],feed_dict={
            self.state_trace_input: state_trace,
            self.init_hidden_cm: init_temporal_hidden_cm
        })#,options=run_options, run_metadata=run_metadata)

        # # Create the Timeline object, and write it to a json
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as js:
        #     js.write(ctf)
        if mode == 0:
            return action[0][0]
        elif mode == 1:
            return last_h
        elif mode == 2:
            return action[0][0], last_h #(last_h[0],last_h[1])

    ###
    # def diff(self, target, behave):
    #     diff = 0.
    #     for param_pair in zip(target, behave):
    #         diff += tf.norm(tf.subtract(param_pair[0], param_pair[1]))
    #     return diff
    #
    # def get_diff(self):
    #     return self.sess.run(self.norm_diff)
    ###

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.DIRECTORY+'/saved_actor_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.last_epi = ''
            modelchkpath = str(checkpoint.model_checkpoint_path)
            modelchkpath_befepi = self.DIRECTORY+'/saved_actor_networks/actor-network-'
            for i in range(len(modelchkpath)-len(modelchkpath_befepi)):
                self.last_epi += modelchkpath[i+len(modelchkpath_befepi)]
            self.last_epi = int(self.last_epi)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            self.last_epi = -1
            print("Could not find old network weights - Actor")

    def save_network(self, episode):
        print('save actor-network...', episode)
        self.saver.save(self.sess, self.DIRECTORY+'/saved_actor_networks/actor-network', global_step=episode)
