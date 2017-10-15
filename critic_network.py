import tensorflow as tf
import numpy as np
import math

# Hyper-Parameters
CONV_CH = 2
LAYER1_SIZE = 4 - CONV_CH
LAYER2_SIZE = 6 - LAYER1_SIZE - CONV_CH
RNN_SIZE = 6
# LAYER3_SIZE = 256

LEARNING_RATE = 1e-3
TAU = 0.
L2 = 1e-5  # 1e-3 originally


class CriticNetwork:
    """docstring for CriticNetwork"""

    def __init__(self, sess, batch_size, state_dim, action_dim, temp_abstract,DIRECTORY):
        self.DIRECTORY = DIRECTORY

        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.temp_abstract = temp_abstract
        self.rnn_size= RNN_SIZE
        # self.rnn_layer= RNN_LAYER

        # create behaviour q network
        self.xav = tf.contrib.layers.xavier_initializer()
        self.state_trace_input, self.action_trace_input,self.init_hidden_cm, \
        self.q_value_trace_output, self.last_hidden_cm, \
        self.net \
            = self.create_q_network(state_dim, action_dim)#, self.trace_length)

        # # create target q network (the same structure with q network)
        # self.var_count = len(self.net)
        # self.target_state_trace_input, self.target_action_trace_input, self.target_init_hidden_cm, \
        # self.target_q_value_trace_output, self.target_last_hidden_cm, \
        # self.target_update_op, self.target_assign_op, self.target_net = \
        #     self.create_target_q_network(state_dim, action_dim, self.net)#, self.trace_length)

        # ###
        # self.norm_diff = self.diff(self.net, self.target_net)
        # ###

        # define training rules, initialization
        self.create_training_method()
        self.sess.run(tf.global_variables_initializer())
        # update target net
        # self.update_target()
        # save & upload
        self.load_network()

    def create_training_method(self):
        # Define training optimizer
        self.y_trace_input = tf.placeholder("float", [None, None, 1])
        self.update_length = tf.placeholder(tf.int32)
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_trace_input - self.q_value_trace_output[:,-self.update_length:,:])) + weight_decay

        self.parameters_gradients = tf.gradients(self.cost, self.net)
        self.grad_norm = tf.reduce_sum([tf.norm(grad) for grad in self.parameters_gradients])
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.net))

        self.action_gradients = tf.gradients(self.q_value_trace_output, self.action_trace_input)

    def network(self, input_list):
        state_trace_agent, state_trace_ridar, action_trace_input, (init_hidden2_c, init_hidden2_m) = input_list

        ###1)State embeddings part
        with tf.name_scope('s_emb') as s_emb:
            #
            W_conv1 = tf.get_variable("W_conv1", shape=[1, 4, 1, CONV_CH], initializer=self.xav)
            b_conv1 = tf.get_variable("b_conv1", shape=[CONV_CH], initializer=self.xav)
            conv1 = tf.nn.relu(tf.nn.bias_add(value=tf.nn.conv2d(input=tf.pad(tf.expand_dims(state_trace_ridar, axis=-1), [[0,0],[0,0],[2,2],[0,0]]),\
                                                                 filter=W_conv1, strides=[1,1,1,1], padding='VALID'),bias=b_conv1))
            conv1_size = conv1.get_shape().as_list()
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,conv1_size[2]], strides=[1,conv1_size[2]], padding="valid")

            state_trace_agent_size = state_trace_agent.get_shape().as_list()
            W1_agent = tf.get_variable("W1_agent", shape=[1, state_trace_agent_size[2], 1, LAYER1_SIZE], initializer=self.xav)
            b1_agent = tf.get_variable("b1_agent", shape=[LAYER1_SIZE], initializer=self.xav)
            layer1_agent = tf.nn.relu(tf.nn.bias_add(\
                tf.nn.conv2d(input=tf.expand_dims(state_trace_agent,axis=-1), filter=W1_agent,strides=[1,1,1,1], padding='VALID')\
                ,bias=b1_agent))

            layer1_state = tf.concat([pool1,layer1_agent],axis=-1)

            # ## -0: RNN(ver 11,12,13)
            # cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
            # init_state1 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden1_c, init_hidden1_m)
            # rnn1_states, rnn1_last_cm = tf.nn.dynamic_rnn(cell1, layer1_state, dtype=tf.float32, initial_state=init_state1, scope='rnn1')

            ## -1: FNN(ver 14)
            layer1_state = tf.squeeze(layer1_state, axis=-2)
            layer1_state_size = layer1_state.get_shape().as_list()
            W2_state = tf.get_variable("W2_state", shape=[1,layer1_state_size[-1],1,CONV_CH+LAYER1_SIZE], initializer=self.xav)
            b2_state = tf.get_variable("b2_state", shape=[CONV_CH+LAYER1_SIZE], initializer=self.xav)
            layer2_in1 = tf.nn.elu(tf.nn.bias_add(\
                value=tf.nn.conv2d(tf.expand_dims(layer1_state,axis=-1), filter=W2_state, strides=[1,1,1,1], padding='VALID'), \
                bias=b2_state))

        ###2)Action embeddings part
        with tf.name_scope('a_emb') as a_emb:
            W2_action = tf.get_variable("W2_action", shape=[1,self.action_dim,1,LAYER2_SIZE], initializer=self.xav)
            b2_action = tf.get_variable("b2_action", shape=[LAYER2_SIZE], initializer=self.xav)
            layer2_in2 = tf.nn.relu(tf.nn.bias_add(\
                tf.nn.conv2d(input=tf.expand_dims(action_trace_input,axis=-1), filter=W2_action,strides=[1,1,1,1], padding='VALID')\
                ,bias=b2_action))

        ###3)State-Action embeddings part
        with tf.name_scope('sa_emb') as sa_emb:
            timesteps_rep_list = tf.concat([layer2_in1,layer2_in2],axis=-1)
            timesteps_rep_list = tf.squeeze(timesteps_rep_list,axis=-2)

        ###4) Q-trace generation
        with tf.name_scope('q_gen') as q_gen:
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            init_state2 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden2_c, init_hidden2_m)
            # cell3 = tf.nn.rnn_cell.BasicLSTMCell(1, state_is_tuple=True)#!
            # init_state3 = tf.nn.rnn_cell.LSTMStateTuple(init_hidden3_c, init_hidden3_m)#!
            cell_wrap = cell2 #tf.nn.rnn_cell.MultiRNNCell([cell2, cell3],state_is_tuple=True)#!
            init_state_wrap = init_state2 #(init_state2, init_state3)#!

            rnn2_states, rnn2_last_cm = tf.nn.dynamic_rnn(cell_wrap, timesteps_rep_list, dtype=tf.float32, initial_state=init_state_wrap, scope='rnn2')
            # return rnn2_states, rnn2_last_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name]

            # ### Take whole time-step into account from rnn to fnn
            # W3 = tf.get_variable("W3", shape=[1, self.rnn_size, 1, LAYER3_SIZE], initializer=self.xav)
            # b3 = tf.get_variable("b3", shape=[LAYER3_SIZE], initializer=self.xav)
            # layer3 = tf.nn.bias_add(\
            #     tf.nn.conv2d(input=tf.expand_dims(rnn2_states,-1), filter=W3,strides=[1,1,1,1], padding='VALID'), \
            #     bias=b3)

            Wf = tf.get_variable("Wf", shape=[1, self.rnn_size, 1, 1], initializer=self.xav) #shape=[1, 1, LAYER3_SIZE, 1]
            bf = tf.get_variable("bf", shape=[1], initializer=self.xav)
            finals = tf.nn.bias_add(\
                tf.nn.conv2d(input=tf.expand_dims(rnn2_states,-1), filter=Wf,strides=[1,1,1,1], padding='VALID'),\
                bias=bf)#input=layer3

        ####Final)Return q-value and network params
        net = [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name]
        return tf.squeeze(finals,axis=2), rnn2_last_cm, net

        # #### Take the last time-step into account from rnn: not feasible if you optimise with trace of TD errors
        # W3 = tf.Variable(tf.truncated_normal([self.rnn_size, 1], 0., 2e-1))
        # b3 = tf.Variable(tf.random_uniform([1], -3e-2, 3e-2))
        # final = tf.identity(tf.matmul(self.rnn_states[:,-1], W3) + b3)
        # return final, rnn_hidden_cm, [v for v in tf.trainable_variables() if tf.contrib.framework.get_name_scope() in v.name]

    def create_q_network(self, state_dim, action_dim): #, trace_length):
        # state_trace_input = tf.placeholder("float", [None, self.trace_length, state_dim])
        state_trace_input = tf.placeholder("float", [None, None, state_dim])
        avg_state_trace_input = state_trace_input
        # state_concat = tf.concat([tf.expand_dims(state_trace_input[:,-1,:], axis=1) for i in range(self.temp_abstract)], axis=1)
        # avg_state_trace_input = tf.reduce_mean(tf.reshape( \
        #     tf.concat([state_trace_input[:,:-1,:],state_concat],axis=1),shape=[-1,(trace_length-1)/self.temp_abstract + 1,self.temp_abstract,state_dim]) \
        #                                        ,axis=2)
        state_trace_agent, state_trace_ridar = tf.split(avg_state_trace_input, [14, 10],axis=2)

        action_trace_input = tf.placeholder("float", [None, None, action_dim])
        avg_action_trace = action_trace_input#tf.concat([action_trace_input,action_last_input],axis=1)

        init_hidden1_c = tf.placeholder("float", [None,self.rnn_size])
        init_hidden1_m = tf.placeholder("float", [None,self.rnn_size])
        init_hidden_cm = (init_hidden1_c, init_hidden1_m)

        with tf.variable_scope('critic_behaviour',reuse=False) as cb:
            input_list = [state_trace_agent, state_trace_ridar, avg_action_trace, init_hidden_cm]
            q_value_trace_output, last_hidden_cm, net = self.network(input_list) # q_value_trace_output = tf.identity(tf.matmul(layer2, W3) + b3)

        return state_trace_input, action_trace_input, init_hidden_cm, q_value_trace_output, last_hidden_cm, net

    # def create_target_q_network(self, state_dim, action_dim, net):#, trace_length):
    #     state_trace_input = tf.placeholder("float", [None, None, state_dim])
    #     avg_state_trace_input = state_trace_input
    #     # state_concat = tf.concat([tf.expand_dims(state_trace_input[:,-1,:], axis=1) for i in range(self.temp_abstract)], axis=1)
    #     # avg_state_trace_input = tf.reduce_mean(tf.reshape( \
    #     #     tf.concat([state_trace_input[:,:-1,:],state_concat],axis=1),shape=[-1,(trace_length-1)/self.temp_abstract + 1,self.temp_abstract,state_dim]) \
    #     #                                        ,axis=2)
    #     state_trace_agent, state_trace_ridar = tf.split(avg_state_trace_input, [14, 10],axis=2)
    #
    #     action_trace_input = tf.placeholder("float", [None, None, action_dim])
    #
    #     target_avg_action_trace = action_trace_input#tf.concat([action_trace_input,action_last_input],axis=1)
    #
    #     init_hidden1_c = tf.placeholder("float", [None,self.rnn_size])
    #     init_hidden1_m = tf.placeholder("float", [None,self.rnn_size])
    #     init_hidden_cm = (init_hidden1_c, init_hidden1_m)
    #
    #     ema = tf.train.ExponentialMovingAverage(decay=1. - TAU)
    #
    #     target_update_op = ema.apply(net)
    #
    #     with tf.variable_scope('critic_target') as ct:
    #         input_list = [state_trace_agent, state_trace_ridar, target_avg_action_trace, init_hidden_cm]
    #         q_value_trace_output, last_hidden_cm, target_net \
    #             = self.network(input_list)#,param_list)
    #
    #         target_assign_op = [[] for i in range(self.var_count)]
    #         for i in range(self.var_count):
    #             target_assign_op[i] = target_net[i].assign(ema.average(net[i]))
    #
    #     return state_trace_input, action_trace_input, init_hidden_cm, q_value_trace_output, last_hidden_cm, target_update_op, target_assign_op, target_net

    # def update_target(self):
    #     self.sess.run(self.target_update_op)
    #     for i in range(self.var_count):
    #         self.sess.run(self.target_assign_op[i].op)

    def train(self, y_trace_batch, update_length, state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch):
        try:
            _, grad_norm = self.sess.run([self.optimizer,self.grad_norm], feed_dict={
                self.y_trace_input: y_trace_batch,
                self.update_length: update_length,
                self.state_trace_input: state_trace_batch,
                self.action_trace_input: action_trace_batch,
                self.init_hidden_cm: init_temporal_hidden_cm_batch
            })
            return grad_norm
        except Exception as e:
            print '?'
            raise

    def gradients(self, update_length, state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch):
        try:
            return self.sess.run(self.action_gradients, feed_dict={
                self.state_trace_input: state_trace_batch,
                self.action_trace_input: action_trace_batch,
                self.init_hidden_cm: init_temporal_hidden_cm_batch
            })[0][:,-update_length:,:]
        except Exception as e:
            print '?'

    def evaluation_trace(self, state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch):#(self, state_trace_batch, action_trace_batch):
        return self.sess.run(self.last_hidden_cm, feed_dict={
            self.state_trace_input: state_trace_batch,
            self.action_trace_input: action_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm_batch #!
        })

    def target_q_trace(self, state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch, mode=0):
        q_value_trace, last_h = self.sess.run([self.q_value_trace_output,self.last_hidden_cm], feed_dict={
            self.state_trace_input: state_trace_batch,
            self.action_trace_input: action_trace_batch,
            self.init_hidden_cm: init_temporal_hidden_cm_batch
        })#target_s,a
        if mode == 0:
            return q_value_trace[:,-1,:]
        elif mode == 1:
            return last_h

    def evaluation(self, state_trace, action_trace, init_temporal_hidden_cm, mode=2):#(self, state_trace_batch, action_trace_batch):
        q_value, last_h = self.sess.run([self.q_value_trace_output, self.last_hidden_cm], feed_dict={
            self.state_trace_input: state_trace,
            self.action_trace_input: action_trace,
            self.init_hidden_cm: init_temporal_hidden_cm #!
        })
        if mode == 0:
            return q_value[0][0]
        elif mode == 1:
            return last_h
        elif mode == 2:
            return q_value[0][0], last_h

    # ###
    # def diff(self, target, behave):
    #     diff = 0.
    #     for param_pair in zip(target, behave):
    #         diff += tf.norm(tf.subtract(param_pair[0], param_pair[1]))
    #     return diff
    #
    # def get_diff(self):
    #     return self.sess.run(self.norm_diff)
    # ###

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.DIRECTORY+'/saved_critic_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights - Critic")

    def save_network(self,episode):
        print('save critic-network...', episode)
        self.saver.save(self.sess, self.DIRECTORY+'/saved_critic_networks/critic-network', global_step = episode)

