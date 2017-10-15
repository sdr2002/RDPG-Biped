# -----------------------------------
# Deep Deterministic Policy Gradient for RNN-DPG suit
# Author: Doo Re Song. Baseline is built from Flood Sung's code on DDPG(https://github.com/songrotek/DDPG)
# Date: 2017.09.14
# -----------------------------------
import tensorflow as tf
import numpy as np
import scipy.stats as stats
from ou_noise import OUNoise
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from replay_buffer_epi import ReplayBuffer
import time
# Hyper Parameters:

REPLAY_BUFFER_SIZE = int(1e5)
REPLAY_START_SIZE = int(2e3) # highgly recommend to be bigger than 2*max_len_trajectory
BATCH_SIZE = 2
GAMMA = 0.99
TRACE_LENGTH = 3
OPT_LENGTH = 2
# trace_length = 100 # (trace_length -1)/TEMP_ABSTRACT must be an integer
TEMP_ABSTRACT = 1

class DDPG:
    """docstring for DDPG"""
    def __init__(self, env, DIRECTORY):
        self.batch_size = BATCH_SIZE
        self.replay_start_size = REPLAY_START_SIZE# self.sub_batch_size = BATCH_SIZE / n_gpu

        self.name = 'DDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))

        self.trace_length = TRACE_LENGTH
        self.temp_abstract = TEMP_ABSTRACT
        self.actor_network = ActorNetwork(self.sess,BATCH_SIZE,self.state_dim,self.action_dim,self.temp_abstract,DIRECTORY)
        self.critic_network = CriticNetwork(self.sess,BATCH_SIZE,self.state_dim,self.action_dim,self.temp_abstract,DIRECTORY)
        
        # initialize replay buffer
        max_len_trajectory = self.environment.spec.timestep_limit + 1 # trace_length
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE,DIRECTORY,max_len_trajectory,self.actor_network.last_epi)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        ###
        self.diff = 0.
        self.discounting_mat_dict = {}
        ###

    def state_initialiser(self,shape,mode='g'):
        if mode == 'z': #Zero
            initial = np.zeros(shape=shape)
        elif mode == 'g': #Gaussian
            # initial = stats.truncnorm.rvs(a=-0.02/0.01,b=0.02/0.01,loc=0.,scale=0.01,size=shape)
            initial = np.random.normal(loc=0.,scale=1./float(shape[1]),size=shape)
        else: # May do some adaptive initialiser can be built in later
            raise NotImplementedError
        return initial

    def train(self, time_step):#,time_step):
        ###1) Get-batch data for opt
        minibatch, trace_length = self.replay_buffer.get_batch(self.batch_size, self.trace_length, time_step)#, self.trace_length)
        try:
            state_trace_batch = np.stack(minibatch[:,:,2].ravel()).reshape(self.batch_size,trace_length,self.state_dim)
            action_trace_batch = np.stack(minibatch[:,:,3].ravel()).reshape(self.batch_size,trace_length,self.action_dim)

            next_state_batch = np.stack(minibatch[:,-1,6].ravel()).reshape(self.batch_size,1,self.state_dim)
            next_state_trace_batch = np.concatenate([state_trace_batch,next_state_batch],axis=1)

            reward_trace_batch = np.stack(minibatch[:,:,4].ravel()).reshape(self.batch_size,trace_length,1)
            done_trace_batch = np.stack(minibatch[:,:,7].ravel()).reshape(self.batch_size,trace_length,1)

        except Exception as e:
            print(str(e))
            raise

        ###2) Painfully initialise initial memories of LSTMs: not super-efficient, but no error guaranteed from tf's None-type zero-state problem
        init_actor_hidden1_cORm_batch = self.state_initialiser(shape=(self.batch_size,self.actor_network.rnn_size),mode='z')
        actor_init_h_batch = (init_actor_hidden1_cORm_batch,init_actor_hidden1_cORm_batch)#((init_hidden1_cORm_batch,init_hidden1_cORm_batch),(init_actor_hidden2_cORm_batch,init_actor_hidden2_cORm_batch))

        init_critic_hidden1_cORm_batch = self.state_initialiser(shape=(self.batch_size,self.critic_network.rnn_size),mode='z')
        critic_init_h_batch = (init_critic_hidden1_cORm_batch,init_critic_hidden1_cORm_batch)#,(init_critic_hidden3_cORm_batch,init_critic_hidden3_cORm_batch))
        ###

        self.dt_list = np.zeros(shape=(15,))
        self.dt_list[-1] = time.time()
        if trace_length <= OPT_LENGTH:
            target_actor_init_h_batch = actor_init_h_batch
            target_critic_init_h_batch = critic_init_h_batch
            pass
        else:
            ### memory stuff 
            actor_init_h_batch = self.actor_network.action(state_trace_batch[:,:-OPT_LENGTH,:], actor_init_h_batch, mode =1)
            target_actor_init_h_batch  = actor_init_h_batch
            critic_init_h_batch = self.critic_network.evaluation(state_trace_batch[:,:-OPT_LENGTH,:], action_trace_batch[:,:-OPT_LENGTH,:], critic_init_h_batch, mode =1)
            target_critic_init_h_batch = critic_init_h_batch

            state_trace_batch = state_trace_batch[:,-OPT_LENGTH:,:]
            next_state_trace_batch = next_state_trace_batch[:,-(OPT_LENGTH+1):,:]
            action_trace_batch = action_trace_batch[:,-OPT_LENGTH:,:]
            reward_trace_batch = reward_trace_batch[:,-OPT_LENGTH:,:]
            done_trace_batch = done_trace_batch[:,-OPT_LENGTH:,:]
        self.dt_list[0] = time.time() - np.sum(self.dt_list)

        ###3) Obtain target output
        next_action_batch = self.actor_network.target_action(next_state_trace_batch, init_temporal_hidden_cm_batch=target_actor_init_h_batch)
        self.dt_list[1] = time.time() - np.sum(self.dt_list)
        next_action_trace_batch = np.concatenate([action_trace_batch, np.expand_dims(next_action_batch,axis=1)],axis=1)
        self.dt_list[2] = time.time() - np.sum(self.dt_list)
        target_lastQ_batch = self.critic_network.target_q_trace(next_state_trace_batch, next_action_trace_batch, init_temporal_hidden_cm_batch=target_critic_init_h_batch)
        self.dt_list[3] = time.time() - np.sum(self.dt_list)

        # Control the length of time-step for gradient
        if trace_length <= OPT_LENGTH:
            update_length = np.minimum(trace_length,OPT_LENGTH // 1) #//denom: 2(opt1) #1(opt0) #OPT_LENGTH(opt2)
        else:
            update_length = OPT_LENGTH // 1 #//denom: 2(opt1) #1(opt0) #OPT_LENGTH(opt2)

        target_lastQ_batch_masked = target_lastQ_batch * (1.- done_trace_batch[:,-1])
        rQ = np.concatenate([np.squeeze(reward_trace_batch[:,-update_length:],axis=-1), target_lastQ_batch_masked],axis=1)
        self.dt_list[4] = time.time() - np.sum(self.dt_list)

        try:
            discounting_mat = self.discounting_mat_dict[update_length]
        except KeyError:
            discounting_mat = np.zeros(shape=(update_length,update_length+1),dtype=np.float)
            for i in range(update_length):
                discounting_mat[i,:i] = 0.
                discounting_mat[i,i:] = GAMMA ** np.arange(0.,-i+update_length+1)
            discounting_mat = np.transpose(discounting_mat)
            self.discounting_mat_dict[update_length] = discounting_mat
        try:
            y_trace_batch = np.expand_dims(np.matmul(rQ,discounting_mat),axis=-1)
        except Exception as e:
            print('?')
            raise
        self.dt_list[5] = time.time() - np.sum(self.dt_list)

        ###4)Train Critic: get next_action, target_q, then optimise
        critic_grad = self.critic_network.train(y_trace_batch,update_length,state_trace_batch, action_trace_batch, init_temporal_hidden_cm_batch=critic_init_h_batch)
        self.dt_list[6] = time.time() - np.sum(self.dt_list)

        ###5) Train Actor: while updated critic, we declared the dQda. Hence sess,run(dQda*dadParam_actor), then optimise actor
        for i in range(update_length):
            actor_init_h_batch_trace = (np.expand_dims(actor_init_h_batch[0],axis=1), np.expand_dims(actor_init_h_batch[1],axis=1))
            critic_init_h_batch_trace = (np.expand_dims(critic_init_h_batch[0],axis=1), np.expand_dims(critic_init_h_batch[1],axis=1))
            if i == 0:
                actor_init_h_batch_stack = actor_init_h_batch_trace
                critic_init_h_batch_stack = critic_init_h_batch_trace
            else:
                actor_init_h_batch_stack = (np.concatenate((actor_init_h_batch_stack[0],actor_init_h_batch_trace[0]),axis=1),np.concatenate((actor_init_h_batch_stack[1],actor_init_h_batch_trace[1]),axis=1))
                critic_init_h_batch_stack = (np.concatenate((critic_init_h_batch_stack[0],critic_init_h_batch_trace[0]),axis=1),np.concatenate((critic_init_h_batch_stack[1],critic_init_h_batch_trace[1]),axis=1))
            action_trace_batch_for_gradients, actor_init_h_batch = self.actor_network.action_trace(np.expand_dims(state_trace_batch[:,i],1), init_temporal_hidden_cm_batch=actor_init_h_batch)
            critic_init_h_batch = self.critic_network.evaluation_trace(np.expand_dims(state_trace_batch[:,i],1), np.expand_dims(action_trace_batch[:,i],1), init_temporal_hidden_cm_batch=critic_init_h_batch)
            if i == 0:
                action_trace_batch_for_gradients_stack = action_trace_batch_for_gradients
            else:
                action_trace_batch_for_gradients_stack = np.concatenate((action_trace_batch_for_gradients_stack,action_trace_batch_for_gradients),axis=1)
                
        self.dt_list[7] = time.time() - np.sum(self.dt_list)
        state_trace_batch_stack = np.reshape(state_trace_batch,(self.batch_size*update_length,1,self.state_dim))
        action_trace_batch_stack = np.reshape(action_trace_batch,(self.batch_size*update_length,1,self.action_dim))
        action_trace_batch_for_gradients_stack = np.reshape(action_trace_batch_for_gradients_stack,(self.batch_size*update_length,1,self.action_dim))
        actor_init_h_batch_stack = (np.reshape(actor_init_h_batch_stack[0],(self.batch_size*update_length,self.actor_network.rnn_size)), np.reshape(actor_init_h_batch_stack[1],(self.batch_size*update_length,self.actor_network.rnn_size)))
        critic_init_h_batch_stack = (np.reshape(critic_init_h_batch_stack[0],(self.batch_size*update_length,self.critic_network.rnn_size)), np.reshape(critic_init_h_batch_stack[1],(self.batch_size*update_length,self.critic_network.rnn_size)))

        q_gradient_trace_batch = self.critic_network.gradients(1, state_trace_batch_stack, action_trace_batch_for_gradients_stack, init_temporal_hidden_cm_batch=critic_init_h_batch_stack)
        self.dt_list[8] = time.time() - np.sum(self.dt_list)

        # Update the actor policy using the sampled gradient:
        actor_grad = self.actor_network.train(q_gradient_trace_batch,1, state_trace_batch_stack, action_trace_batch_stack, init_temporal_hidden_cm_batch=actor_init_h_batch_stack)
        self.dt_list[9] = time.time() - np.sum(self.dt_list)

        # Update the target networks via EMA & Indicators
        # self.critic_network.update_target()
        self.dt_list[10] = time.time() - np.sum(self.dt_list)
        # self.actor_network.update_target()
        self.dt_list[11] = time.time() - np.sum(self.dt_list)

        # actor_diff = self.actor_network.get_diff()
        self.dt_list[12] = time.time() - np.sum(self.dt_list)
        # critic_diff = self.critic_network.get_diff()
        self.dt_list[13] = time.time() - np.sum(self.dt_list)

        self.dt_list = np.delete(self.dt_list,-1)
        return actor_grad, critic_grad, # actor_diff, actor_grad, critic_diff, critic_grad

    def action(self,state_trace,init_hidden_cm,epi,noisy=True):
        # Select action a_t according to the current policy and exploration noise
        action, last_hidden_cm= self.actor_network.action([state_trace], init_hidden_cm, mode=2)
        if noisy:
            noise = self.exploration_noise.noise()#epi)
            return action+noise, last_hidden_cm#, dt#, np.linalg.norm(noise)
        else:
            return action, last_hidden_cm

    def evaluation(self,state_trace,action_trace,action_last,init_hidden_cm):
        return self.critic_network.evaluation([state_trace],[action_trace],action_last,init_hidden_cm, mode=2) #q_value, last_hidden_cm

    # def perceive(self,actor_init_hidden_cm,critic_last_hidden_cm,state,action,reward,next_state,done,time_step,epi):
    def perceive(self,state,action,reward,next_state,done,time_step,epi):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        # self.replay_buffer.add(actor_init_hidden_cm,critic_last_hidden_cm,state,action,reward,next_state,done,epi)
        done = float(done)
        self.replay_buffer.add(state,action,reward,next_state,done,epi,time_step)

        # Store transitions to replay start size then start training
        if (self.replay_buffer.num_experiences >  REPLAY_START_SIZE):
            # Non-zero diff should be found
            self.actor_grad, self.critic_grad = self.train(time_step)
            # self.actor_diff, self.actor_grad, self.critic_diff, self.critic_grad = self.train(time_step)
        else:
            # Zero diff as is not trained
            # self.actor_diff = 0.
            self.actor_grad = 0.
            # self.critic_diff = 0.
            self.critic_grad = 0.

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()










