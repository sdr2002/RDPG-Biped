import os
from collections import deque, OrderedDict
import random
import itertools
import numpy as np
import pickle
import time

class ReplayBuffer(object):

    def __init__(self, buffer_size,DIRECTORY,max_len_trajectory,last_saved_epi):
        self.DIRECTORY = DIRECTORY
        self.buffer_size = buffer_size

        self.len_trajectory = max_len_trajectory #> 1600 + TRACE_LENGTH
        self.initial_add = True
        # self.buffer = deque()
        self.buffer_dict = OrderedDict()
        self.categories = 6 #(s,a,s',r,done)
        self.dim = np.zeros(shape=(6,),dtype=int)
        self.maxed = False

        self.num_experiences = 0
        self.earliest_episode = 0 # earliest-episode of the buffer
        self.next_episode = 0 # earliest-episode to feed into the buffer
        self.last_saved_epi = last_saved_epi
        self.index = 0 # current time-step in current episode
        self.counts_per_epi = np.array([])

        # Prioritised experience replay
        self.discount_length = 10
        self.epsilon = 1.0 # 1 - self.epsilon = prob to apply prioritised replay
        self.gamma = 0.99
        self.annealing = 0.97 # forget factor for each visit during sampling
        self.rewards_per_epi = np.array([])

        if len([buf for buf in os.listdir(self.DIRECTORY+'/results') if 'buffer' in buf])>1:
            self.merge_pickle()
        else:
            self.load_pickle()


    def get_batch(self, batch_size, max_trace_length, time_step):
        # Control your len(trace) with this!!
        trace_length = int(np.min([max_trace_length,np.amax(self.counts_per_epi[:-1]), time_step+1]))
        if (trace_length < max_trace_length):# Just in case when the trace_length of the model is too long but episodes end ealier
            length_mode = 'short'
        else:
            length_mode = 'long'
        available_epi_mask = np.where(self.counts_per_epi >= trace_length)

        ### Dict - by epi
        if not len(available_epi_mask) == 1:
            weights = self.counts_per_epi[available_epi_mask][:-1]
            epi_list = np.arange(self.earliest_episode,self.next_episode)[available_epi_mask][:-1]
            reward_list = np.array(self.rewards_per_epi)[:-1]
        else:
            weights = self.counts_per_epi[available_epi_mask]
            epi_list = np.arange(self.earliest_episode,self.next_episode)[available_epi_mask]
            reward_list = np.array(self.rewards_per_epi)
        try:
            # Prioritised replay sampling based on rewards/len(epi)
            if length_mode == 'long':
                priority_smoother_epi = np.random.uniform(low=0.,high=1.) # Dynamic inverse Gibbs temperature
                if self.epsilon < priority_smoother_epi:
                    weights = weights * np.exp(reward_list/weights * priority_smoother_epi)
            sampled_epi_list = np.random.choice(epi_list,size=batch_size,replace=True, p=weights / np.sum(weights))
        except Exception as e:
            print('Epi sampling err: ' + str(e))
            raise

        return_rdpg = np.zeros(shape=(batch_size,trace_length,self.categories),dtype=object)
        row = 0

        for sampled_epi in sampled_epi_list:
            try:
                if length_mode == 'long':
                    priority_smoother_index = np.random.uniform(low=0.,high=1.)
                    if self.epsilon < priority_smoother_index: # PV(r) based sampling
                        index_range = np.arange(trace_length-1,self.counts_per_epi[sampled_epi-self.earliest_episode],dtype=int)
                        if 0.5 <= np.random.uniform(low=0.,high=1.): # Sweet memory
                            index_weights = np.exp(np.asarray(self.buffer_dict[sampled_epi][:,5][index_range],np.float)*priority_smoother_index) # Weight sampling via PV(r) for each index
                        else: # Bitter memory
                            index_weights = np.exp(-np.asarray(self.buffer_dict[sampled_epi][:,5][index_range],np.float)*priority_smoother_index) # Weight sampling via PV(r) for each index
                        index_last = np.random.choice(index_range,size=1,p=index_weights/np.sum(index_weights))[0]
                    else:
                        index_last = np.random.randint(low=trace_length-1, high=self.counts_per_epi[sampled_epi-self.earliest_episode], size=1, dtype=int)[0]
                    self.buffer_dict[sampled_epi][index_last,5] *= self.annealing # Smooth annealing
                elif length_mode == 'short':
                    # index_last = np.random.randint(low=trace_length-1, high=self.counts_per_epi[sampled_epi-self.earliest_episode], size=1, dtype=int)[0]
                    index_last = trace_length - 1
                    # index_last = trace_length-1
            except Exception as e:
                print('Index sample err: ' +str(e))
                raise

            try:
                return_rdpg[row] = self.buffer_dict[sampled_epi][index_last-trace_length+1:index_last+1]
            except Exception as e:
                print('Return_rdpge err: ' + str(e))
                raise
            row += 1
        return return_rdpg, trace_length
        ###

    def add(self, state, action, reward, new_state, done, epi, time_step):
        t0 = time.time()
        experience = [epi, time_step, state, action, reward, reward, new_state, done] # Will do prioritisation with the second reward!
        if self.initial_add is True:
            self.dim = np.zeros(shape=(len(experience)),dtype=object)
            for i in range(len(experience)):
                self.dim[i] = np.shape(experience[i])
            self.categories = len(experience) # 6 for (s,a,s',r,done)
            self.initial_add = False

        if self.num_experiences < self.buffer_size:
            pass
        else:
            if not self.maxed:
                print('Buffer maxed!')
                self.maxed = True
            print('Pop out the oldest episode')
            self.buffer_dict.pop(self.earliest_episode)
            self.earliest_episode += 1

            self.num_experiences -= self.counts_per_epi[0] # self.num_experiences = self.count()
            self.counts_per_epi = np.delete(self.counts_per_epi,0,axis=0)
            self.rewards_per_epi = np.delete(self.rewards_per_epi,0,axis=0)

        if self.next_episode == epi: # First time-step for each epi
            self.buffer_dict[epi] = np.zeros(shape=(self.len_trajectory,self.categories),dtype=object) # 9 = len(experience), self.len_trajectory > 1600 + TRACE_LENGTH = env.spec.timestep_limit + TRACE_LENGTH
            self.index = 0
            try:
                self.buffer_dict[epi][self.index] = experience
            except Exception as e:
                print('Experience saving err(1st time step):', experience)
                raise
            self.next_episode += 1
            self.counts_per_epi = np.append(self.counts_per_epi, 1)
            self.rewards_per_epi = np.append(self.rewards_per_epi, experience[2])

        else:
            self.index += 1
            try:
                self.buffer_dict[epi][self.index] = experience
            except Exception as e:
                print('Experience saving err(non-1st time step):'+str(e))
                raise
            self.counts_per_epi[-1] += 1
            self.rewards_per_epi[-1] = self.rewards_per_epi[-1]*float(self.index)/(float(self.index)+1.) + experience[4] * 1./(float(self.index)+1.)

        if done:
            deletion = np.arange(self.index+1,self.len_trajectory)
            self.buffer_dict[epi] = np.delete(self.buffer_dict[epi],deletion,axis=0)

            reward_series_discounted = self.buffer_dict[epi][:,4]
            for i in range(self.discount_length):
                if i == 0:
                    self.reward_series_discounted = reward_series_discounted
                else:
                    reward_series_discounted = self.gamma * np.append(np.delete(reward_series_discounted,0,0),0.)
                    self.reward_series_discounted += reward_series_discounted
            self.buffer_dict[epi][:,5] = self.reward_series_discounted

        t1 = time.time()
        dt = t1-t0
        # if done:
        #     print 'dt_add_buffer:' + str(dt)
        self.num_experiences += 1

    def save_pickle(self):
        try:
            pickle.dump(obj=self.buffer_dict,file=open(name=self.DIRECTORY+'/results/buffer.p',mode='wb'))
            print("Successfuly saved: RDPG Buffer")
            print("Buffer length saved: " + str(self.num_experiences))
        except Exception as e:
            print("Error on saving buffer: " + str(e))

    def load_pickle(self,merged=False):
        try:
            if not merged:
                self.buffer_dict = pickle.load(file=open(name=self.DIRECTORY+'/results/buffer.p',mode='rb'))
            self.num_experiences = sum(len(v) for v in self.buffer_dict.itervalues())
            print("Successfuly loaded: RDPG Buffer")
            print("Buffer length loaded: " + str(self.num_experiences))

            episode_keys = self.buffer_dict.keys()
            for i in range(episode_keys[0],episode_keys[-1]+1):
                self.buffer_dict[i-episode_keys[-1]+self.last_saved_epi] = self.buffer_dict.pop(i)
            episode_keys = self.buffer_dict.keys()
            self.earliest_episode = episode_keys[0] # earliest-episode in the buffer
            self.next_episode = episode_keys[-1]+1
            self.counts_per_epi = np.array([len(v) for v in self.buffer_dict.itervalues()],dtype=np.float)
            self.rewards_per_epi = np.array([np.sum(a=v[:,4]) for v in self.buffer_dict.itervalues()],dtype=np.float)
        except Exception as e:
            print("Could not find old buffer: " + str(e))
            self.earliest_episode= self.last_saved_epi+1
            self.next_episode = self.last_saved_epi+1
            # raise
            pass

    def merge_pickle(self):
        print('---Buffer merging....')
        self.buffer_dict = OrderedDict()
        first_episode = 0
        for file in os.listdir(self.DIRECTORY+'/results'):
            if 'buffer' in file:
                new = pickle.load(file=open(name=self.DIRECTORY+'/results/'+file,mode='rb'))
                episode_keys = new.keys()
                for i in range(episode_keys[0],episode_keys[-1]+1):
                    self.buffer_dict[i-episode_keys[0]+first_episode] = new.pop(i)
                first_episode += (episode_keys[-1]+1)
                os.remove(self.DIRECTORY+'/results/'+file)
        self.num_experiences += sum(len(v) for v in self.buffer_dict.itervalues())
        self.save_pickle()
        self.load_pickle(merged=True)
        print('---Merging done')

        pass
        # self.buffer_dict = merged_dict

if __name__ == '__main__':
    replay_buffer = ReplayBuffer(buffer_size=int(1e6),\
                                 DIRECTORY=os.path.dirname(os.path.realpath(__file__)),\
                                 max_len_trajectory=2001)
