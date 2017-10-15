import os

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
print('Current directory: '+str(DIRECTORY))

# import filter_env
import gym
from ddpg import *
import gc
import time
import pickle

from decimal import Decimal
gc.enable()

ENV_NAME = 'BipedalWalkerHardcore-v2' # 'BipedalWalker-v2' , 'InvertedPendulum-v1'
EPISODES = 10
TEST_trial = 1
Monitor_tr = 1
Monitor_test = 1

MONITOR_DIR = DIRECTORY+'/results/gym_ddpg'
SUMMARY_DIR = DIRECTORY+'/results/tf_ddpg'

def main():
    env = gym.make(ENV_NAME)
    agent = DDPG(env, DIRECTORY)

    env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=lambda episode: episode % (Monitor_test+TEST_trial) <=  TEST_trial, force=True)

    f = open(SUMMARY_DIR+'/log.txt', 'w')

    reward_list_te = np.array([])
    for episode in range(agent.actor_network.last_epi+1,EPISODES+agent.actor_network.last_epi+1):
        t0 = time.time()
        state = env.reset()
        action = np.zeros(shape=(agent.action_dim,))

        ep_reward = 0.
        ep_info = 0.
        for step in range(env.spec.timestep_limit):
            if step == 0:
                init_actor_hidden1_c = agent.state_initialiser(shape=(1,agent.actor_network.rnn_size),mode='g') # output(or cell) hidden state
                init_actor_hidden1_m = agent.state_initialiser(shape=(1,agent.actor_network.rnn_size),mode='g') # memory hidden state
                actor_init_hidden_cm = (init_actor_hidden1_c, init_actor_hidden1_m)#((init_temporal1_c, init_temporal1_m),(actor_init_temporal2_c, actor_init_temporal2_m))

            state_trace = np.expand_dims(state,axis=0)
            action_trace = np.expand_dims(action, axis=0)
            action, actor_last_hidden_cm = agent.action(state_trace,actor_init_hidden_cm,episode,noisy=True) # Act
            actor_init_hidden_cm = actor_last_hidden_cm

            next_state,reward,done,info = env.step(action)
            print('vel='+str(state[2]))
            state = next_state

            ep_reward += reward
            ep_info += info['true_reward']
            if done:
                t1 = time.time()
                logging_tr = 'episode{0}: Acc.Reward(Real)={1:.1f}({2:.1f})/ Time={3:.1f}s({4})'.format(episode,ep_reward,ep_info,t1-t0,step)
                f.write(logging_tr + '\n')
                f.flush()
                print(logging_tr)

                reward_list_te = np.append(reward_list_te, ep_reward)
                break

    f.close()

if __name__ == '__main__':
    main()
