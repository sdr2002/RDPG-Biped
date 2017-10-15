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

ENV_NAME = 'BipedalWalkerHardcore-v2'# , 'InvertedPendulum-v1'
EPISODES = 5000
TEST_trial = 1
Monitor_tr = 200
Monitor_test = 200
# TRACE_LENGTH imported from ddpg.py

MONITOR_DIR = DIRECTORY+'/results/gym_ddpg'
SUMMARY_DIR = DIRECTORY+'/results/tf_ddpg'

def main():
    def loader():
        try:
            perf_dict = pickle.load(file=open(name=DIRECTORY+'/results/perf.p',mode='rb'))
            try:
                return perf_dict['R_tr'], perf_dict['R_te'], perf_dict['dParam_Actor'], perf_dict['dParam_Critic'], perf_dict['R_tr_real']#, \
            except Exception as e:
                return perf_dict['R_tr'], perf_dict['R_te'], perf_dict['dParam_Actor'], perf_dict['dParam_Critic'], np.array([])
                   #perf_dict['dEMA_Actor'], perf_dict['dEMA_Critic']
        except Exception as e:
            print('Perf_dict load error:'+str(e))
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])#, np.array([])#, np.array([])
    def saver(flag_error=False):
        # Save model periodically
        agent.actor_network.save_network(episode)
        agent.critic_network.save_network(episode)
        # Buffer pickling
        if not flag_error:
            agent.replay_buffer.save_pickle()
            perf_dict = {'R_tr': reward_list_tr, 'R_te': reward_list_te,\
                         'dParam_Actor': actor_grad_avg_list, 'dParam_Critic': critic_grad_avg_list}#,\
                         # 'dEMA_Actor': actor_diff_avg_list, 'dEMA_Critic': critic_diff_avg_list}#
            try:
                perf_dict['R_tr_real'] = reward_list_tr_real
            except Exception as e:
                pass
            pickle.dump(obj=perf_dict,file=open(name=DIRECTORY+'/results/perf.p',mode='wb'))# Performance pickling

    env = gym.make(ENV_NAME)
    agent = DDPG(env, DIRECTORY)

    env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, resume= True, force=False)

    ###
    # TRACE_LENGTH = agent.trace_length
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    # reward_list_tr, reward_list_te, actor_grad_avg_list, critic_grad_avg_list, actor_diff_avg_list, critic_diff_avg_list = loader()
    reward_list_tr, reward_list_te, actor_grad_avg_list, critic_grad_avg_list, reward_list_tr_real = loader()
    # actor_diff_avg_list = np.array([])
    # critic_diff_avg_list = np.array([])
    # noise_avg_list = np.array([])
    ###

    f = open(DIRECTORY+'/results/log.txt', 'w')

    best_reward = 365.
    forget = 0.99
    try:
        for episode in range(agent.actor_network.last_epi+1,EPISODES+agent.actor_network.last_epi+1):
            t0 = time.time()
            state = env.reset()
            action = np.zeros(shape=(agent.action_dim,))

            ###!
            ep_reward = 0.
            ep_info = 0.
            # actor_diff = np.array([])
            actor_grad = np.array([])
            # critic_diff = np.array([])
            critic_grad = np.array([])

            # noise = np.array([])
            # noise_avg = 0.
            ###!

            # Training:
            ## tick = 0

            for step in range(env.spec.timestep_limit):
                # Control on the initial state of RNN
                if step == 0:
                    init_actor_hidden1_c = agent.state_initialiser(shape=(1,agent.actor_network.rnn_size),mode='g') # output(or cell) hidden state
                    init_actor_hidden1_m = agent.state_initialiser(shape=(1,agent.actor_network.rnn_size),mode='g') # memory hidden state
                    actor_init_hidden_cm = (init_actor_hidden1_c, init_actor_hidden1_m)#((init_temporal1_c, init_temporal1_m),(actor_init_temporal2_c, actor_init_temporal2_m))

                    # critic_init_temporal2_c = agent.state_initialiser(shape=(1,agent.critic_network.rnn2_size),mode='z')
                    # critic_init_temporal2_m = agent.state_initialiser(shape=(1,agent.critic_network.rnn2_size),mode='z')
                    # critic_init_temporal3_c = agent.state_initialiser(shape=(1,1),mode='z') # output(or cell) hidden state
                    # critic_init_temporal3_m = agent.state_initialiser(shape=(1,1),mode='z') # memory hidden state
                    # critic_init_hidden_cm = ((init_temporal1_c, init_temporal1_m),(critic_init_temporal2_c, critic_init_temporal2_m),(critic_init_temporal3_c, critic_init_temporal3_m))
                #Run
                # if (episode % Monitor_tr == 0):
                #     env.render()

                state_trace = np.expand_dims(state, axis=0)
                action_trace = np.expand_dims(action, axis=0)
                action, actor_last_hidden_cm = agent.action(state_trace,actor_init_hidden_cm,episode,noisy=True) # Act

                # q_value, critic_last_hidden_cm = agent.evaluation(state_trace,action_trace,action,critic_init_hidden_cm) # Criticise
                next_state,reward,done,info = env.step(action)

                # Reaction from the environment
                agent.perceive(state,action,reward,next_state,done,step,episode)
                # if step % 50 == 1:
                #     try:
                #         dt_list_str = '/'.join(['%.0E' % Decimal(agent.dt_list[i]) for i in range(len(agent.dt_list))])
                #         print('dt for update: ' + dt_list_str)
                #     except Exception as e:
                #         pass
                # s,a trace stacker
                state = next_state# np.vstack((state_trace, next_state))
                actor_init_hidden_cm = actor_last_hidden_cm
                # critic_init_hidden_cm = critic_last_hidden_cm

                ####!
                ep_reward += reward
                ep_info += info['true_reward']
                # actor_diff = np.append(actor_diff,agent.actor_diff)
                actor_grad = np.append(actor_grad,agent.actor_grad)
                # critic_diff = np.append(critic_diff,agent.critic_diff)
                critic_grad = np.append(actor_grad,agent.critic_grad)
                # noise = np.append(noise, noise_norm)
                ####!

                if done:
                    # actor_diff_avg = np.average(actor_diff)
                    actor_grad_avg = np.average(actor_grad)
                    # critic_diff_avg = np.average(critic_diff)
                    critic_grad_avg = np.average(critic_grad)
                    # actor_diff_avg_list = np.append(actor_diff_avg_list, actor_diff_avg)
                    actor_grad_avg_list = np.append(actor_grad_avg_list, actor_grad_avg)
                    # critic_diff_avg_list = np.append(critic_diff_avg_list, critic_diff_avg)
                    critic_grad_avg_list = np.append(critic_grad_avg_list, critic_grad_avg)
                    # noise_avg = np.average(noise)
                    # noise_avg_list = np.append(noise_avg_list, noise_avg)
                    t1 = time.time()
                    # logging_tr = 'episode{0}: sum(Reward)={1:.1f}/ Time={2:.1f}s({3})/ A-C avg. dEM(G)=[{4:.2E}({5:.2E})-{6:.2E}({7:.2E})]'.format(episode,ep_reward,t1-t0,step,actor_diff_avg,actor_grad_avg,critic_diff_avg,critic_grad_avg)
                    logging_tr = 'ep{0}: Acc.R(Real)={1:.1f}({2:.1f})/ Time={3:.1f}s({4})/ A-C avg. Grad=({5:.2E})-({6:.2E})'.format(episode,ep_reward,ep_info,t1-t0,step,actor_grad_avg,critic_grad_avg)
                    f.write(logging_tr + '\n')
                    f.flush()
                    print(logging_tr)
                    reward_list_tr = np.append(reward_list_tr, ep_reward)

                    if best_reward < ep_reward:
                        best_reward = ep_reward
                        # Save best performed model
                        saver()
                    elif episode % Monitor_tr == Monitor_tr-1:
                        saver()
                        pass
                    break

    except Exception as e:
        logging_err = '=====Running Error:' + str(e) + '====='
        f.write(logging_err + '\n')
        f.flush()
        print(logging_err)
        f.close()
        # saver(flag_error=True)
        raise

    f.close()
    saver()

if __name__ == '__main__':
    main()
