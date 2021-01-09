import aienvs
from multi_DQRN import DeepQNetwork, Agent
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb
import csv
 
def saver(data, name):
    name = str(name)
    filename = name+'.csv'
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerows(map(lambda x:[x], data))


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))

        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx, :] = frame

    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]

        stacked_frames[buffer_size-1, :] = frame

    stacked_frame = stacked_frames
    stacked_state = stacked_frames.transpose(1,2,0)[None, ...]

    return stacked_frame, stacked_state


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    with open("configs/eight_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)

    #load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')

    mem_size = 30000

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                  act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32)

    #if load_checkpoint:
     #   agent.load_models()

    maximum_time_steps = 500
    train_time_steps_score = []
    train_time_steps_delay = []
    train_time_steps_waitingtime = []
    train_episode_score = []
    train_travel_time = []
    stack_size = 1
    i = 0

    test_int = 10000 #test interval
    maximum_test_time = 4000 #no of test simulation
    total_simulation = 7

    test_reward = []
    test_delay = []
    test_waitingtime = []
    test_average_travel_time = []
    

    print("Loading up the agent's memory with random gameplay")
    while agent.mem_cntr < mem_size:
        done = False
        observation = env.reset()
        observation, stacked_state = stack_frames(stacked_frames = None, frame = observation, buffer_size = stack_size)

        while (not done) and (agent.mem_cntr < mem_size):
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames = observation, frame = observation_, buffer_size = stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            agent.upgrade()
            observation = observation_
            stacked_state = stacked_state_
            print('MEMORY_COUNTER: ', agent.mem_cntr)
    print("Done with random game play. Game on.")

    while i < maximum_time_steps:
        done = False
        if i>0:    
            try:
                observation, average_train_times, average_train_time = env.reset(i)
                train_travel_time.append(average_train_time)
                print(train_travel_time)
                train_episode_score.append(score)
            except:
                observation = env.reset()
            observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
            agent.reset()
        else:     
            observation = env.reset()
            observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
            agent.reset()

        score = 0
        while (not done) and  i < maximum_time_steps:
            action = agent.choose_action(stacked_state)
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            train_time_steps_score.append(reward['result'])
            train_time_steps_delay.append(reward['total_delay'])
            train_time_steps_waitingtime.append(reward['total_waiting'])
            score +=reward['result']
            observation = observation_
            stacked_state = stacked_state_
            print("reward: ", reward['result'])
            print("waiting Time: ", reward['total_waiting'])
            print("Delay: ", reward['total_delay'])
            agent.learn()

            test_it = i
            
            if test_it % test_int==0 and i>0:
                done = False 
                try:
                    observation, average_train_times, average_train_time = env.reset(i)
                    train_travel_time.append(average_train_time)
                    print(train_travel_time)
                except:
                    observation = env.reset()
                observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                agent.reset()
                test_it +=1

                for tt in range(total_simulation):
                    sim_step = 0
                    while (not done):
                        action = agent.test(stacked_state) 
                        observation_, reward, done, info = env.step(action)
                        observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)
                        agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
                        test_reward.append(reward['result'])
                        test_delay.append(reward['total_delay'])
                        test_waitingtime.append(reward['total_waiting'])
                        agent.upgrade()
                        sim_step+=1
                        print('SIMULATION STEP: ',sim_step)
                        print('T: ', tt)
                    done = False
                    try:
                        observation, average_travel_times, average_travel_time = env.reset(tt+1)
                        test_average_travel_time.append(average_travel_time)
                        print(test_average_travel_time)
                    except:  
                        observation = env.reset()
                    observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                    agent.reset()
            i +=1
            try:
                if sim_step >0:
                    sim_step=0
                    done=True
            except:
                continue
            agent.upgrade()            
            if i == maximum_time_steps:
                try:
                    observation, average_train_times, average_train_time = env.reset(i)
                    train_travel_time.append(average_train_time)
                    print(train_travel_time)
                    train_episode_score.append(score)
                except:
                    observation = env.reset()
                observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                agent.reset()
                break

    saver(data=train_time_steps_score, name='train_time_steps_score_reward')
    saver(data=train_time_steps_delay, name='train_time_steps_score_delay')
    saver(data=train_time_steps_waitingtime, name='train_time_steps_score_waitingtime')
    saver(data=train_episode_score, name='train_episode_score')
    saver(data=train_travel_time, name='train_travel_time')
    saver(data=test_reward, name='test_reward')
    saver(data=test_delay, name='test_delay')
    saver(data=test_waitingtime, name = 'test_waitingtime')
    saver(data=test_average_travel_time, name='test_average_travel_time') 
    env.close()
