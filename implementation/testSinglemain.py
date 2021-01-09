import aienvs
from improved_DQRN import DeepQNetwork, Agent
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb
import csv

path = os.getcwd()

def saver(data, name, iternr):
    path = os.getcwd()
    name = str(name)
    filename = 'test_result/single/'+ name + str(iternr) +'.csv'
    pathname = os.path.join(path, filename)
    outfile = open(pathname, 'w')
    writer = csv.writer(outfile)
    writer.writerows(map(lambda x:[x], data))

def save(test_result, iternr):
    for keys in test_result.keys():
        saver(test_result[keys], keys, iternr)

def fileinitialiser(test_result, i):
    path = os.getcwd()
    for key in test_result.keys():
        filename = 'test_result/single/'+ key + '_' + str(i)  +'.csv'
        pathname = os.path.join(path, filename)
        with open(pathname, "w") as my_empty_csv:
            pass

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

    with open("configs/vertical_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)
    mem_size = None
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                  act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32, test=True)

    total_number_simulation = 8
    stack_size = 1

    def result_initialiser():
        test_result = {}
        test_result['score'] = []
        test_result['delay'] = []
        test_result['waitingtime'] = []
        test_result['traveltime'] = []
        return test_result

    test_result = result_initialiser()
    path = os.getcwd()
    
    for i in range(0, 1000000, 10000):

        if i>10000:
            env.reset_test_cntr()
            observation, average_train_times, average_train_time = env.reset(i)
            test_result['traveltime'].append(average_train_time)
            print(test_result['traveltime'])
            save(test_result, i)
        else:
            observation= env.reset()

        observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
        agent.reset()

        try:
            filename = 'deepqnet.ckpt-' + str(i)
            chkpt = os.path.join(*[path, 'tmp', 'q_eval', filename])
            agent.load_models(chkpt)
            print('LOADED CHECKPOINT:', filename)
        except:
            env.close()

        for j in range(total_number_simulation):
            done = False   
            try:
                observation, average_train_times, average_train_time = env.reset(j)
                test_result['traveltime'].append(average_train_time)
                print(test_result['traveltime'])
            except:
                observation = env.reset()
            observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
            agent.reset()

            while not done:
                action = agent.test(stacked_state)
                observation_, reward, done, info = env.step(action)

                print(reward['result'])

                observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)

                test_result['score'].append(reward['result'])
                test_result['delay'].append(reward['total_delay'])
                test_result['waitingtime'].append(reward['total_waiting'])

                observation = observation_
                stacked_state = stacked_state_
    
    observation, average_test_times, average_test_time = env.reset(i)
    test_result['traveltime'].append(average_test_time)
    save(test_result, i)
    env.close()
