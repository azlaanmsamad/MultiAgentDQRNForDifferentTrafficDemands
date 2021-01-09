import numpy as np
import glob
import pandas as pd
import os
import sys
import yaml
import logging
from LoggedTestCase import LoggedTestCase
from aienvs.Sumo.sumogym import SumoGymAdapter
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from factor_graph import factor_graph
import csv
import time
from maxplus import maxplus

class experimentor():
    
    def __init__(self, total_simulation=8):
        logging.info("Starting test_traffic_new")
        #with open("configs/testconfig.yaml", 'r') as stream:
        with open("configs/eight_twofactor_config.yaml", 'r') as stream:
            try:
                parameters=yaml.safe_load(stream)['parameters']
            except yaml.YAMLError as exc:
                print(exc)
        self.i = 0
        #self.modelnumber = modelnr
        self.env = SumoGymAdapter(parameters)
        self.total_simulation = total_simulation
        self._parameters = parameters
        for keys in self._parameters['testmodelnr'].keys():
            self.index = keys
        self.num_agents=0
        for keys in self._parameters['lightPositions'].keys():
            self.num_agents +=1
  
        #************************  DO NOT FORGET TO CHANGE THE PADDER AND CONFIG FILE   ***********************
        self.factor_graph = factor_graph(factored_graph=self._parameters['factored_agents'], 
                                         num_agents=self.num_agents,
                                         parameters = self._parameters,
                                         car_pr = self._parameters['car_pr'], 
                                         factored_agent_type=self._parameters['factored_agent_type'],
                                         modelnr = self._parameters['testmodelnr'],
                                         algorithm = self._parameters['coordination_algo'])
        if self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus = maxplus(regular_factor=self._parameters['factored_agents'], agent_neighbour_combo=self._parameters['agent_neighbour_combo'], max_iter=self._parameters['max_iter'])
            print('USING MAXPLUS ALGORITHM FOR COORDINATION')
        self.result_initialiser()
        self.algo_timer = []
        self.fileinitialiser()
        self.factored_agent_type = self._parameters['factored_agent_type']
        #self.result_appender('results/six_intersection/0.4/maxplus/trial/150000')


    def tester(self):
        path = os.getcwd()
        self.modelnumber=None
        if self.modelnumber ==None:
            self.test()
        
            for self.model in range(10000, 1010000, 10000):
                self.env.reset_test_cntr()
            
                if self.index == 'individual':
                    filename = 'single' + '_' + str(self._parameters['car_pr']) + '_deepqnet.ckpt-' + str(self.model)
                    try:
                        chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'q_eval', filename])
                        self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                    except: 
                        chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'trial1', 'q_eval', filename])
                        self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                    print('LOADED CHECKPOINT:', filename)

                elif self.index == 'vertical':
                    filename = 'two_' + str(self._parameters['car_pr']) + '_deepqnet.ckpt-' + str(self.model)
                    chkpt = os.path.join(*[path, 'tmp', 'two_intersection', str(self._parameters['car_pr']), 'q_eval', filename])
                    self.factor_graph.Q_function_dict['vertical'].load_models(chkpt)
                    print('LOADED CHECKPOINT:', filename)
                    
                self.test()

        else:
            for self.model in range(self.modelnumber, 250000, 10000):
                self.env.reset_test_cntr()

                if self.index == 'individual':
                    filename = 'single' + '_' + str(self._parameters['car_pr']) + '_deepqnet.ckpt-' + str(self.model)
                    try:
                        chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'q_eval', filename])
                        self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                    except:
                        chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'trial1', 'q_eval', filename])
                        self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                    print('LOADED CHECKPOINT:', filename)

                elif self.index == 'vertical':
                    filename = 'two_' + str(self._parameters['car_pr']) + '_deepqnet.ckpt-' + str(self.model)
                    chkpt = os.path.join(*[path, 'tmp', 'two_intersection', str(self._parameters['car_pr']), 'q_eval', filename])
                    self.factor_graph.Q_function_dict['vertical'].load_models(chkpt)
                    print('LOADED CHECKPOINT:', filename)

                self.test()        

    def result_initialiser(self):
        self.test_result = {}
        self.test_result['result'] = []
        self.test_result['num_teleports'] = []
        self.test_result['emergency_stops'] = []
        self.test_result['total_delay'] = []
        self.test_result['total_waiting'] = []
        self.test_result['traveltime'] = []

    def file_finder(self, data):
        path = os.getcwd()
        self.result_path = os.path.join(path, data)
        reward_files = [files for files in glob.glob(os.path.join(path, data, 'result*'))]
        data_dict = {}
        self.modelnumber=None
        if len(reward_files)!=0:
            sorted_reward_files = []
            for i in range(len(reward_files)):
                csv = reward_files[i].split('result')[-1]
                sorted_reward_files.append(int(csv.split('.')[0]))
            sorted_reward_files.sort()
            data_dict['result'] =  'result' + str(sorted_reward_files[-1]) + '.csv'
            data_dict['traveltime']  = 'traveltime' + str(sorted_reward_files[-1]) + '.csv'
            #data_dict['algo_timer'] = 'algo_timer' + str(sorted_reward_files[-1]) + '.csv'
            self.modelnumber = sorted_reward_files[-1] + 10000
        print('MODEL NUMBER BEING USED IS: ', self.modelnumber)
        return data_dict, bool(data_dict)

    def result_appender(self, file_path):
        data_dict, bool_value = self.file_finder(file_path)
        if bool_value == True:
            for keys in data_dict.keys():
                path = os.path.join(self.result_path, data_dict[keys])
                data = pd.read_csv(path, header=None)
                for j in range(len(data)):
                    if keys =='algo_timer':
                        self.algo_timer.append(data.values[j][0])
                    else:
                        self.test_result[keys].append(data.values[j][0])
        else:
            print("NO PREVIOUS SAVED RESULT")

    def store_result(self, reward):
        for keys in reward.keys():
            self.test_result[keys].append(reward[keys])

    def shape(self, ob):
        for keys in ob[0].keys():
            print(keys, ob[0][keys].shape)

    def store_tt(self, tt):
        self.test_result['traveltime'].append(tt)

    def saver(self, data, name, iternumber):
        path = os.getcwd()
        filename = str(name) + str(iternumber) + '.csv'
        pathname = os.path.join(*[path, 'results', self._parameters['scene'], str(self._parameters['car_pr']), self._parameters['coordination_algo'], 'trial', 'zeroiter',filename])
        outfile = open(pathname, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], data))
        outfile.close()	

    def fileinitialiser(self):
        path = os.getcwd()
        for key in self.test_result.keys():
            filename = key + '.csv'
            pathname = os.path.join(*[path, 'results', self._parameters['scene'], str(self._parameters['car_pr']), self._parameters['coordination_algo'], 'trial', 'zeroiter', filename])
            if os.path.exists(os.path.dirname(pathname)):
                print('Result directroy already exists: ', pathname)
            else:
                os.makedirs(os.path.dirname(pathname))

    def file_rename(self, name, iternr):
        path = os.getcwd()
        res_dir = os.path.join(path, 'test_result', str(self.result_folder))
        oldname = str(name) + str(iternr-10000)  + '.csv'
        newname = str(name) + str(iternr) + '.csv'
        os.rename(res_dir+ '/' + oldname, res_dir + '/' + newname)

    def reset(self):
        self.factor_graph.reset()

    def qarr_key_changer(self, q_arr):
        q_val = {}
        for keys in q_arr.keys():
            q_val[str(self._parameters['factored_agents'][keys])] = q_arr[keys]
        return q_val

    def take_action(self, state_graph):
        q_arr = self.factor_graph.get_factored_Q_val(state_graph)
        if self._parameters['coordination_algo'] == 'brute':
            start = time.process_time()
            sum_q_value, best_action, sumo_act = self.factor_graph.b_coord(q_arr)
            self.algo_timer.append(time.process_time() - start)
            print(sum_q_value, sumo_act)
        elif self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus.initialise_again()
            q_arr = self.qarr_key_changer(q_arr)
            start = time.process_time()
            payoff, sumo_act = self.maxplus.max_plus_calculator(q_arr)
            self.algo_timer.append(time.process_time() - start)
        else:
            start = time.process_time()
            sumo_act = self.factor_graph.individual_coord(q_arr)
            self.algo_timer.append(time.process_time() - start)
        return sumo_act

    def save(self, data, iternr):
        for key in data.keys():
            result = data[key]
            self.saver(data=result, name=key, iternumber=iternr)

    def stack_frames(self, stacked_frames, frame, buffer_size, config):
        if stacked_frames is None:
            stacked_frames = np.zeros((buffer_size, *frame.shape))
            for idx, _ in enumerate(stacked_frames):
                if config=='horizontal':
                    stacked_frames[idx, :] = frame.transpose()
                else:
                    stacked_frames[idx, :] = frame
        else:
            stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
            if config== 'horizontal':
                stacked_frames[buffer_size-1, :] = frame.transpose()
            else:
                stacked_frames[buffer_size-1, :] = frame

        stacked_frame = stacked_frames
        stacked_state = stacked_frames.transpose(1,2,0)[None, ...]

        return stacked_frame, stacked_state

    def stack_state_initialiser(self):
        self.stacked_state_dict = {}
        self.ob_dict = {}

    def stacked_graph(self, ob, initial=True):
        for keys in self._parameters['factored_agents'].keys():
            if initial==True:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=None, frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys]) 

            else:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=self.ob_dict[keys], frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys])

        return self.ob_dict, self.stacked_state_dict

    def six_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '0':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[0][keys] = ob[0][keys][:84,:]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[0][keys] = ob[0][keys][:84,:]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,16)), 'constant', constant_values=(0,0))
            elif keys == '6':
                ob[0][keys] = ob[0][keys][:84, :]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def three_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(1,0)), 'constant', constant_values= (0,0))
        return ob

    def four_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '0':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((14,14),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(14,14)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def six_ind_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[0][keys] = np.pad(ob[0][keys], ((1,0),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[0][keys] = np.pad(ob[0][keys], ((1,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[0][keys] = np.pad(ob[0][keys], ((1,0),(2,2)), 'constant', constant_values=(0,0))
        return ob

    def eight_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '0':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys =='6':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='7':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='8':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='9':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def eight_ind_padder(self, ob):
        for keys in ob[0].keys():
            if (keys == '0' or keys =='1' or keys =='2' or keys=='3'):
                ob[0][keys] = np.pad(ob[0][keys], ((1,0),(0,0)), 'constant', constant_values=(0,0))
        return ob

    def test(self):
        for i in range(self.total_simulation):
            done = False
            self.stack_state_initialiser()
            if i > 0:
                try:
                    ob, avg_travel_times, avg_travel_time = self.env.reset(i)
                    self.store_tt(avg_travel_time)
                    print(self.test_result['traveltime'])
                except:
                    ob = self.env.reset()
            else:
                ob = self.env.reset()
            ob = self.eight_padder(ob)
            self.ob_dict, self.stacked_state_dict = self.stacked_graph(ob=ob[0], initial=True)
            self.reset()
            while not done:
                action = self.take_action(self.stacked_state_dict)
                ob_, reward, done, info = self.env.step(action)
                ob_ = self.eight_padder(ob_)
                print(reward[1]['result'])
                self.store_result(reward[1])
                self.ob_dict, self.stacked_state_dict  = self.stacked_graph(ob_[0], initial=False)
        ob, avg_travel_times, avg_travel_time = self.env.reset(i)
        self.store_tt(avg_travel_time)
        self.save(self.test_result, self.model)
        

if __name__=="__main__":
    #************************************************CHANGE PADDER*****************************************
    exp = experimentor(total_simulation=8)
    exp.tester()
