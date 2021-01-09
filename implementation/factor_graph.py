import os
import time
import sys
import numpy as np
from improved_DQRN import Agent
import itertools as it
import pdb
import collections
import tensorflow as tf

class factor_graph():
   
    def __init__(self, factored_graph, num_agents, parameters, car_pr, factored_agent_type, modelnr, algorithm):
        self.factored_graph = factored_graph
        self.num_agents = num_agents
        self._parameters = parameters
        for keys in self._parameters['testmodelnr'].keys():
            self.index = keys
        self.car_pr = car_pr
        # Num_agents should be the actual number of intersections even in the case of individual algorithm
        self.factored_agent_type = factored_agent_type
        self.modelnr = modelnr
        self.algorithm = algorithm
        self.num_factors = len(factored_graph.keys())
        self.act_per_agent = 2
        self.action_space = [i for i in range(self.act_per_agent)]
        self.num_actions = self.act_per_agent**self.num_agents
        self.generate_agents()
        #self.load_models()
        self.set_action_list()

    def set_action_list(self):
        ''' This produces a tuple of actions of each agent in the form of (0,0,0,0) or 
            (0,1,0,1) where 0 and 1s are the actions and lenth of the tuple is equal 
            to the number of agents'''
        self.action_list = []
        for i in it.product(self.action_space, repeat= self.num_agents):
            self.action_list.append(i)

    def get_factored_Q_val(self, state_graph):
        '''Outputs Q value for all of the factored graph/agents as an array'''
        q_array= {}
        for keys in self.factored_agent_type.keys():
            q_array[keys] = self.get_q_value(state_graph[keys])
        return q_array


    def generate_agents(self):
        '''Generates q_network for factored agents which in general is two agents per factor'''
        self.Q_function_dict = {} 	
        mem_size = None
        if self.algorithm == 'individual':
            q_func = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                       act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32, replace_target=30000, q_next_dir="tmp/six/q_next", q_eval_dir="tmp/six/q_eval", test=True)
            self.Q_function_dict['individual']= q_func

        elif self.index == 'vertical':
            # *********************Change this depending on the type of factor chosen in case of Transfer Planning approach*****************************
            
            ver_q_func = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                       act_per_agent=2, num_agents=2, mem_size=mem_size, batch_size=32, replace_target=30000, q_next_dir="tmp/six/q_next", q_eval_dir="tmp/six/q_eval", test=True)
            self.Q_function_dict['vertical'] = ver_q_func

        else:
            three_q_func = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                      act_per_agent=2, num_agents=3, mem_size=mem_size, batch_size=32, replace_target=30000, q_next_dir="tmp/six/q_next", q_eval_dir="tmp/six/q_eval", test=True)
            self.Q_function_dict['three'] = three_q_func


    def model_load_type(self, modeltype:str, modelnr: str):
        path = os.getcwd()
        if modeltype==None:
            #try:
             #   modelname = 'tmp/' + 'single/' + str(self.car_pr) + '/q_eval/' + 'single_' + str(self.car_pr) + '_deepqnet.ckpt-' + str(modelnr) 
            #except:
            modelname = 'tmp/' + 'q_eval/' + 'deepqnet.ckpt-' + str(modelnr)
        else:
            #try:
             #   modelname = 'tmp/'+ str(modeltype) + '/'  + str(self.car_pr) + '/q_eval/' + str(modeltype) + '_' + str(self.car_pr)  + '_deepqnet.ckpt-' + str(modelnr)
            #except:
            modelname = 'tmp/'+ str(modeltype) + '/q_eval/' + str(modeltype) + '_deepqnet.ckpt-' + str(modelnr)
        return os.path.join(path, modelname)

    def load_models(self):
        if self.algorithm == 'individual':
            individual_checkpoint_file = self.model_load_type(modelnr=self.modelnr['individual'], modeltype=None)
            self.Q_function_dict['individual'].load_models(individual_checkpoint_file)
        elif self.index == 'vertical':
            ver_checkpoint_file = self.model_load_type(modelnr=self.modelnr['vertical'], modeltype='vertical')
            self.Q_function_dict['vertical'].load_models(ver_checkpoint_file)
        else:
            #three_checkpoint_file = self.model_load_type(modelnr=self.modelnr['three'], modeltype='three')
            path = os.getcwd()
            three_checkpoint_file = os.path.join(path, 'tmp/three/q_eval', 'three_deepqnet.ckpt-260000')
            self.Q_function_dict['three'].load_models(three_checkpoint_file)

    def reset(self):
        for keys in self.Q_function_dict.keys():
            self.Q_function_dict[keys].reset()
   
    def get_q_value(self, local_state):
        '''returns q_value for the factored agents based on their local state'''
        if self.algorithm == 'individual':
            q_value = self.Q_function_dict['individual'].get_qval(local_state)
        elif self.index == 'vertical':
            q_value = self.Q_function_dict['vertical'].get_qval(local_state)
        else:
            q_value = self.Q_function_dict['three'].get_qval(local_state)
        return q_value	

    def store_exp(self, factored_exp):
        '''Used to store transitions (state, action, reward, next_state, terminal )'''
        for key in factored_exp.keys():
            self.Q_function[key].store_transition(factored_exp[key][0], factored_exp[key][1], factored_exp[key][2], factored_exp[key][3], factored_exp[key][4])

    def choose_action(self, factored_state):
        '''choose action based on the local state'''
        actions = {}
        for keys in self.factored_graph.keys():
            actions[keys] = self.Q_function_dict[keys].choose_action(factored_state[keys])
        return actions

    def agents_to_action(self, agent_list, actions):
        tup = []
        for idx, val in enumerate(agent_list):
            tup.append(actions[val])
        return tuple(tup)

    def b_coord(self, q_array):
        sum_q_value = []
        #self.index='vertical'
        #Considering the fact that the factored q value comprises only of two agents
        if self.index == 'vertical':
            action_tuple = [i for i in it.product((0,1), repeat=2)]
        else:
            action_tuple = [i for i in it.product((0,1), repeat=3)]
        for i in range(len(self.action_list)):
            q_val = 0
            all_actions = self.action_list[i]
            for keys in self.factored_graph.keys():
                act_tup = self.agents_to_action(self.factored_graph[keys], all_actions)
                for key, val in enumerate(action_tuple):
                    if act_tup == val:
                        index = key
                        break
                q_val += q_array[keys][0][index]
            sum_q_value.append(q_val)
        idx = np.random.choice(np.flatnonzero(np.asarray(sum_q_value)==np.asarray(sum_q_value).max()))
        best_action = self.action_list[idx]
        sumo_act = self.act_to_dict(best_action)
        return  sum_q_value[idx], best_action, sumo_act

    def individual_coord(self, q_arr):
        best_action = collections.OrderedDict()
        for keys in q_arr.keys():
            index = np.random.choice(np.flatnonzero(q_arr[keys] == q_arr[keys].max()))
            best_action[keys] = index
        return best_action
  
    def act_to_dict(self, best_action):
        action = collections.OrderedDict()
        for indexx, val in enumerate(best_action):
            action[str(indexx)] = val
        return action           

 
if __name__=="__main__":
    import numpy as np
    import csv
    factored_agents= {"0": [0, 1],
                      "1": [2, 3],
                      "2": [0, 2],
                      "3": [1, 3]}
  
