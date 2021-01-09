import os
import pdb
import sys
import numpy as np
np.random.seed(0)
#from factor_graph import factor_graph
import collections
import itertools as it
import random
import collections
import csv
import timeit
import time

class maxplus():

    def __init__(self, regular_factor, agent_neighbour_combo, max_iter):#, factored_agent_type, modelnr):
        self.regular_factor = regular_factor
        self.factored_agent = agent_neighbour_combo
        self.qval_combo_list = self.qval_combo_initialiser()
        # factored_agent should be an agent(vertex) as keys and value as a list of neighbour: THIS IS DONEEEEEEEE!!!!!!!
        self.num_actions = 2
        #self.initialise_msg()
        #self.payoff_initialiser()
        self.max_iter = max_iter
        #print(self.max_iter)
        self.factored_action_list = [i for i in it.product((0,1), repeat = 2)]
        #self.action_payoff_tup = []
        self.alpha = 0.5
        self.threshold = 0.5

    def initialise_again(self):
        self.initialise_msg()
        self.payoff_initialiser()
        self.action_payoff_tup = [] 

    def initialise_sent_dict(self):
        for keys in self.factored_agent.keys():
            for c in self.factored_agent[keys]:
                self.sent_and_received[keys][c] = False
     
    def qval_combo_initialiser(self):
        q_val_combination = []
        for keys in self.regular_factor.keys():
            q_val_combination.append(list(self.regular_factor[keys]))
        return q_val_combination

    def initialise_msg(self):
        self.sen_msg_bool = {}
        # sen_msg_bool is to determine the fixed point has arrived yet or not.
        self.sen_msg = {}
        self.difference_dict = {}
        self.action_payoff= {}
        self.sent_and_received = {}
        for keys in self.factored_agent.keys():
            self.sen_msg[keys] = {}
            self.sen_msg_bool[keys] = {}
            self.difference_dict[keys] = {}
            self.sent_and_received[keys] = {}
            for c in self.factored_agent[keys]:
                self.sen_msg[keys][c] = {}
                self.sen_msg_bool[keys][c] = {}
                self.difference_dict[keys][c] = {}
                self.sent_and_received[keys][c] = False
                for i in range(self.num_actions):
                    self.sen_msg[keys][c][str(i)] = [np.random.normal(0,1)]
                    self.sen_msg_bool[keys][c][str(i)] = [True]
                    self.difference_dict[keys][c][str(i)] = []
 
    def difference_dict_print(self):
        diff = []
        for keys in self.difference_dict.keys():
            for key in self.difference_dict[keys].keys():
                for c in self.difference_dict[keys][key].keys():
                    #print(self.difference_dict[keys][key][c][-1])
                    diff.append(self.difference_dict[keys][key][c][-1])
        #print(np.mean(diff))
        return np.mean(diff), np.std(diff)

    '''Initialise msg initialises the message dictionary as sent message.
    with keys as the neighbouring agents and values are stored inside a list. For example:
    sen_msg ={'0'= {'1':{'0':[], '1':[]}, '2':{'0':[], '1':[]}}, 
              '1'= {'0':{'0':[], '1':[]}, '2':{'0':[], '1':[]}},
              '2' = {'1':{'0':[], '1':[]}, '0':{'0':[], '1':[]}}}'''

    '''Payoff is a dict== {agentid: {action_0:[], action_1:[] }}
       Here the action represents the action of agent represented by agentid.'''

    def payoff_initialiser(self):
        self.payoff = {}
        self.best_action = {}
        for keys in self.factored_agent.keys():
            self.payoff[keys] = {}
            self.best_action[keys] = [] 
            for i in range(self.num_actions):
                self.payoff[keys][str(i)] = []

    def random_agent_selector(self):
        sample = random.sample(list(self.factored_agent.keys()), len(self.factored_agent.keys()))
        return sample

    def normaliser(self, agent_sending, agent_receiving):
        normalising_value = []
        for i in range(self.num_actions):
            normalising_value.append(self.sen_msg[agent_sending][agent_receiving][str(i)][-1])
        return normalising_value
 
    def connected_nodes(self, agentid):
        return list(self.factored_agent[agentid]) 

    def dict_to_array(self, dictionary:dict):
        lst = []
        for keys in dictionary.keys():
            lst.append(dictionary[keys])
        return np.asarray(lst)

    '''From here on we define the methods that we make use after the initialisation and after running the first iteration'''         

    def factored_q_val_extracter(self, agentid, one_left_neighbour, q_val):
        '''From the Q_value_dictionary it returns the q_value as an array for shape(1, action_per_agent*num_actions), in my case its (1,4)'''
        lst = list((agentid, one_left_neighbour))
        for val in self.qval_combo_list:
            # actual_combo = list((int(keys[1]), int(keys[4]))) 
            if collections.Counter(lst) == collections.Counter(val):
                qval = q_val[str(val)]
                break
        return qval
    
    def extract_order(self, agentid, one_left_neighbour):
        '''one left neighbour indicates a single neighbour from the list of neighbouring from which it receives 
           message in order to send it to other neighbour which is not included in the list'''
        bool_val = False
        for val in self.qval_combo_list:
            if list((agentid, one_left_neighbour))== val:
                bool_val = True
                break
        return bool_val

    def max_outgoing_cal(self, reduced_q_arr, incoming_msg_dict):
        lst = []
        try:
            for keys in incoming_msg_dict.keys():
                lst.append(incoming_msg_dict[keys])
        except:
            pdb.set_trace()
            pass

        try:
            msg_action_val = np.asarray(reduced_q_arr) + np.asarray(lst[0]) + np.asarray(lst[1]) + np.asarray(lst[2])
        except:
            try:
                msg_action_val = np.asarray(reduced_q_arr) + np.asarray(lst[0]) + np.asarray(lst[1]) 
            except:
                try:
                    msg_action_val = np.asarray(reduced_q_arr) + np.asarray(lst[0])
                except:
                    msg_action_val = np.asarray(reduced_q_arr)
         
        index = np.argmax(msg_action_val)
        return msg_action_val[index], index

    def q_array_reducer(self, specific_q_val, bool_val, action: str):
        '''returns the q_value as a list containing only the relevant q_val'''
        q_arr = []
        if bool_val == True:
            if action == str(0):
                q_arr.append(specific_q_val[0][0])
                q_arr.append(specific_q_val[0][2])
                #print("Selecting q_val for ", action)
            else:
                q_arr.append(specific_q_val[0][1])
                q_arr.append(specific_q_val[0][3])
        else:
            if action == str(0):
                q_arr.append(specific_q_val[0][0])
                q_arr.append(specific_q_val[0][1])
                #print("Selecting q_val for ", action)
            else:
                q_arr.append(specific_q_val[0][2])
                q_arr.append(specific_q_val[0][3])
        return q_arr

    def agent_incoming_msg(self, agent_receiving, sending_neighbours: list, exclude: bool, exc_agent = None):
        '''returns a dict of incoming messages with the actions for the given agentid.
           rec_message= {neighbour1:{action0: [], action1: []}, neighbour2:{action0:[], action1:[]}}'''
        rec_message = {}
        if exclude:
                sending_neighbours.remove(exc_agent)
        for val in sending_neighbours:
            for keys in self.sen_msg[val].keys():
                if agent_receiving==keys:
                    rec_message[val] = self.sen_msg[val][keys]
                    break
        return rec_message

    def msg_last_action_dict(self, agentid, left_neighbour: list):
        '''returns a dictionary of agent_neighbour as keys with list as values containing last message for each of the action
           This is used for input to the mu_ij calculator for max_ai
           msg_dict = {neighbour1:[action 0 last message value, action1 last message value] , neighbour2: [action 0 last message value, action1 last message value]}'''
        # uses agent_incoming_message() in order to get the incoming message for the particular agentid (without exclusion).
        incoming_message = self.agent_incoming_msg(agent_receiving=agentid, sending_neighbours=left_neighbour, exclude = False, exc_agent = None)
        message_dict = {}
        for keys in incoming_message.keys():
            message_dict[keys] = []
            message_dict[keys].append(incoming_message[keys]['0'][-1])
            message_dict[keys].append(incoming_message[keys]['1'][-1])
        return message_dict

    def intsum_incoming_messges(self, agent_number):
        '''Sums all the incoming messages for a particular agent i'''
        ''' Can be used when we are initialising the messages for the first time using predefined normal noise for the message'''
        connected_agents = self.connected_nodes(agent_number)
        rec_message = self.agent_incoming_msg(agent_receiving=agent_number, sending_neighbours=connected_agents, exclude=False, exc_agent=None)
        sum_action_array = self.sum_array_initialiser()
        for keys in rec_message.keys():
            for key in rec_message[keys].keys():
                sum_action_array[key].append(rec_message[keys][key][-1])
        msg_dict = self.action_value_summer(sum_action_array)
        return msg_dict 

    def action_value_summer(self, msg:dict):
        msg_array = {}
        for keys in msg.keys():
            msg_array[keys] = np.sum(msg[keys])
        return msg_array

    def sum_array_initialiser(self):
        sum_arr = {}
        for i in range(self.num_actions):
            sum_arr[str(i)] = []
        return sum_arr

    def msg_dict_update(self, agentid, neighbour, action, message):
        ''' Here use the update rule for the message as told by Frans and also return the difference between the last and the current value'''
        last_msg = self.sen_msg[agentid][neighbour][action][-1]
        normaliser = self.normaliser(agentid, neighbour)
        new_msg = message - np.sum(normaliser)/2
        new_msg = self.alpha*(last_msg) + (1-self.alpha)*new_msg
        msg_diff = new_msg - last_msg
        if msg_diff >= self.threshold:
             self.sen_msg_bool[agentid][neighbour][action].append(False)
        else:
             self.sen_msg_bool[agentid][neighbour][action].append(True)    
        self.sen_msg[agentid][neighbour][action].append(new_msg)
        self.difference_dict[agentid][neighbour][action].append(msg_diff)
   
    def send_outgoing_msg(self, agentsending, agent_receiving, message_dict, q_val):
        bool_val = self.extract_order(agentsending, agent_receiving) # to check if the configuration of the factored agents is the same as agent and its neighbour
        q_arr = self.factored_q_val_extracter(agentsending, agent_receiving, q_val) #This is an array of shape (1,4).    
        #for keys in message_dict.keys():
        for action in range(self.num_actions):
            action = str(action)
            # this is the action of the agent receiving the message
            #print('Working on Action', action, 'for agent', agentsending, 'for its neighbour', keys)
            q_val = self.q_array_reducer(q_arr, bool_val, action)
            # q_val is a list: [1,2]
            # normliser is a list, sum all the values in the list and divide by 2 and then substract from each action.
            message, index = self.max_outgoing_cal(q_val, message_dict)
            self.msg_dict_update(agentsending, agent_receiving, action, message)
            '''message is the single maximum value. Index can also be treated as the string value of action of 
               agentid(agent 'i' the one which sends message to 'j').'''

    def one_neighbour_outgoing_msg(self, agentsending, agent_receiving, q_val):
        bool_val = self.extract_order(agentsending, agent_receiving) # to check if the configuration of the factored agents is the same as agent and its neighbour
        q_arr = self.factored_q_val_extracter(agentsending, agent_receiving, q_val)
        for action in range(self.num_actions):
                action = str(action)
                # this is the action of the agent receiving the message
                #print('Working on Action', action, 'for agent', agentsending, 'for its neighbour', keys)
                q_val = self.q_array_reducer(q_arr, bool_val, action)
                # q_val is a list: [1,2]
                # normliser is a list, sum all the values in the list and divide by 2 and then substract from each action.
                message, index = self.max_outgoing_cal(q_val, None)
                self.msg_dict_update(agentsending, agent_receiving, action, message)

    def payoff_update(self, agentid):
        '''compute global payoff and append on the list and make a variable which contains the best joint action found so far which yields the maximum
           value of global payoff'''
        summed_message_dict = self.intsum_incoming_messges(agentid)
        summed_message_list = []
        for keys in summed_message_dict.keys():
            self.payoff[agentid][keys].append(summed_message_dict[keys])
            summed_message_list.append(summed_message_dict[keys])       
        best_action = np.argmax(summed_message_list)
        for keys in summed_message_dict.keys():
            val = summed_message_dict[keys]
            if val == summed_message_list[best_action]:
                best_action = keys
                break
        best_action = str(best_action)
        self.best_action[agentid].append(best_action)

    def get_best_joint_action(self):
        best_joint_action = {}
        for keys in self.best_action.keys():
             best_joint_action[keys] = self.best_action[keys][-1]
        return best_joint_action

    def global_payoff_cal(self, q_val):
        '''Calculates the global payoff u(a) = Summ(f_ij), for the best joint action found so far.''' 
        best_joint_action = self.get_best_joint_action()
        global_payoff = 0
        for keys in self.regular_factor.keys():
            action_tup = []
            q_ij = self.regular_factor[keys]
            qvalue = q_val[str(q_ij)] # this is a (1,4) shaped q value.
            for val in q_ij:
                action_tup.append(int(best_joint_action[val]))
            action_tup = tuple(action_tup)
            for key, val in enumerate(self.factored_action_list):
                #******************CHANGE FOR 3 FACTORED AGENTS*********************#
                if action_tup==val:
                    index = key
                    break
            global_payoff += qvalue[0][index] 
        return global_payoff, best_joint_action

    def action_payoff_tup_to_sumo_action(self, action_pay_tup):
        action = action_pay_tup
        sumo_act = collections.OrderedDict()
        for keys in action.keys():
            sumo_act[str(keys)] = int(action[keys])
        return sumo_act

    def checker(self, agentsending, agentreceiving):
        val = self.sent_and_received[agentsending][agentreceiving]
        return val

    def max_plus_calculator(self, q_val):
        #agent_schedule = self.random_agent_selector()  #[1, 3, 0, 2] #self.random_agent_selector()
        #agent_schedule = [3,1,2,4,5,0]
        #fixed_point = False
        for i in range(self.max_iter):
            self.initialise_sent_dict()
            agent_schedule = self.random_agent_selector()
            for val in agent_schedule:
                agent_neighbour = self.connected_nodes(val)
                for neighbour in agent_neighbour:
                    #print(neighbour, ' selected from ',agent_neighbour
                    left_neighbour = [i for i in agent_neighbour if i!=neighbour]
                    if len(left_neighbour)==0:
                       if self.checker(val, neighbour)==False:
                        pass
                       self.one_neighbour_outgoing_msg(val, neighbour, q_val)
                       self.sent_and_received[val][neighbour]= True
                       break
                    if self.checker(val, neighbour)==False:
                        pass
                    else:
                        continue
                    message_dict = self.msg_last_action_dict(val, left_neighbour)
                    #print("Received incoming message from left neighbours: ", message_dict) 
                    self.send_outgoing_msg(agentsending=val, agent_receiving=neighbour, message_dict=message_dict, q_val=q_val)
                    self.sent_and_received[val][neighbour]= True
                self.payoff_update(val)
            glo_pay, joint_action = self.global_payoff_cal(q_val)
            if i ==0:
                self.action_payoff['payoff'] = glo_pay
                self.action_payoff['action'] = joint_action
            else:
                if glo_pay > self.action_payoff['payoff']:
                    self.action_payoff['payoff'] = glo_pay
                    self.action_payoff['action'] = joint_action
            #self.action_payoff_tup.append(tuple((glo_pay, joint_action)))
        #self.lstprinter()
        #diff, std = self.difference_dict_print()
        #print(self.sen_msg)
        #print(self.action_payoff_tup)
        #payoff, action  = self.final_action()
        #print(payoff, action)
        return self.action_payoff['payoff'], self.action_payoff_tup_to_sumo_action(self.action_payoff['action'])
        


    def lstprinter(self):
        for keys in self.sen_msg.keys():
            for key in self.sen_msg[keys].keys():
                for k in self.sen_msg[keys][key].keys():
                    print(len(self.sen_msg[keys][key][k]))

    def final_action(self):
        payoff = self.action_payoff_tup[0][0]
        action = self.action_payoff_tup[0][1]
        for i in range(len(self.action_payoff_tup)):
            if payoff < self.action_payoff_tup[i][0]:
                payoff = self.action_payoff_tup[i][0]
                action = self.action_payoff_tup[i][1]
        return payoff, action

if __name__=="__main__":

    ''' Fix factor_agent in the main input.'''
    #    *****FIXED*******
    ''' Fix q_val such that the keys are represented as the combination of the agents and values are arrays of shape(1,4)'''
    #    ****** FIXED *****
    ''' Change the name of the factored agent to the name of the agent and its neighbours'''
    #   ****** FIXED ********


    '''agent_neighbour_combo = {0: [1,3],
                            1: [0, 2, 4],
                            2: [1, 5],
                            3: [0, 4],
                            4: [1, 3, 5],
                            5: [2, 4]}


    '''
    '''agent_neighbour_combo= {0: [1,4],
                            1: [0, 2, 5],
                            2: [1, 3, 6],
                            3: [2, 7],
                            4: [0, 5],
                            5: [1, 4, 6],
                            6: [2, 5, 7],
                            7: [3, 6]}

    regular_factor = {"0": [0, 1],
                      "1": [1, 2],
                      "2": [2, 3],
                      "3": [4, 5],
                      "4": [5, 6],
                      "5": [6, 7],
                      "6": [0, 4],
                      "7": [1, 5],
                      "8": [2, 6],
                      "9": [3, 7]}'''


   
