import gym
import logging
from gym import spaces
import os
from aienvs.Sumo.LDM import ldm
from aienvs.Sumo.SumoHelper import SumoHelper
from aienvs.Sumo.State_representation import *
import time
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import random
from aienvs.Sumo.SumoHelper import SumoHelper
from aienvs.Environment import Env
import copy
import time
from aienvs.Sumo.TrafficLightPhases import TrafficLightPhases
import yaml
from aienvs.Sumo.statics_control import *


class SumoGymAdapter(Env):
    """
    An adapter that makes Sumo behave as a proper Gym environment.
    At top level, the actionspace and percepts are in a Dict with the
    trafficPHASES as keys.
    
    @param maxConnectRetries the max number of retries to connect. 
        A retry is needed if the randomly chosen port 
        to connect to SUMO is already in use. 
    """
    _DEFAULT_PARAMETERS = {'gui':True,  # gui or not
                'scene':'four_grid',  # subdirectory in the aienvs/scenarios/Sumo directory where 
                'tlphasesfile':'cross.net.xml',  # file 
                'box_bottom_corner':(0, 0),  # bottom left corner of the observable frame
                'box_top_corner':(10, 10),  # top right corner of the observable frame
                'resolutionInPixelsPerMeterX': 1,  # for the observable frame
                'resolutionInPixelsPerMeterY': 1,  # for the observable frame
                'y_t': 6,  # yellow time
                'car_pr': 0.5,  # for automatic route/config generation probability that a car appears
                'car_tm': 2,  #  for automatic route/config generation when the first car appears?
                'route_starts' : [],  #  for automatic route/config generation, ask Rolf 
                'route_min_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_max_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_ends' : [],  #  for automatic route/config generation, ask Rolf
                'generate_conf' : True,  # for automatic route/config generation
                'libsumo' : False,  # whether libsumo is used instead of traci
                'waiting_penalty' : 1,  # penalty for waiting
                'new_reward': False,  # some other type of reward ask Miguel
                'lightPositions' : {},  # specify traffic light positions
                'scaling_factor' : 1.0,  # for rescaling the reward? ask Miguel
                'maxConnectRetries':50,  # maximum reattempts to connect by Traci
                }

    def __init__(self, parameters:dict={}):
        """
        @param path where results go, like "Experiment ID"
        @param parameters the configuration parameters.
        gui: whether we show a GUI. 
        scenario: the path to the scenario to use
        """
        logging.debug(parameters)
        self._parameters = copy.deepcopy(self._DEFAULT_PARAMETERS)
        self._parameters.update(parameters)

        dirname = os.path.dirname(__file__)
        tlPhasesFile = os.path.join(dirname, "../../scenarios/Sumo/", self._parameters['scene'], self._parameters['tlphasesfile'])
        self._tlphases = TrafficLightPhases(tlPhasesFile)
        self.ldm = ldm(using_libsumo=self._parameters['libsumo'])
        
        self._takenActions = {}
        self._yellowTimer = {}
        self._chosen_action = None
        self.seed(42)  # in case no seed is given
        self._action_space = self._getActionSpace()
        self.stats_control = Control(self._parameters['tripinfofolder'])
        self.factor_graph = self._parameters['factored_agents']
        self.n_factors = len(list(self.factor_graph.keys()))
        self.factored_coords = self._parameters['factored_coords']
        #self.pixelsPerMeter = self._parameters['pixelsPerMeter']
        self.testseed = list(self._parameters['test_seed'])
        self.seed_cntr = 0
    
    def step(self, actions:dict):
        self._set_lights(actions)
        self.ldm.step()
        obs = self._observe()
        done = self.ldm.isSimulationFinished()
        global_reward = self._computeGlobalReward()
        '''self.action_switches(actions)
        actual_reward = self.actual_global_reward(global_reward)'''       

        # as in openai gym, last one is the info list
        return obs, global_reward, done, []

    def reset_test_cntr(self):
        self.seed_cntr = 0

    '''def actual_global_reward(self, global_reward):
        global_reward['result'] += -0.1*self._action_switches['0']
        return global_reward

    def action_switches(self, actions:spaces.Dict):
        self._action_switches = {}
        for intersectionId in actions.keys():
            if len(self._takenActions[intersectionId])==1:
                self._action_switches[intersectionId] = 0    
            else:
                prev_action = self._takenActions[intersectionId][-1]
                if prev_action != self._intToPhaseString(intersectionId, actions.get(intersectionId)):          
                    self._action_switches[intersectionId] = 1
                else:
                    self._action_switches[intersectionId] = 0    
        return self._action_switches'''

    def reset(self, episode=None):
        try:
            logging.debug("LDM closed by resetting")
            self.ldm.close()
        except:
            logging.debug("No LDM to close. Perhaps it's the first instance of training")

        if episode!=None:
            average_travel_times, average_travel_time = self.stats_control.log()
            logging.info("Starting SUMO environment...")
            self._startSUMO()
            # TODO: Wouter: make state configurable ("state factory")
            self._state = FactoredLDMMatrixState(self.ldm, [self._parameters['box_bottom_corner'], self._parameters['box_top_corner']], factored_agents=self.factor_graph, factored_coords=self.factored_coords)
            return self._observe(), average_travel_times, average_travel_time

        else:
            logging.info("Starting SUMO environment...")
            self._startSUMO()
            # TODO: Wouter: make state configurable ("state factory")
            self._state = FactoredLDMMatrixState(self.ldm, [self._parameters['box_bottom_corner'], self._parameters['box_top_corner']], factored_agents=self.factor_graph, factored_coords=self.factored_coords)

            return self._observe()
        
        # TODO: change the defaults to something sensible
    def render(self, delay=0.0):
        import colorama
        colorama.init()

        def move_cursor(x, y):
            print ("\x1b[{};{}H".format(y + 1, x + 1))

        def clear():
            print ("\x1b[2J")

        clear()
        move_cursor(100, 100)
        import numpy as np
        np.set_printoptions(linewidth=100)
        print(self._observe())
        time.sleep(delay)
    
    def seed(self, seed):
        random.seed(seed)
        self._seed = int(time.time())

    def close(self):
        self.__del__()

    # TODO: Wouter: this needs to return a space and be somehow unified with gym.spaces
    @property
    def observation_space(self):
        return self._state.update_state()

    @property
    def action_space(self):
        return self._action_space

    ########## Private functions ##########################
    def __del__(self):
        logging.debug("LDM closed by destructor")
        if 'ldm' in locals():
            self.ldm.close()

    def _startSUMO(self):
        """
        Start the connection with SUMO as a subprocess and initialize
        the traci port, generate route file.
        """
        val = 'sumo-gui' if self._parameters['gui'] else 'sumo'
        maxRetries = self._parameters['maxConnectRetries']
        sumo_binary = checkBinary(val)
        self.dirname = os.path.dirname(__file__)
        outfile = self._parameters['tripinfofolder']
        self.out = os.path.join(*[self.dirname, "../../test/Stats", outfile, "tripinfo.xml"])

        # Try repeatedly to connect
        while True:
            try:
                # this cannot be seeded
                self._port = random.SystemRandom().choice(list(range(10000, 20000)))
                if self._parameters['test']==False:
                    self._seed = self._seed + random.randint(0, 276574)
                else:
                    try:
                        self._seed = self.testseed[self.seed_cntr]
                    except:
                        self._seed = self._seed + random.randint(0, 276574)
                    self.seed_cntr +=1
                self._sumo_helper = SumoHelper(self._parameters, self._port, int(self._seed))
                conf_file = self._sumo_helper.sumocfg_file
                logging.info("Configuration: " + str(conf_file))
                sumoCmd = [sumo_binary, "-c", conf_file, "--tripinfo-output", self.out, "--seed", str(self._seed)]
                self.ldm.start(sumoCmd, self._port)
            except Exception as e:
                if str(e) == "connection closed by SUMO" and maxRetries > 0:
                    maxRetries = maxRetries - 1
                    continue
                else:
                    raise
            else:
                break

        self.ldm.init(waitingPenalty=self._parameters['waiting_penalty'], new_reward=self._parameters['new_reward'])  # ignore reward for now
        # used to set boundaries to compute the network space and it computes the states, i can use this to compute states based on the fatored graphs.
        self.ldm.setResolutionInPixelsPerMeter(self._parameters['resolutionInPixelsPerMeterX'], self._parameters['resolutionInPixelsPerMeterY'])
        self.ldm.setPositionOfTrafficLights(self._parameters['lightPositions'])

        if list(self.ldm.getTrafficLights()) != self._tlphases.getIntersectionIds():
            raise Exception("environment traffic lights do not match those in the tlphasesfile " 
                    +self._parameters['tlphasesfile'] + str(self.ldm.getTrafficLights())
                    +str(self._tlphases.getIntersectionIds()))
            
    def _intToPhaseString(self, intersectionId:str, lightPhaseId: int):
        """
        @param intersectionid the intersection(light) id
        @param lightvalue the PHASES value
        @return the intersection PHASES string eg 'rrGr' or 'GGrG'
        """
        logging.debug("lightPhaseId" + str(lightPhaseId))
        return self._tlphases.getPhase(intersectionId, lightPhaseId)
                
    def _observe(self): 
        """
        Fetches the Sumo state and converts in a proper gym observation.
        The keys of the dict are the intersection IDs (roughly, the trafficLights)
        The values are the state of the TLs
        """
        return self._state.update_state()

    def _computeGlobalReward(self):
        """
        Computes the global reward
        """
        return self._state.update_reward()
    
    def _getActionSpace(self):
        """
        @returns the actionspace: a dict containing <id,phases> where 
        id is the intersection id and value is
         all possible actions for each id as specified in tlphases
        """
        return spaces.Dict({inters:spaces.Discrete(self._tlphases.getNrPhases(inters)) \
                            for inters in self._tlphases.getIntersectionIds()})

    def _set_lights(self, actions:spaces.Dict):
        """
        Take the specified actions in the environment
        @param actions a list of
        """
        for intersectionId in actions.keys():
            action = self._intToPhaseString(intersectionId, actions.get(intersectionId))
            # Retrieve the action that was taken the previous step
            try:
                prev_action = self._takenActions[intersectionId][-1]
            except KeyError:
                # If KeyError, this is the first time any action was taken for this intersection
                prev_action = action
                self._takenActions.update({intersectionId:[]})
                self._yellowTimer.update({intersectionId:0})

            # Check if the given action is different from the previous action
            if prev_action != action:
                # Either the this is a true switch or coming grom yellow
                action, self._yellowTimer[intersectionId] = self._correct_action(prev_action, action, self._yellowTimer[intersectionId])

            # Set traffic lights
            self.ldm.setRedYellowGreenState(intersectionId, action)
            self._takenActions[intersectionId].append(action)

    def _correct_action(self, prev_action, action, timer):
    
        """
        Check what we are going to do with the given action based on the
        previous action.
        """
        # Check if the agent was in a yellow state the previous step
        if 'y' in prev_action:
            # Check if this agent is in the middle of its yellow state
            if timer > 0:
                new_action = prev_action
                timer -= 1
            # Otherwise we can get out of the yellow state
            else:
                new_action = self._chosen_action
                if not isinstance(new_action, str):
                    raise Exception("chosen action is illegal")
        # We are switching from green to red, initialize the yellow state
        else:
            self._chosen_action = action
            if self._parameters['y_t'] > 0:
                new_action = prev_action.replace('G', 'y')
                timer = self._parameters['y_t'] - 1
            else:
                new_action = action
                timer = 0

        return new_action, timer

