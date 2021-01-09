from aienvs.Environment import Env
from gym import spaces
from aienvs.runners.Episode import Episode
from aienvs.runners.DefaultRunner import DefaultRunner
from aienvs.listener.DefaultListenable import DefaultListenable
from aienvs.listener.Listener import Listener
from time import time
from itertools import cycle
import logging
from scipy import stats

class Experiment(DefaultRunner, DefaultListenable, Listener):
    """
    Contains all info to run an experiment

    """

    def __init__(self, agent, env: Env, maxSteps:int, seedlist=None, render:bool=False, renderDelay=0):
        """
        @param agent an AgentComponent holding an agent
        @param env the openai gym Env that we are running in
        @param render True iff environment must be rendered each step.
        """
        super().__init__()
        self._agent = agent
        self._env = env
        self._maxSteps = maxSteps
        self._render = render
        self._renderDelay = 0

        if seedlist is not None:
            self._seedcycle=cycle(seedlist)
        else:
            self._seedcycle=None

    def _getSeed(self):
        if self._seedcycle is not None:
            return next(self._seedcycle)
        else:
            return int(time()) 
        
    def run(self):
        """
        Resets env. Loop env.step and agent.step() until number of steps have been made.
        If an env is done before the number of steps have been reached, the env is reset.
        @return the total reward divided by the total number of episodes 
        """
        steps = 0
        episodeCount = 0
        totalReward = 0

        episodeRewards = []
    
        while steps < self._maxSteps:
            self._env.seed(self._getSeed())
            obs = self._env.reset()
            episode = Episode(self._agent, self._env, obs, self._render, self._renderDelay)
            episode.addListener(self)
            episodeSteps, episodeReward = episode.run()
            logging.info("New episode")
            steps += episodeSteps
            totalReward += episodeReward
            episodeRewards.append(episodeReward)
            logging.info("Episode reward: " + str(episodeReward))
            episodeCount += 1
    
        try:    
            return stats.describe(episodeRewards), stats.bayes_mvs(episodeRewards)
        except ValueError as err:
            print(err)

    def notifyChange(self, data):
        self.notifyAll(data)
    
