from aienvs.Environment import Env
from gym import spaces
import yaml
from aienvs.runners.DefaultRunner import DefaultRunner
from aienvs.listener.DefaultListenable import DefaultListenable


class Episode(DefaultRunner, DefaultListenable):
    """
    Contains all info to run an episode (single run till environment is done)
    """

    def __init__(self, agent, env: Env, initialObs, render:bool=False, renderDelay=0):
        """
        @param agent an AgentComponent holding an agent
        @param env the openai gym Env that we are running in
        @param firstActions the actions for the first step to be taken by agent
        @param render True iff environment must be rendered each step.
        """
        super().__init__()
        self._agent = agent
        self._env = env
        self._initialObs = initialObs
        self._render = render
        self._renderDelay = renderDelay

    def step(self, obs, globalReward, done):
        """
        One step of the RL loop
        """

        actions = self._agent.step(obs, globalReward, done)
        self.notifyAll({'actions':actions, 'observation': obs, 'reward':globalReward, 'done':done})

        obs, globalReward, done, info = self._env.step(actions)
        return obs, globalReward, done

    def run(self):
        """
        Loop env.step and agent.step() until env is done.
        @return the number of steps it took to reach done state, total reward
        """
        done = False
        steps = 0
        globalReward = 0
        obs = self._initialObs
        totalReward = 0
    
        while True:
            steps += 1
            obs, globalReward, done = self.step(obs, globalReward, done)
            totalReward += globalReward

            if done:
                break

            if self._render:
                self._env.render(self._renderDelay)

        return steps, totalReward
