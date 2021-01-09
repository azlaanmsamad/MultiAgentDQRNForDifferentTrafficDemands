from aienvs.Env import Env

class Simulator(Env):
    """
    to be used as base class only
    """
    @property
    def __init__(self, environment: Env):
        super().__init__(Env)


    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state


