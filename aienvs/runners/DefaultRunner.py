from aienvs.listener.Listenable import Listenable


class DefaultRunner(Listenable):
    """
    Runners should notify listeners about
    the current #steps and reward when a step is taken, through a data object of the form  
    {"steps",Nsteps, "reward": Reward} 

    """
    
