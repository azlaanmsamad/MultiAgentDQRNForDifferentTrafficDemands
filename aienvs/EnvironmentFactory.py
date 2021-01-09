from aienvs.Environment import Env


def createEnvironment(fullname:str, parameters:dict) -> Env:
    '''
    Create a gym Env from a given full path name
    @param fullname the full.path.name to the agent to create, eg
    "aienvs.FactoryFloor.FactoryFloor"
    @param general parameters for the environment initialization
    @return an initialized Env
    '''
    klass = classForNameTyped(fullname, Env)
    obj = klass(parameters)
    return obj


def classForNameTyped(klsname:str, expectedkls):
    """
    @param klsname the string full path to the class to load. 
    Eg "aiagents.single.RandomAgent.RandomAgent".
    The class to load has to be on the classpath.
    @param expectedkls the expected class, eg AgentComponent
    @return a class object that is subclass of expectedkls. You can make instances of this class object 
    by calling it with the constructor arguments.
    @raise exception if klsname does not contain expected class or subclass of it. 
    """
    klass = classForName(klsname)
    if not issubclass(klass, expectedkls):
        raise Exception("Class " + klsname + " does not extend " + str(expectedkls))
    return klass


def classForName(kls:str):
    """
    @param kls the string full path to the class to load. 
    Eg "aiagents.single.RandomAgent.RandomAgent".
    The class to load has to be on the classpath.
    @return a class object. You can make instances of this class object 
    by calling it with the constructor arguments.
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

