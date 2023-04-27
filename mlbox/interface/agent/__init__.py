from mlbox.interface.agent.acting import Acting
from mlbox.interface.agent.env import Environment
from mlbox.interface.agent.io import IO
from mlbox.interface.agent.network import Network
from mlbox.interface.agent.tensorboard import Tensorboard
from mlbox.interface.agent.training import Training
from mlbox.types import T_Action, T_Obs


class Agent(
    Environment[T_Obs, T_Action],
    Acting[T_Obs, T_Action],
    Training[T_Obs, T_Action],
    IO[T_Obs, T_Action],
    Tensorboard[T_Obs, T_Action],
    Network[T_Obs, T_Action],
):
    '''
    Define the interface of an Agent

    To create an concrete agent that is runnable, the concrete class together
    with all its sub-classes must provide implementation to all of the above 
    interfaces.
    '''
    ...
