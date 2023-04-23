from mlbox.agent.a2c import A2CAgent
from mlbox.agent.a2c.discrete.props import Props
from mlbox.types import T_Action, T_Obs


class A2CDiscreteAgent(Props[T_Obs, T_Action],
                       A2CAgent[T_Obs, T_Action]):
    def __init__(self) -> None:
        super().__init__()
