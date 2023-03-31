from typing import Literal, Self

from gymnasium import Env
from typing_extensions import override

from mlbox.agent import Agent
from mlbox.trenv.queue import TerminatedError
from mlbox.types import T_Action, T_Obs


class BasicAgent(Agent[T_Obs, T_Action]):
    '''
    Implement common props of an Agent
    '''

    device: Literal['cuda', 'cpu', ]

    def __new__(cls) -> Self:
        try:
            # ensure attrs are implemented in subclass instance
            cls.device
            return super().__new__(cls)
        except AttributeError as e:
            raise NotImplementedError(e.name) from None

    def __init__(self) -> None:
        super().__init__()
        self.progress = 0.0

    #
    # env
    #

    @property
    def env(self) -> Env[T_Obs, T_Action]:
        ''' a gym.Env compatible object '''
        try:
            return self._env
        except AttributeError:
            raise NotImplementedError('env') from None

    @env.setter
    def env(self, env: Env[T_Obs, T_Action]) -> None:
        self._env = env

    #
    # training
    #

    @override
    def play(self,
             max_step: int,
             *,
             env: Env[T_Obs, T_Action] | None = None) -> float:
        # select env
        env = env if env is not None else self.env
        # reset to a new environment
        obs, *_ = env.reset()
        # run the env
        total_reward = 0.0
        for _ in range(max_step):
            action = self.exploit(obs)
            try:
                next_obs, reward, terminated, *_ = env.step(action)
            except TerminatedError:
                break
            obs = next_obs
            total_reward += float(reward)
            if terminated:
                break
        return total_reward
