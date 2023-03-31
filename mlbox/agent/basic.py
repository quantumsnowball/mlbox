from inspect import currentframe
from pathlib import Path
from typing import Literal, Self

import numpy as np
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

    #
    # acting
    #

    @override
    def explore(self) -> T_Action:
        random_action = self.env.action_space.sample()
        return random_action

    @override
    def decide(self,
               obs: T_Obs,
               *,
               epsilon: float = 0.5) -> T_Action:
        if np.random.random() > epsilon:
            return self.explore()
        else:
            return self.exploit(obs)

    #
    # I/O
    #

    @override
    def prompt(self,
               name: str,
               *,
               start_training: bool = False) -> None:
        # prepare caller info
        frame = currentframe()
        caller_frame = frame.f_back if frame else None
        globals = caller_frame.f_globals if caller_frame else None
        script_path = Path(globals['__file__']) if globals else Path()
        base_dir = Path(script_path).parent.relative_to(Path.cwd())
        path = base_dir / name
        if path.is_file():
            if input(f'Model {path} exists, load? (y/[n]) ').upper() == 'Y':
                # load agent
                self.load(path)
        if start_training or input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train()
            if input(f'Save model? [y]/n) ').upper() != 'N':
                self.save(path)
