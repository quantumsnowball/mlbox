from collections import deque
from inspect import currentframe
from pathlib import Path
from typing import Self

import torch as T
from gymnasium import Env
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import override

from mlbox.agent import Agent
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class BasicAgent(Agent[T_Obs, T_Action]):
    '''
    Implement common props of an Agent
    '''

    device: T.device

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
        if self.tensorboard:
            self.writer = SummaryWriter(self.tensorboard_logdir)

    #
    # training
    #

    max_step = 1000

    @override
    def play(self,
             max_step: int | None = None,
             *,
             env: Env[T_Obs, T_Action] | None = None) -> float:
        # determine max step
        max_step = max_step if max_step is not None else self.max_step
        # select env
        env = env if env is not None else self.env
        # reset to a new environment
        obs, *_ = env.reset()
        # run the env
        total_reward = 0.0
        for _ in range(max_step):
            action = self.decide(obs)
            try:
                next_obs, reward, terminated, *_ = env.step(action)
            except TerminatedError:
                break
            obs = next_obs
            total_reward += float(reward)
            if terminated:
                break
        return total_reward

    rolling_reward_ma = 5

    def reset_rolling_reward(self) -> None:
        self.rolling_reward = deque[float](maxlen=self.rolling_reward_ma)

    #
    # progress report
    #

    print_hash_every = 10

    def print_progress_bar(self,
                           i: int,
                           *,
                           chr: str = '#') -> None:
        if i % self.print_hash_every == 0:
            print(chr, end='', flush=True)

    report_progress_every = 10
    mean_reward_display_format = '+.1f'

    def print_validation_result(self,
                                i: int,
                                ) -> None:
        if i % self.report_progress_every == 0:
            self.rolling_reward.append(self.play())
            mean_reward = sum(self.rolling_reward)/len(self.rolling_reward)
            print(f' | Episode {i:>4d} | {mean_reward=:{self.mean_reward_display_format}}')

    render_every: int | None = None

    def render_showcase(self,
                        i: int) -> None:
        if self.render_every is not None and i % self.render_every == 0:
            self.play(env=self.render_env)

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

    #
    # tensorboard
    #

    tensorboard = False
    tensorboard_logdir = './.runs'
