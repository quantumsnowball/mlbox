from collections import deque
from inspect import currentframe
from pathlib import Path
from statistics import mean
from typing import Self

import torch as T
from gymnasium import Env
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import override

from mlbox.interface.agent import Agent
from mlbox.agent.props import BasicAgentProps
from mlbox.events import TerminatedError
from mlbox.types import T_Action, T_Obs


class BasicAgent(BasicAgentProps[T_Obs, T_Action],
                 Agent[T_Obs, T_Action]):
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
        self.vald_rolling_reward = deque[float](maxlen=self.rolling_reward_ma)
        self.vald_score_high: float | None = None

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
    auto_save = False
    auto_save_filename = 'model.pth'
    auto_save_start_eps = 3

    def print_evaluation_result(self,
                                i: int) -> None:
        if i % self.report_progress_every == 0:
            # count
            print(f' | Episode {i:>4d}', end='')
            # train
            self.rolling_reward.append(self.play())
            train_score = mean(self.rolling_reward)
            print(f' | train: {train_score:{self.mean_reward_display_format}}', end='')
            # validation
            self.vald_rolling_reward.append(self.play(env=self.vald_env))
            vald_score = mean(self.vald_rolling_reward)
            print(f' | vald: {vald_score:{self.mean_reward_display_format}}', end='')
            # save best model
            if (self.auto_save
                    and len(self.vald_rolling_reward) >= self.auto_save_start_eps
                    and (self.vald_score_high is None or vald_score > self.vald_score_high)):
                self.save(self.script_basedir / self.auto_save_filename)
                print(f' [saved]', end='')
                self.vald_score_high = vald_score
            # newline
            print('')

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
                print(f'Loaded model: {path}')
        if start_training or input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train()
            if input(f'Save model? y/[n]) ').upper() == 'Y':
                self.save(path)
                print(f'Saved model: {path}')

    #
    # tensorboard
    #

    tensorboard = False
    tensorboard_logdir = './.runs'
