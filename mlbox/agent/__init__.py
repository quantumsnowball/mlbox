from collections import deque
from inspect import currentframe
from pathlib import Path
from statistics import mean
from typing import Self

import torch as T
from gymnasium import Env
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import override

from mlbox.agent.props import Props
from mlbox.events import TerminatedError
from mlbox.interface.agent import Agent
from mlbox.types import T_Action, T_Obs
from mlbox.utils.io import scan_for_files, state_dict_info


class BasicAgent(Props[T_Obs, T_Action],
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
    state_dict_file_ext = 'pth'

    @override
    def prompt(self,
               name: str,
               *,
               start_training: bool = False) -> None:
        # scan for .pth files
        state_dict_files = scan_for_files(self.script_basedir, ext=self.state_dict_file_ext)
        if len(state_dict_files) > 0:
            print(f'{len(state_dict_files)} has been found:')
            for i, state_dict_path in enumerate(state_dict_files):
                print(f'\n{i+1}. {str(state_dict_path.relative_to(self.script_basedir))}')
                print(state_dict_info(T.load(state_dict_path)))
            while True:
                choice = input(f'\nChoose a model to load (Enter to skip): ')
                if choice == '':
                    break
                try:
                    selected_idx = int(choice) - 1
                    assert 0 <= selected_idx < len(state_dict_files)
                    chosen_file = state_dict_files[selected_idx]
                except Exception:
                    print(f'Please enter a valid model index [1 - {len(state_dict_files)}]')
                    continue
                # load agent
                self.load(chosen_file)
                print(f'Loaded model: {str(chosen_file.relative_to(self.script_basedir))}')
                break

        if start_training or input(f'Start training the agent? ([y]/n) ').upper() != 'N':
            # train agent
            self.train()
            if input(f'Save model? y/[n]) ').upper() == 'Y':
                save_path = self.script_basedir / name
                self.save(save_path)
                print(f'Saved model: {save_path}')

    #
    # tensorboard
    #

    tensorboard = False
    tensorboard_logdir = './.runs'
