from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

if TYPE_CHECKING:
    from mlbox.agent import BasicAgent

from mlbox.utils.datetime import localnow_string


class TeeInput:
    def __init__(self, input: TextIO, file: TextIO) -> None:
        self.input = input
        self.file = file

    def readline(self) -> str:
        line = self.input.readline()
        self.file.write(line)
        return line


class TeeOutput:
    def __init__(self, output: TextIO, file: TextIO) -> None:
        self.output = output
        self.file = file

    def write(self, obj: str) -> None:
        for f in (self.output, self.file):
            f.write(obj)
            f.flush()

    def flush(self) -> None:
        for f in (self.output, self.file):
            f.flush()


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.file = open(path, 'w')

    def __enter__(self) -> TextIO:
        sys.stdin = TeeInput(sys.stdin, self.file)
        sys.stdout = TeeOutput(sys.stdout, self.file)
        return self.file

    def __exit__(self, *_) -> None:
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__
        self.file.close()


def log_output(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    @wraps(fn)
    def wrapped(self: BasicAgent,
                *args: Any,
                **kwargs: Any) -> None:
        if self.auto_log_output:
            save_dir = self.script_basedir / '.output'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'stdout_{localnow_string()}.txt'
            with TeeLogger(save_path):
                fn(self, *args, **kwargs)
        else:
            breakpoint()
            fn(self, *args, **kwargs)
    return wrapped
