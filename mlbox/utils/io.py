from pathlib import Path

from torch import Tensor


def state_dict_info(d: dict[str, dict]) -> str:
    '''
    returns a state dict save file's info
    '''
    repr = ''
    for i, (key, val, ) in enumerate(d.items()):
        # a tensor node
        if isinstance(val, Tensor):
            repr += f"{key}{tuple(val.shape)}"
            if i < len(d) - 1:
                repr += ' '
        # a nested dict
        elif isinstance(val, dict):
            repr += key + ' : '
            repr += state_dict_info(val)
            if i < len(d) - 1:
                repr += '\n'
    return repr


def scan_for_files(root: Path | str, ext='pth') -> list[Path]:
    '''
    scan recursively for files with `ext` under `root`
    '''
    result = []
    for path in Path(root).iterdir():
        if path.is_file() and path.suffix == f'.{ext}':
            result.append(path)
        elif path.is_dir():
            result += scan_for_files(path)
    return result
