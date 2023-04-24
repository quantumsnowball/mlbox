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


def scan_for_files(root: Path | str,
                   ext: str = 'pth',
                   sort_by_last_modified: bool = False) -> list[tuple[Path, float]]:
    '''
    scan recursively for files with `ext` under `root`
    '''
    result = []
    for path in Path(root).iterdir():
        if path.is_file() and path.suffix == f'.{ext}':
            last_modified = path.stat().st_mtime
            result.append((path, last_modified, ))
        elif path.is_dir():
            result += scan_for_files(path)
    if sort_by_last_modified:
        result.sort(key=lambda x: x[-1])
    return result
