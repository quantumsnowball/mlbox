from torch import Tensor


def print_state_dict(d: dict[str, dict]) -> str:
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
            repr += print_state_dict(val)
            if i < len(d) - 1:
                repr += '\n'
    return repr
