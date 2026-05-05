"""

save and load json

"""


from typing import Callable, Dict

from enum import Enum

import json
from json import JSONEncoder


import torch
import numpy as np






class CustomEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray) or isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, Enum):
            return o.name
        if o is None or isinstance(o, Callable):
            return ''    
        return o.__dict__


def save_as_json(obj: object, save_to: str) -> None:
    json_str = json.dumps(obj, indent = 4,
                          cls = CustomEncoder)
    
    with open(save_to, 'w') as f:
        f.write(json_str)


def load_json(load_from: str) -> Dict:
    
    with open(load_from, 'r') as f:
        json_obj = json.load(f)
        
    return json_obj
