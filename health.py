import os
import torch
import numpy as np
import types

def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


if __name__=="__main__":
    imports()
