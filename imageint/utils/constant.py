import numpy as np
import json
import os
from imageint.constant import PROJ_ROOT


def load_metadata():
    with open(os.path.join(PROJ_ROOT, 'metadata.json')) as f:
        metadata = json.load(f)
    return metadata


def load_error_metadata():
    with open(os.path.join(PROJ_ROOT, 'error_metadata.json')) as f:
        metadata = json.load(f)
    return metadata
