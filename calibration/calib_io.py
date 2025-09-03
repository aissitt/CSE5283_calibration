import json
import numpy as np
from typing import Dict


def save_calibration_json(calib: Dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration_json(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    data["_K_np"] = np.array(data["K"], dtype=float)
    data["_dist_np"] = np.array(data["dist"], dtype=float).reshape(-1, 1)
    return data
