"""
Model plane points I/O and checkerboard generator (Assignment 2)

>>> import numpy as np
>>> import tempfile, json, csv
>>> arr = np.array([[0.1,0.2],[0.3,0.4]])
>>> with tempfile.NamedTemporaryFile('w+', suffix='.csv', delete=False) as f:
...     csv.writer(f).writerows(arr)
...     fname = f.name
>>> out = load_model_points(fname)
>>> out.shape == (2,2)
True
>>> np.allclose(out, arr)
True
>>> with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as f:
...     json.dump(arr.tolist(), f)
...     fname = f.name
>>> out = load_model_points(fname)
>>> out.shape == (2,2)
True
>>> np.allclose(out, arr)
True
>>> pts = generate_checkerboard(3,2,0.01)
>>> pts.shape == (6,2)
True
>>> np.allclose(pts[0], [0,0])
True
"""
import numpy as np
import json
import csv
import os

def load_model_points(path: str) -> np.ndarray:
    """
    Load model points from CSV or JSON. Returns (N,2) float64 in meters.
    """
    ext = os.path.splitext(path)[1].lower()
    arr = None
    if ext == '.csv':
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            arr = np.array([[float(row[0]), float(row[1])] for row in reader], dtype=np.float64)
    elif ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'points' in data:
                arr = np.array(data['points'], dtype=np.float64)
                units = data.get('units', 'm')
                if units == 'cm':
                    arr /= 100.0
            else:
                arr = np.array(data, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    if arr.shape[1] < 2:
        raise ValueError("Model points must have at least two columns (X,Y)")
    return arr[:, :2]

def generate_checkerboard(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """
    Generate checkerboard inner corners in row-major order.
    Returns (N,2) float64.
    """
    pts = np.array([[i*square_size_m, j*square_size_m] for j in range(rows) for i in range(cols)], dtype=np.float64)
    return pts
