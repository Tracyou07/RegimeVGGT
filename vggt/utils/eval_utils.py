from collections import deque
import cv2
import random


def shuffle_deque(dq, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Convert deque to list, shuffle, and convert back
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
