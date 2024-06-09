import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
from typing import Optional

import cv2
import numpy as np
from cv2.typing import MatLike

Method = Literal["average", "block_mean", "md5"]

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL = BASE_DIR / "models" / "digits.png"
DEFAULT_LABELS = BASE_DIR / "models" / "digits.json"


avg_hash = cv2.img_hash.AverageHash.create()
block_mean_hash = cv2.img_hash.BlockMeanHash.create()


# BlockMeanHash is 10x slower than AverageHash
def compute_hash(img: MatLike, method: Method):
    if method == "average":
        return avg_hash.compute(img).tobytes()
    if method == "block_mean":
        return block_mean_hash.compute(img).tobytes()
    if method == "md5":
        return hashlib.md5(img.tobytes()).digest()
    raise NotImplementedError(f"Hash method {method} not implemented.")


def compare_hash(hsh1: bytes, hsh2: bytes, method: Method):
    if method == "average":
        h1 = np.frombuffer(hsh1, np.uint8)
        h2 = np.frombuffer(hsh2, np.uint8)
        return avg_hash.compare(h1, h2)
    if method == "block_mean":
        h1 = np.frombuffer(hsh1, np.uint8)
        h2 = np.frombuffer(hsh2, np.uint8)
        return block_mean_hash.compare(h1, h2)
    if method == "md5":
        return 0.0 if hsh1 == hsh2 else float("inf")
    raise NotImplementedError(f"Hash method {method} not implemented.")


def threshold_image(img: MatLike) -> MatLike:
    white = np.array([255, 255, 255])
    img = cv2.inRange(img, white, white)
    return img


@lru_cache
def get_model(
    model_path: str | Path = DEFAULT_MODEL,
    label_path: str | Path = DEFAULT_LABELS,
    method: Method = "block_mean",
):
    with open(label_path) as fp:
        labels: list[Optional[str]] = json.load(fp)

    img = cv2.imread(str(model_path))
    img = threshold_image(img)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(cnts)
    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])

    output: list[tuple[str, bytes]] = []
    for i, cnt in enumerate(cnts):
        char = labels[i]
        if char is None:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        char_img: MatLike = img[y : y + h, x : x + w]
        hsh = compute_hash(char_img, method=method)
        output.append((char, hsh))
    return output


def compute_distances(
    threshed_img: MatLike,
    model_path: str | Path = DEFAULT_MODEL,
    label_path: str | Path = DEFAULT_LABELS,
    method: Method = "block_mean",
) -> list[list[tuple[str, float]]]:
    model = get_model(model_path, label_path, method)
    cnts, _ = cv2.findContours(
        threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = list(cnts)
    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])
    output: list[list[tuple[str, float]]] = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        letter_img: MatLike = threshed_img[y : y + h, x : x + w]
        hsh = compute_hash(letter_img, method=method)
        data = [
            (char, compare_hash(hsh, hsh1, method)) for char, hsh1 in model
        ]
        data.sort(key=lambda d: d[1])
        output.append(data)
    return output


def get_characters(
    threshed_img: MatLike,
    model_path: str | Path = DEFAULT_MODEL,
    label_path: str | Path = DEFAULT_LABELS,
    max_dist: int = 80,
    method: Method = "block_mean",
) -> list[tuple[str, float]]:
    output: list[tuple[str, float]] = []
    distances = compute_distances(threshed_img, model_path, label_path, method)
    for data in distances:
        data = [d for d in data if d[1] <= max_dist]
        if data == []:
            continue
        best = data[0]
        output.append(best)
    return output


def get_word(
    threshed_img: MatLike,
    model_path: str | Path = DEFAULT_MODEL,
    label_path: str | Path = DEFAULT_LABELS,
    max_dist: int = 80,
    method: Method = "block_mean",
) -> str:
    """
    This function assumes that the image contains only one word
    """
    chars = get_characters(
        threshed_img, model_path, label_path, max_dist, method
    )
    text = "".join(char for char, _ in chars)
    return text
