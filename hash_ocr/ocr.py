import json
import os
from functools import lru_cache
from typing import Optional
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "model", "digits.png")
hash_obj = cv2.img_hash.BlockMeanHash.create()

Hash = npt.NDArray[np.uint8]


def threshold_image(img: MatLike) -> MatLike:
    white = np.array([255, 255, 255])
    img = cv2.inRange(img, white, white)
    return img


@lru_cache
def get_model(file_path: str = DEFAULT_MODEL):
    label_file_path = file_path.replace(".png", ".txt")
    with open(label_file_path) as fp:
        labels: list[Optional[str]] = json.load(fp)

    img = cv2.imread(file_path)
    img = threshold_image(img)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(cnts)
    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])

    output: list[tuple[str, Hash]] = []
    for i, cnt in enumerate(cnts):
        char = labels[i]
        if char is None:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        char_img: MatLike = img[y : y + h, x : x + w]
        hsh = compute_hash(char_img)
        output.append((char, hsh))
    return output


def compute_hash(img: MatLike) -> npt.NDArray[np.uint8]:
    hsh = hash_obj.compute(img)
    hsh = cast(npt.NDArray[np.uint8], hsh)
    return hsh


def compute_distances(
    threshed_img: MatLike, model_path: str = DEFAULT_MODEL
) -> list[list[tuple[str, float]]]:
    model = get_model(file_path=model_path)
    cnts, _ = cv2.findContours(
        threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = list(cnts)
    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])
    output: list[list[tuple[str, float]]] = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        letter_img: MatLike = threshed_img[y : y + h, x : x + w]
        hsh = compute_hash(letter_img)
        data = [(char, hash_obj.compare(hsh, hsh1)) for char, hsh1 in model]
        data.sort(key=lambda d: d[1])
        output.append(data)
    return output


def get_characters(
    threshed_img: MatLike, model_path: str = DEFAULT_MODEL, max_dist: int = 80
) -> list[tuple[str, float]]:
    output: list[tuple[str, float]] = []
    distances = compute_distances(threshed_img, model_path=model_path)
    for data in distances:
        data = [d for d in data if d[1] <= max_dist]
        if data == []:
            continue
        best = data[0]
        output.append(best)
    return output


def get_word(
    threshed_img: MatLike, model_path: str = DEFAULT_MODEL, max_dist: int = 80
) -> str:
    """
    This function assumes that the image contains only one word
    """
    chars = get_characters(
        threshed_img, model_path=model_path, max_dist=max_dist
    )
    text = "".join(char for char, _ in chars)
    return text
