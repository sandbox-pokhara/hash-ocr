from pathlib import Path
from typing import List
from typing import Literal
from typing import Tuple

import cv2
from cv2.typing import MatLike

from hash_ocr.models import AverageHash
from hash_ocr.models import Model

Method = Literal["average", "block_mean", "md5"]

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "digits.png"
DEFAULT_LABEL_PATH = BASE_DIR / "models" / "digits.json"
DEFAUTL_MODEL = AverageHash(DEFAULT_MODEL_PATH, DEFAULT_LABEL_PATH)


def compute_distances(
    threshed_img: MatLike, model: Model = DEFAUTL_MODEL
) -> List[Tuple[float, str, Tuple[int, int, int, int]]]:
    cnts, _ = cv2.findContours(
        threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = list(cnts)
    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])
    output: List[Tuple[float, str, Tuple[int, int, int, int]]] = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        letter_img: MatLike = threshed_img[y : y + h, x : x + w]
        score, char = model.classify_letter(letter_img)
        output.append((score, char, (x, y, w, h)))
    return output


def get_characters(
    threshed_img: MatLike, model: Model = DEFAUTL_MODEL, max_dist: int = 80
) -> List[Tuple[float, str, Tuple[int, int, int, int]]]:
    output: List[Tuple[float, str, Tuple[int, int, int, int]]] = []
    distances = compute_distances(threshed_img, model)
    for dist, letter, rect in distances:
        if dist > max_dist:
            continue
        output.append((dist, letter, rect))
    return output


def get_word(
    threshed_img: MatLike, model: Model = DEFAUTL_MODEL, max_dist: int = 80
) -> str:
    """
    This function assumes that the image contains only one word
    """
    chars = get_characters(threshed_img, model, max_dist)
    text = "".join(char for _, char, _ in chars)
    return text
