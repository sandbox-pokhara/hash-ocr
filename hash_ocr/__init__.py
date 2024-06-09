import hashlib
import json
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "digits.png"
DEFAULT_LABEL_PATH = BASE_DIR / "models" / "digits.json"


class Model:
    def __init__(
        self,
        model_path: str | Path,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ):
        self.model_path = model_path
        self.label_path = label_path

    def classify_letter(self, img: MatLike) -> Tuple[float, str]:
        raise NotImplementedError

    def threshold_img(self, img: MatLike):
        white = np.array([255, 255, 255])
        img = cv2.inRange(img, white, white)
        return img

    def get_model_chars(self):
        with open(self.label_path) as fp:
            labels: list[Optional[str]] = json.load(fp)
        img = cv2.imread(str(self.model_path))
        img = self.threshold_img(img)
        cnts, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = list(cnts)
        cnts.sort(key=lambda x: cv2.boundingRect(x)[0])
        output: list[tuple[str, MatLike]] = []
        for i, cnt in enumerate(cnts):
            char = labels[i]
            if char is None:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            char_img: MatLike = img[y : y + h, x : x + w]
            output.append((char, char_img))
        return output

    def compute_distances(
        self, threshed_img: MatLike
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
            score, char = self.classify_letter(letter_img)
            output.append((score, char, (x, y, w, h)))
        return output

    def get_characters(
        self, threshed_img: MatLike, max_dist: int = 80
    ) -> List[Tuple[float, str, Tuple[int, int, int, int]]]:
        output: List[Tuple[float, str, Tuple[int, int, int, int]]] = []
        distances = self.compute_distances(threshed_img)
        for dist, letter, rect in distances:
            if dist > max_dist:
                continue
            output.append((dist, letter, rect))
        return output

    def get_word(self, threshed_img: MatLike, max_dist: int = 80) -> str:
        """
        This function assumes that the image contains only one word
        """
        chars = self.get_characters(threshed_img, max_dist)
        text = "".join(char for _, char, _ in chars)
        return text


class CV2HashModel(Model):
    def __init__(
        self,
        hasher: cv2.img_hash.ImgHashBase,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ):
        super().__init__(model_path, label_path)
        self.hasher = hasher
        self.hashes = self.get_model_hashes()

    def compute_hash(self, img: MatLike):
        return self.hasher.compute(img)

    def get_model_hashes(self):
        output: list[tuple[str, MatLike]] = []
        for char, img in self.get_model_chars():
            output.append((char, self.compute_hash(img)))
        return output

    def classify_letter(self, img: MatLike) -> Tuple[float, str]:
        hsh = self.compute_hash(img)
        scores: List[tuple[float, str]] = []
        for char, hsh1 in self.hashes:
            score = self.hasher.compare(hsh, hsh1)
            scores.append((score, char))
        scores.sort()
        return scores[0]


class AverageHashModel(CV2HashModel):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ):
        super().__init__(
            cv2.img_hash.AverageHash.create(),
            model_path,
            label_path,
        )


# BlockMeanHash is 10x slower than AverageHash
class BlockMeanHashModel(CV2HashModel):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ):
        super().__init__(
            cv2.img_hash.BlockMeanHash.create(), model_path, label_path
        )


# MD5Hash is the fatest algorithm, but the letters need to match perfectly
class MD5HashModel(Model):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_path: str | Path = DEFAULT_LABEL_PATH,
    ):
        super().__init__(model_path, label_path)
        self.hash_map = {
            self.compute_hash(i): c for c, i in self.get_model_chars()
        }

    def compute_hash(self, img: MatLike):
        return hashlib.md5(img.tobytes()).digest()

    def classify_letter(self, img: MatLike) -> Tuple[float, str]:
        hsh = self.compute_hash(img)
        char = self.hash_map.get(hsh, "")
        score = 0.0 if char else float("inf")
        return score, char
