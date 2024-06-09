import hashlib
import json
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike


class Model:
    def __init__(self, model_path: str | Path, label_path: str | Path):
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


class CV2Hash(Model):
    def __init__(
        self,
        model_path: str | Path,
        label_path: str | Path,
        hasher: cv2.img_hash.ImgHashBase,
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


class AverageHash(CV2Hash):
    def __init__(self, model_path: str | Path, label_path: str | Path):
        super().__init__(
            model_path, label_path, cv2.img_hash.AverageHash.create()
        )


# BlockMeanHash is 10x slower than AverageHash
class BlockMeanHash(CV2Hash):
    def __init__(self, model_path: str | Path, label_path: str | Path):
        super().__init__(
            model_path, label_path, cv2.img_hash.BlockMeanHash.create()
        )


# MD5Hash is the fatest algorithm, but the letters need to match perfectly
class MD5Hash(Model):
    def __init__(self, model_path: str | Path, label_path: str | Path):
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
