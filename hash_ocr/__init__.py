import hashlib
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "beaufont.png"
DEFAULT_LABEL_PATH = BASE_DIR / "models" / "beaufont.json"


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

    def unify_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        """
        Returns a bounding box that encloses a list of bounding boxes
        """
        # Initialize variables to extreme values
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for x, y, w, h in boxes:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Calculate the unified bounding box coordinates
        unified_x = int(min_x)
        unified_y = int(min_y)
        unified_w = int(max_x - min_x)
        unified_h = int(max_y - min_y)

        return (unified_x, unified_y, unified_w, unified_h)

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
        output: List[Tuple[str, MatLike]] = []
        for i, cnt in enumerate(cnts):
            char = labels[i]
            if char is None:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            char_img: MatLike = img[y : y + h, x : x + w]
            output.append((char, char_img))
        return output

    def compute_scores(
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

    def get_char_boxes(
        self, threshed_img: MatLike, score_threshold: int = 80
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        output: List[Tuple[str, Tuple[int, int, int, int]]] = []
        scores = self.compute_scores(threshed_img)
        for score, letter, rect in scores:
            if score > score_threshold:
                continue
            output.append((letter, rect))
        return output

    def get_word_box(
        self, threshed_img: MatLike, score_threshold: int = 80
    ) -> Tuple[str, Tuple[int, int, int, int]]:
        chars = self.get_char_boxes(threshed_img, score_threshold)
        if not chars:
            return "", (-1, -1, -1, -1)
        text = "".join(char for char, _ in chars)
        box = self.unify_boxes([b for _, b in chars])
        return text, box

    def get_word(
        self, threshed_img: MatLike, score_threshold: int = 80
    ) -> str:
        chars = self.get_char_boxes(threshed_img, score_threshold)
        text = "".join(char for char, _ in chars)
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
        connected_chars: bool = False,
    ):
        super().__init__(model_path, label_path)
        self.chars = self.get_model_chars()
        self.hash_map = self.create_hash_map()
        if connected_chars:
            self.partial_hash_map = self.create_partial_hash_map()
        else:
            self.partial_hash_map = {}

    def create_hash_map(self):
        return {self.compute_hash(i): c for c, i in self.chars}

    def create_partial_hash_map(self):
        """
        Hash map of partial letters
        Crops each letter into smallar chunks and store the hash
        """
        w = [i.shape[1] for _, i in self.chars]
        min_w = min(w)
        max_w = max(w)
        output: Dict[int, Dict[bytes, Tuple[str, int]]] = {}
        for i in range(min_w, max_w + 1):
            mapping: Dict[bytes, Tuple[str, int]] = {}
            duplicate_hsh: set[bytes] = set()
            for char, img in self.chars:
                cropped = img[:, :i]
                cnts, _ = cv2.findContours(
                    cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                x, y, w, h = cv2.boundingRect(cnts[0])
                cropped = cropped[y : y + h, x : x + w]
                hsh = self.compute_hash(cropped)
                if hsh in mapping:
                    duplicate_hsh.add(hsh)
                mapping[hsh] = char, img.shape[1]
            for hsh in duplicate_hsh:
                del mapping[hsh]
            output[i] = mapping
        return output

    def compute_hash(self, img: MatLike):
        return hashlib.md5(img.tobytes()).digest()

    def classify_connected_letters(self, img: MatLike) -> Tuple[float, str]:
        """
        Classify the connected letters by splitting them into
        small chunks and check the hash in partial_hash_map
        """
        for w, hsh_map in self.partial_hash_map.items():
            if w > img.shape[1]:
                return float("inf"), ""
            partial_img = img[:, :w]
            cnts, _ = cv2.findContours(
                partial_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                char_img: MatLike = partial_img[y : y + h, x : x + w]
                hsh = self.compute_hash(char_img)
                if hsh in hsh_map:
                    char, char_w = hsh_map[hsh]
                    return (
                        0.0,
                        char
                        + self.classify_connected_letters(img[:, char_w:])[1],
                    )

        return float("inf"), ""

    def classify_letter(self, img: MatLike) -> Tuple[float, str]:
        hsh = self.compute_hash(img)
        char = self.hash_map.get(hsh, "")
        if not char and self.partial_hash_map:
            # if the char is not in hash map
            # it can be a contour of connected characters
            return self.classify_connected_letters(img)
        score = 0.0 if char else float("inf")
        return score, char
