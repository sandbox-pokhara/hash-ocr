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
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "beaufont.png"
DEFAULT_LABEL_PATH = BASE_DIR / "models" / "beaufont.json"


class Model:
    def __init__(
        self,
        model_path: str | Path,
        label_path: str | Path = DEFAULT_LABEL_PATH,
        score_threshold: float = 80.0,
        max_letter_gap: int = 4,
        max_word_gap: int = 10,
    ):
        self.model_path = model_path
        self.label_path = label_path
        self.score_threshold = score_threshold
        self.max_letter_gap = max_letter_gap
        self.max_word_gap = max_word_gap

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

    # text boxes
    def get_char_boxes(
        self, threshed_img: MatLike
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        output: List[Tuple[str, Tuple[int, int, int, int]]] = []
        scores = self.compute_scores(threshed_img)
        for score, letter, rect in scores:
            if score > self.score_threshold:
                continue
            output.append((letter, rect))
        return output

    def get_word_boxes(
        self, threshed_img: MatLike
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        word_seperator = cv2.dilate(
            threshed_img, np.ones((1, self.max_letter_gap))
        )
        cnts, _ = cv2.findContours(
            word_seperator, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        words: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            word_img = threshed_img[y : y + h, x : x + w]
            word = self.get_word(word_img)
            if word:
                words.append((word, (x, y, w, h)))
        words.sort(key=lambda w: w[1][0])
        return words

    def get_line_boxes(
        self, threshed_img: MatLike
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        line_seperator = cv2.dilate(
            threshed_img, np.ones((1, self.max_word_gap))
        )
        cnts, _ = cv2.findContours(
            line_seperator, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lines: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for c in reversed(cnts):
            x, y, w, h = cv2.boundingRect(c)
            line_img = threshed_img[y : y + h, x : x + w]
            words = self.get_word_boxes(line_img)
            word = " ".join([c for c, _ in words])
            if word:
                lines.append((word, (x, y, w, h)))
        return lines

    # texts
    def get_word(self, threshed_img: MatLike) -> str:
        chars = self.get_char_boxes(threshed_img)
        text = "".join(char for char, _ in chars)
        return text

    def get_line(self, threshed_img: MatLike) -> str:
        chars = self.get_word_boxes(threshed_img)
        text = " ".join(char for char, _ in chars)
        return text

    def get_text(self, threshed_img: MatLike):
        lines = self.get_line_boxes(threshed_img)
        text = "\n".join([c for c, _ in lines])
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
        """
        :param connected_chars: set this to True if the characters of
        the font can touch each other, but this will affect the performance
        of the OCR
        """
        super().__init__(model_path, label_path)
        self.chars = self.get_model_chars()
        self.hash_map = self.create_hash_map()
        # max/min width of characters of the model
        self.min_w = min([s[1].shape[1] for s in self.chars])
        self.max_w = max([s[1].shape[1] for s in self.chars])
        self.connected_chars = connected_chars

    def create_hash_map(self):
        return {self.compute_hash(i): c for c, i in self.chars}

    def compute_hash(self, img: MatLike):
        return hashlib.md5(img.tobytes()).digest()

    def classify_connected_letters(self, img: MatLike) -> Tuple[float, str]:
        """
        Classify the connected letters by splitting them into small chunks
        This method is slow
        """
        if img.shape[0] * img.shape[1] == 0:
            return float("inf"), ""

        min_w = self.min_w
        # improves performance a lot
        max_w = min(img.shape[1], self.max_w)
        for w in reversed(range(min_w, max_w + 1)):
            partial_img = img[:, :w]
            cnts, _ = cv2.findContours(
                partial_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                mask = np.zeros_like(partial_img, np.uint8)
                cv2.drawContours(mask, [c], -1, [255], -1)
                char_img = cv2.bitwise_and(partial_img, mask)
                char_img = char_img[y : y + h, x : x + w]

                hsh = self.compute_hash(char_img)
                if hsh in self.hash_map:
                    return (
                        0.0,
                        self.hash_map[hsh]
                        + self.classify_connected_letters(img[:, w:])[1],
                    )
        return float("inf"), ""

    def classify_letter(self, img: MatLike) -> Tuple[float, str]:
        if img.shape[1] < self.min_w:
            return float("inf"), ""
        hsh = self.compute_hash(img)
        char = self.hash_map.get(hsh, "")
        if not char and self.connected_chars:
            # if the char is not in hash map
            # it can be a contour of connected characters
            return self.classify_connected_letters(img)
        score = 0.0 if char else float("inf")
        return score, char


def draw_text_boxes(
    img: MatLike,
    text_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
    font: int = 0,
    size: float = 0.4,
    offset: int = -5,
):
    for txt, rect in text_boxes:
        x, y, _, _ = rect
        cv2.rectangle(img, rect, (0, 255, 0))
        cv2.putText(img, txt, (x, y + offset), font, size, (0, 255, 0))
