import argparse
import json
from typing import Optional

import cv2
from cv2.typing import MatLike

from hash_ocr.ocr import threshold_image


def draw_labels(
    img: MatLike, labels: list[Optional[str]], cnts: list[MatLike]
):
    for i, label in enumerate(labels):
        try:
            if label is None:
                continue
            x, y, _, h = cv2.boundingRect(cnts[i])
            cv2.putText(img, label, (x, y + h), 0, 1, (0, 0, 255), 2)
        except IndexError:
            pass


def label(file_path: str):
    label_file_path = file_path.replace(".png", ".txt")

    img = cv2.imread(file_path)
    img = threshold_image(img)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(cnts)

    labels: list[Optional[str]]
    try:
        with open(label_file_path) as fp:
            labels = json.load(fp)
    except FileNotFoundError:
        labels = [None] * len(cnts)

    cnts.sort(key=lambda x: cv2.boundingRect(x)[0])
    index = 0
    while True:
        index = index % len(cnts)
        x, y, w, h = cv2.boundingRect(cnts[index])

        display = img_color.copy()

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_labels(display, labels, cnts)
        cv2.imshow("", display)
        key = cv2.waitKeyEx()

        # enter
        # save labels and move to next image
        if key == 13:
            with open(label_file_path, "w") as fp:
                json.dump(labels, fp)
            break

        # escape or close window
        if key == -1 or key == 27:
            return

        # delete
        if key == 3014656:
            labels[index] = None
            continue

        # right arrow
        # go to next index
        if key == 2555904:
            index += 1
            continue

        # left arrow or backspace
        # go back to previous index
        if key == 2424832 or key == 8:
            index -= 1
            continue

        # if valid character
        # assign to label
        try:
            labels[index] = chr(key)
            index += 1
        except ValueError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    label(args.input)
