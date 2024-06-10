# hash-ocr

An ocr designed to read in game texts

## Installation

You can install the package via pip:

```
pip install hash-ocr
```

## Usage

```python
import cv2

from hash_ocr import MD5HashModel

img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

# default model can only read digits
model = MD5HashModel(connected_chars=True)

print(model.get_word(img))
# Igillessabl

print(model.get_word_box(img))
# ('Igillessabl', (2, 2, 69, 14))

print(model.get_char_boxes(img))
# [('I', (2, 2, 4, 11)), ('g', (7, 5, 7, 11)), ('ill', (15, 2, 12, 11)), ('e', (29, 5, 6, 8)), ('s', (37, 5, 5, 8)), ('s', (44, 5, 5, 8)), ('a', (50, 5, 7, 8)), ('b', (57, 2, 8, 11)), ('l', (67, 2, 4, 11))]
```

## Custom Models

A model in `hash-ocr` contains an image and a json file.

Example image:

![Model Image](hash_ocr/models/digits.png)

Example label:

```json
["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
```

Example:

```python
from hash_ocr.models import MD5HashModel

model = MD5HashModel(
    model_path="hash_ocr/models/digits.png",
    label_path="hash_ocr/models/letters.json",
)
```

## License

This project is licensed under the terms of the MIT license.
