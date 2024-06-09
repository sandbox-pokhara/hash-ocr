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

from hash_ocr import compute_distances
from hash_ocr import get_word

img = cv2.imread("test_data/382.png", cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

print(get_word(img))
# 382

for d in compute_distances(img):
    print(d)
# [('3', 24.0), ('8', 66.0), ('2', 74.0), ('7', 77.0), ...]
# [('8', 24.0), ('6', 60.0), ('0', 62.0), ('3', 68.0), ...]
# [('2', 20.0), ('3', 70.0), ('1', 76.0), ('7', 85.0), ...]
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
get_word(
    img,
    model_path="path/to/image",
    label_path="path/to/label",
)
```

## License

This project is licensed under the terms of the MIT license.
