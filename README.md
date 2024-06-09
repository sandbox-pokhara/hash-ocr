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

from hash_ocr import AverageHashModel

img = cv2.imread("test_data/382.png", cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

# default model can only read digits
model = AverageHashModel()

print(model.get_word(img))
# 382

print(model.compute_distances(img))
# [(8.0, '3', (5, 6, 18, 25)), (7.0, '8', (24, 5, 18, 26)), (10.0, '2', (42, 6, 20, 24))]
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
from hash_ocr.models import AverageHashModel

model = AverageHashModel(
    model_path="hash_ocr/models/digits.png",
    label_path="hash_ocr/models/letters.json",
)
```

## License

This project is licensed under the terms of the MIT license.
