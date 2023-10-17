# hash-ocr

An ocr designed to read in game texts

## Installation

You can install the package via pip:

```bash
pip install hash-ocr
```

## Usage

```python
import cv2

from hash_ocr import compute_distances
from hash_ocr import get_characters
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

## License

This project is licensed under the terms of the MIT license.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

If you want to contact me you can reach me at pradishbijukchhe@gmail.com.
