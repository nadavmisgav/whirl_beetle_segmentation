import os
import shutil
from glob import glob
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
brush_size = 5


def tag_image(image_path: Path):
    global brush_size
    # mouse callback function

    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, brush_size

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                cv.circle(img, (x, y), brush_size, (255, 255, 255), -1)
                cv.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            cv.circle(img, (x, y), brush_size, (255, 255, 255), -1)
            cv.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

    # read image from path and add callback
    name = image_path.stem
    orig_image = cv.imread(str(image_path))
    img = cv.resize(orig_image, (1280, 720))

    img = cv.convertScaleAbs(img, alpha=3, beta=100)
    mask = np.zeros_like(img)
    cv.namedWindow('image')
    # cv.setMouseCallback('image', draw_lines)
    cv.setMouseCallback('image', draw_circle)

    while (1):
        cv.imshow('image', img)
        cv.putText(img, f"brush_size: {brush_size}", (50, 50),
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255))
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('+'):
            brush_size += 5
        elif k == ord('-'):
            brush_size -= 5
            brush_size = max(brush_size, 1)

    # orig_image = cv.resize(orig_image, (128, 128))
    mask = cv.resize(mask, orig_image.shape[:2])

    cv.imwrite(str(TAG_ROOT / "masked" / f"{name[:-4]}_masked.jpg"), mask)
    cv.imwrite(str(TAG_ROOT / "cropped" / name), orig_image)

    cv.destroyAllWindows()


print("Creating directories")
TAG_ROOT = Path(__file__).parent / "data" / "tag"
(TAG_ROOT / "cropped").mkdir(exist_ok=True)
(TAG_ROOT / "data").mkdir(exist_ok=True)
(TAG_ROOT / "masked").mkdir(exist_ok=True)
(TAG_ROOT / "tagged").mkdir(exist_ok=True)
os.chdir(TAG_ROOT)


for image in tqdm((TAG_ROOT / "data").glob("*.jpg")):
    tag_image(image)
    print(image)
    shutil.move(str(image.resolve())., "tagged")
