import os
import shutil
from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import cc3d
import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def sub_median(images: np.ndarray):
    # remove median
    med = np.expand_dims(np.median(images, axis=2), axis=2)
    images = images - med
    images[images < 0] = 0
    return images


def find_max_cc(images: np.ndarray):
    labels_out = cc3d.connected_components(images, connectivity=6)
    labels_vec = labels_out.flatten()
    bins = np.bincount(labels_vec[labels_vec > 0])
    max_label = bins.argmax()
    labels_out[labels_out != max_label] = 0.0
    labels_out[labels_out == max_label] = 255.0

    return labels_out

def transform_image(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    # img = img / 255.0
    # assert np.max(img) <= 1
    # assert np.min(img) >= 0
    img[img >= 215] = 255
    img[img < 215] = 0
    return img


def get_frames(video_path, batch_size=200, max_frames=200, transform=lambda img: img):
    cap = cv2.VideoCapture(video_path)
    images = []

    success, image = cap.read()
    images.append(image)
    count = 0
    while success:
        while success:
            success, image = cap.read()
            if not success:
                break
            images.append(image)

            if len(images) == batch_size:
                break

        # orig = np.stack(images, axis=2)
        orig = np.stack(list(map(lambda x: cv2.cvtColor(
            x, cv2.COLOR_BGR2GRAY), images)), axis=2)
        image_3d = np.stack(list(map(transform_image, images)), axis=2)
        yield orig, image_3d
        count += batch_size
        if count >= max_frames:
            print(f"reached max_frames={max_frames} stopping processing")
            break
        images = []


def crop_video(orig_images, image_3d, height, width):
    subbed = sub_median(image_3d)
    cc_image = find_max_cc(subbed)
    orig_subbed = sub_median(orig_images)

    for idx in range(orig_images.shape[-1]):
        gray_img = cc_image[:, :, idx].astype("uint8")
        row, col = ndimage.center_of_mass(gray_img)
        debug_img = orig_subbed[:, :, idx]

        rot_rectangle = ((col, row), (height, width), 0)
        box = cv2.boxPoints(rot_rectangle)
        [x, y, _, _] = cv2.boundingRect(box)

        padded = cv2.copyMakeBorder(
            debug_img, height, height, width, width, cv2.BORDER_CONSTANT, value=0)
        yield padded[y+height:y+2*height, x+width:x+2*width]


def main(video_path: Path, height: int, width: int, batch_size: int, num_batches: int):
    OUTPUT_DIR = Path(__file__).parent / "data" / (video_path.stem + "_frames")
    try:
        OUTPUT_DIR.mkdir()
    except OSError:
        print(f"Path {OUTPUT_DIR} exists, delete it first")
        return

    c = count()
    for images, image_3d in get_frames(video_path, batch_size=batch_size, max_frames=batch_size*num_batches, transform=transform_image):
        batch_idx = next(c)
        for idx, frame in tqdm(enumerate(crop_video(images, image_3d, height, width)), total=batch_size):
            idx_str = str(batch_idx*batch_size + idx).zfill(5)
            path = OUTPUT_DIR / f"frame{idx_str}.jpg"
            cv2.imwrite(str(path), frame)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("video_path", help="path to video", type=Path)
    parser.add_argument("--height", type=int,
                        default=300, help="height of box")
    parser.add_argument("--width", type=int,
                        default=300, help="width of box")
    parser.add_argument("-b", "--batch", help="Batch size for processing", default=200)
    parser.add_argument("-c", "--count", help="Number of batches to create", default=1)
    args = parser.parse_args()

    main(args.video_path, args.height, args.width, args.batch, args.count)
