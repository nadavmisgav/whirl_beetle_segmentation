from pathlib import Path

import cc3d
import cv2
import numpy as np


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
    img[img >= 215] = 255
    img[img < 215] = 0
    return img


def get_frames(video_path: Path, batch_size=200, max_frames=200, transform=lambda img: img):
    cap = cv2.VideoCapture(str(video_path))
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
