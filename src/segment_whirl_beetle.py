import os
import sys
from argparse import ArgumentParser
from pathlib import Path

LIB = Path(__file__).parent.parent / "lib"
FGSEGNET = LIB / "FgSegNet"
sys.path.insert(0, str(LIB.parent)) 
sys.path.insert(0, str(FGSEGNET)) 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from lib.FgSegNet.FgSegNet_v2_module import acc, acc2, loss, loss2
from lib.FgSegNet.instance_normalization import InstanceNormalization
from lib.FgSegNet.my_upsampling_2d import MyUpSampling2D
from scipy import ndimage

from utils import (calc_velocity_angle, chunks, draw_arrow, find_max_cc,
                   get_frames, sub_median, transform_image)


def predict(weights: Path, data: np.ndarray) -> np.ndarray:
    """
    caches model after load weights
    """
    if predict.model is None:
        predict.model = load_model(str(weights), custom_objects={'MyUpSampling2D': MyUpSampling2D, 'InstanceNormalization': InstanceNormalization, 'loss':loss, 'acc':acc, 'loss2':loss2, 'acc2':acc2})
    probability_mask: np.ndarray = predict.model.predict(data, batch_size=data.shape[0], verbose=1)
    return probability_mask

predict.model = None

def crop_video(orig_images, image_3d, height, width, prev=[(None, None)]):
    subbed = sub_median(image_3d)
    cc_image = find_max_cc(subbed)
    orig_subbed = sub_median(orig_images)
    velocity_angle = None

    for idx in range(orig_images.shape[-1]):
        prev_row, prev_col = prev.pop()
        gray_img = cc_image[..., idx].astype("uint8")
        row, col = ndimage.center_of_mass(gray_img)
        debug_img = orig_subbed[..., idx]
        orig = orig_images[..., idx]

        if prev_row is not None:
            velocity_angle = calc_velocity_angle(
                (col, row), (prev_col, prev_row))

        prev.append((row, col))

        col = int(np.floor(col))
        row = int(np.floor(row))

        rot_rectangle = ((col, row), (height, width), 0)
        box = cv2.boxPoints(rot_rectangle)
        [x, y, _, _] = cv2.boundingRect(box)

        padded = cv2.copyMakeBorder(
            debug_img, height, height, width, width, cv2.BORDER_CONSTANT, value=0)
        yield (row,col), velocity_angle, padded[y+height:y+2*height, x+width:x+2*width], orig



def preprocess_video(video_path: Path, mask_height: int, mask_width: int, batch_size=200, num_batches=1):
    frames = []
    orig_frames = []
    centers = []
    velocities = []

    for images, image_3d in get_frames(video_path, batch_size=batch_size, max_frames=batch_size*num_batches, transform=transform_image):
        for center, velocity, frame, orig_frame in crop_video(images, image_3d, mask_height, mask_width):
            frame = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_GRAY2BGR)
            orig_frame = cv2.cvtColor(orig_frame.astype('uint8'), cv2.COLOR_GRAY2BGR)
            frames.append(frame)
            orig_frames.append(orig_frame)
            centers.append(center)
            velocities.append(velocity)
    
    return orig_frames, frames, centers, velocities



def process_video(video_writer, weight_path, orig_frames, frames, centers, velocities):
    for chunk in chunks(list(zip(frames, centers, velocities, orig_frames)), 50):
        chunk_x, chunk_center, chunk_velocity, chunk_orig = zip(*chunk)
        prob = predict(weight_path, np.asarray(chunk_x))
        for f, center, velocity, f_orig in zip(prob, chunk_center, chunk_velocity, chunk_orig):
            row, col = center
            HEIGHT, WIDTH = f_orig.shape[:2]
            MASK_H, MASK_W = f.shape[:2]


            green_mask = np.zeros((MASK_H,MASK_W,3))
            mask_pred = (255*f.squeeze()).astype('uint8')
            green_mask[..., 1] = mask_pred
            green_mask = cv2.cvtColor(green_mask.astype('uint8'), cv2.COLOR_RGB2BGR)

            # # check where the mask should go
            # from IPython import embed
            # embed()

            if col + MASK_W//2 <= WIDTH: # not padded
                right_border = MASK_W
            else: # padded
                right_border = WIDTH - col + (MASK_W // 2)

            if col - MASK_W//2 > 0: # not padded
                left_border = 0
            else:
                left_border = MASK_W//2 - col

            if row + MASK_H//2 <= HEIGHT: # not padded
                bottom_border = MASK_H
            else: # padded
                bottom_border = HEIGHT - row + (MASK_H // 2)

            if row - MASK_H//2 > 0: # not padded
                top_border = 0
            else:
                top_border = MASK_H//2 - row


            assert 0 <= bottom_border - top_border <= MASK_H
            assert 0 <= right_border - left_border <= MASK_W
            assert 0 <= bottom_border <= MASK_H
            assert 0 <= top_border <= MASK_H
            assert 0 <= left_border <= MASK_W
            assert 0 <= right_border <= MASK_W
            

            overlay = green_mask[top_border:bottom_border, left_border:right_border, :]
            alpha = np.ones(overlay.shape)*0.7
            patch = f_orig[row+top_border-MASK_H//2:row+bottom_border-MASK_H//2, col+left_border-MASK_W//2:col+right_border-MASK_W//2, :]
            patched_alpha = (patch*alpha + overlay*(1-alpha)).astype('uint8')
            f_orig[int(row)+top_border-MASK_H//2:int(row)+bottom_border-MASK_H//2, int(col)+left_border-MASK_W//2:int(col)+right_border-MASK_W//2, :] = patched_alpha[:]

            cv2.drawMarker(f_orig, (int(col), int(row)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            if velocity is not None:
                draw_arrow(f_orig, (col, row), velocity, 50, (0, 255, 0), 2)

            video_writer.write(f_orig)
    video_writer.release()

def main(video_path: Path, weight_path: Path, height: int, width: int, batch_size: int, num_batches: int):
    orig_frames, frames, centers, velocities = preprocess_video(video_path, height, width, batch_size, num_batches)
    DATA_DIR = Path(__file__).parent.parent / "data" / "segmented"
    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / f"{video_path.stem}_segemented.avi"
    cropped_video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'XVID'), 30, orig_frames[0].shape[:2], True)
    process_video(cropped_video, weight_path, orig_frames, frames, centers, velocities)
    print(f"Done! Segmented video written to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("video_path", help="path to video", type=Path)
    parser.add_argument("weights_path", help="path to weights", type=Path)
    parser.add_argument("--height", type=int,
                        default=300, help="height of box")
    parser.add_argument("--width", type=int,
                        default=300, help="width of box")
    parser.add_argument("-b", "--batch", help="Batch size for processing", default=200)
    parser.add_argument("-c", "--count", help="Number of batches to create", default=1)
    args = parser.parse_args()

    main(args.video_path, args.weights_path, args.height, args.width, args.batch, args.count)
