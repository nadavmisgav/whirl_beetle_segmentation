from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import cv2
from scipy import ndimage
from tqdm import tqdm

from utils import find_max_cc, get_frames, sub_median, transform_image


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
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / (video_path.stem + "_frames")
    OUTPUT_DIR.mkdir()

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
    parser.add_argument("-b", "--batch", help="Batch size for processing", default=200, type=int)
    parser.add_argument("-c", "--count", help="Number of batches to create", default=1, type=int)
    args = parser.parse_args()

    main(args.video_path, args.height, args.width, args.batch, args.count)
