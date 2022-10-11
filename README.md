# Whirl Beetle Segmentation
Video segmentation of a whirl beetle and the ripples it creates.
## Getting started
### Prerequisite

The project assumes python version 3.7 to be compatible with Google Collab. 
Install the requirements,
```bash
pip install -U pip
pip install -r requirements.txt
```

### Segmenting
To segment a video use our `src/segment_whirl_beetle.py` it accepts a video path and a weight path (there is pretrained weights in `data/weights/pretrained.h5`) the output segmented video will be saved in `data/segmented` folder.

```text
usage: segment_whirl_beetle.py [-h] [--height HEIGHT] [--width WIDTH]
                               [-b BATCH] [-c COUNT]
                               video_path weights_path

positional arguments:
  video_path            path to video
  weights_path          path to weights

optional arguments:
  -h, --help            show this help message and exit
  --height HEIGHT       height of box
  --width WIDTH         width of box
  -b BATCH, --batch BATCH
                        Batch size for processing
  -c COUNT, --count COUNT
                        Number of batches to create
```

> Note not the entire video will be processed only the amount of batches specified

## Creating training data

### Generate frames
To generate frames from a given video use our tool `src/create_training_data.py` it accepts a video path and frames will be saved in the `data/<video_name>_frames` folder.

```text
usage: create_training_data.py [-h] [--height HEIGHT] [--width WIDTH]
                               [-b BATCH] [-c COUNT]
                               video_path

positional arguments:
  video_path            path to video

optional arguments:
  -h, --help            show this help message and exit
  --height HEIGHT       height of box
  --width WIDTH         width of box
  -b BATCH, --batch BATCH
                        Batch size for processing
  -c COUNT, --count COUNT
                        Number of batches to create
```

> Note that not many frames are needed for training and this was tested only for the default HEIGHT and WIDTH.

### Tagging
To tag the ripples use our tagging tool `src/tag_data.py`, to use the tool place images to tag inside the `data/tag/data` folder. Run `src/tag_data.py` and use the following keys,
1. `MOUSE-L` - Draw.
2. `+` - Increase brush size.
3. `-` - Decrease brush size.
4. `ESC` - Finish tagging image.

> Note that closing the window will move to the next image without tagging the current image.
The following folder structure holds the tagged data,
```text
data/tag
├── cropped  - Holds the cropped image.
├── data     - Images to be tagged by the script.
├── masked   - Holds the mask for the image.
└── tagged   - Images that have been tagged by the script.
```
 
