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

## Tagging
To tag the ripples one can you our tagging tool `src/tag_data.py`, to use the tool place images to tag inside the `data/tag/data` folder. Run `src/tag_data.py` and use the following keys,
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
 
