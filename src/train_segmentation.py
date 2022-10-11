import os
import sys
from argparse import ArgumentParser
from pathlib import Path

LIB = Path(__file__).parent.parent / "lib"
FGSEGNET = LIB / "FgSegNet"
sys.path.insert(0, str(LIB.parent)) 
sys.path.insert(0, str(FGSEGNET)) 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import lib.FgSegNet.FgSegNet_v2_UCSD as FgSegNet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", help="path to cropped frames", type=Path)
    parser.add_argument("label_path", help="path to labels", type=Path)
    parser.add_argument("-e", "--epochs", help="number of epochs", default=20, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-4, type=float)
    args = parser.parse_args()

    FgSegNet.main(args.data_path, args.label_path, args.epochs, args.lr)
