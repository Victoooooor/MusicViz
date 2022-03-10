import sys
import librosa
import numpy as np
from tqdm import tqdm


from musicviz.visualize import visualize

import multiprocessing

class Param:
    def __init__(self):
        self.Valance_Path = './ckpt/V'
        self.Arousal_Path = './ckpt/A'
        self.c1_path = "./data/image/c1.jpg"
        self.c2_path = "./data/image/c2.jpg"
        self.c3_path = "./data/image/c3.jpg"
        self.c4_path = "./data/image/c4.jpg"
        self.base_step = 25

if __name__ == "__main__":

    name = './Symphony.mp3'
    # name = "./data/Test-Audio/Legends.wav"
    param = Param()
    viz = visualize()
    viz.run(name, param, "./Symphony.mp4")
