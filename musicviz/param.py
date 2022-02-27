

class Param:
    def __init__(self):
        self.Valance_Path = './ckpt/V'
        self.Arousal_Path = './ckpt/A'
        self.chroma = './data/Extracted-Features/chroma.npy'
        self.centroid = './data/Extracted-Features/spectral-centroid.npy'
        self.rolloff = './data/Extracted-Features/spectral-rolloff.npy'
        self.mfcc = './data/Extracted-Features/mel-cepstral-coeffs.npy'
        self.timbre = './data/Extracted-Features/timbre.npy'
        self.stats = "./data/Annotations/static_annotations.csv"