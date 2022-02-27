import os
import sys

import librosa
import numpy as np
import tqdm

class Feature_extract:

    def __init__(self):
        None

    def _getfeature(self, y, sr, percent = 1.0):
        length = y.shape[0]
        length = int(length * percent)

        y = y[:length]
        C = np.abs(librosa.cqt(y, sr=sr))
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

        y_cp = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_cp, sr=sr)

        # # hop_length = 512
        # oenv = librosa.onset.onset_strength(y=y, sr=sr)
        # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        # print(tempogram.shape)
        contrast = librosa.feature.spectral_contrast(S=C, sr=sr)

        zero_cross = librosa.feature.zero_crossing_rate(y)

        power = librosa.feature.rms(y=y)

        features = np.concatenate([power, zero_cross, contrast, chroma_cq, tonnetz], axis=0)

        return np.average(features, axis=1)


    def process(self, music_folder,
                class_limit = 10,
                percent = 1.0):
        """Preprocesses musics in the given folder.
        """

        self._music_folder = music_folder
        self._pose_class_names = sorted(
            [n for n in os.listdir(music_folder) if not n.startswith('.')
             if os.path.isdir(os.path.join(self._music_folder, n))]
        )
        if len(self._pose_class_names) == 0:
            raise FileNotFoundError

        features = []

        for pose_class_name in self._pose_class_names:
            print('Preprocessing:', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            music_folder = os.path.join(self._music_folder, pose_class_name)

            # Detect landmarks in each music and write it to a CSV file

            music_names = sorted(
                [n for n in os.listdir(music_folder) if not n.startswith('.')])

            if class_limit is not None:
                music_names = music_names[:class_limit]

            valid_music_count = 0

            # Detect pose landmarks from each music
            for music_name in tqdm.tqdm(music_names):
                music_path = os.path.join(music_folder, music_name)

                try:
                    y, sr = librosa.load(music_path)
                except:
                    print("load music failed", file=sys.stderr)
                    continue


                # Get landmarks and scale it to the same size as the input music
                feature = self._getfeature(y,sr,percent)

                features.append(feature)

        return np.array(features)



