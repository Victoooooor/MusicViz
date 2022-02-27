import os
import pandas as pd
import ast
import numpy as np

import librosa

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
import pickle

class feature:
    def __init__(self):
        self.genre_data = None
        self.AUDIO_PATHS = None

    def audio_paths(self,AUDIO_DIR):
        AUDIO_PATHS = []
        # iterate through all the directories with songs in them
        for path in [os.path.join(AUDIO_DIR, p)
                     for p in os.listdir(AUDIO_DIR)
                     if not p.endswith(('checksums', '.txt', '.DS_Store'))]:
            # add all songs to the list
            AUDIO_PATHS = AUDIO_PATHS + [os.path.join(path, track).replace('\\', '/') for track in os.listdir(path)]
        self.AUDIO_PATHS = AUDIO_PATHS
        return AUDIO_PATHS

    def metadata_load(self, filepath):

        filename = os.path.basename(filepath)

        if 'features' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'echonest' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'genres' in filename:
            return pd.read_csv(filepath, index_col=0)

        if 'tracks' in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                       ('track', 'genres'), ('track', 'genres_all')]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                       ('album', 'date_created'), ('album', 'date_released'),
                       ('artist', 'date_created'), ('artist', 'active_year_begin'),
                       ('artist', 'active_year_end')]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = ('small', 'medium', 'large')
            try:
                tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    pd.CategoricalDtype(categories=SUBSETS, ordered=True))
            except ValueError:
                # the categories and ordered arguments were removed in pandas 0.25
                tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    pd.CategoricalDtype(categories=SUBSETS, ordered=True))

            COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                       ('album', 'type'), ('album', 'information'),
                       ('artist', 'bio')]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype('category')

            return tracks

    def track_genre_information(self, GENRE_PATH, TRACKS_PATH, FILE_PATHS, subset):
        """
        GENRE_PATH (str): path to the csv with the genre metadata
        TRACKS_PATH (str): path to the csv with the track metadata
        FILE_PATHS (list): list of paths to the mp3 files
        subset (str): the subset of the data desired
        """
        # get the genre information
        genres = pd.read_csv(GENRE_PATH)

        # load metadata on all the tracks
        tracks = self.metadata_load(TRACKS_PATH)

        # focus on the specific subset tracks
        subset_tracks = tracks[tracks['set', 'subset'] <= subset]

        # extract track ID and genre information for each track
        subset_tracks_genre = np.array([np.array(subset_tracks.index),
                                        np.array(subset_tracks['track', 'genre_top'])]).T

        # extract track indices from the file paths
        track_indices = []
        for path in FILE_PATHS:
            track_indices.append(path.split('/')[-1].split('.')[0].lstrip('0'))

        # get the genre associated with each file path, thanks to the path ID
        track_indices = pd.DataFrame({'file_path': FILE_PATHS, 'track_id': np.array(track_indices).astype(int)})
        tracks_genre_df = pd.DataFrame({'track_id': subset_tracks_genre[:, 0], 'genre': subset_tracks_genre[:, 1]})
        track_genre_data = track_indices.merge(tracks_genre_df, how='left')

        # label classes with numbers
        encoder = LabelEncoder()
        track_genre_data['genre_nb'] = encoder.fit_transform(track_genre_data.genre)
        self.genre_data = track_genre_data
        return track_genre_data

    def compute_mfcc(self,file_path):
        x, sr = librosa.load(file_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(y=x, sr=sr)
        return mfccs

    def compute_zcr(self,file_path):
        x, sr = librosa.load(file_path, sr=None, mono=True)
        zcr = librosa.feature.zero_crossing_rate(y=x)
        return zcr

    def compute_chroma_stft(self,file_path, hop_length=512):
        x, sr = librosa.load(file_path, sr=None, mono=True)
        stft = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)
        return stft

    def compute_spectral_centroid(self,file_path):
        x, sr = librosa.load(file_path, sr=None, mono=True)
        centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
        return centroid

    def compute_spectral_rolloff(self,file_path):
        x, sr = librosa.load(file_path, sr=None, mono=True)
        rolloff = librosa.feature.spectral_rolloff(y=x + 0.01, sr=sr)
        return rolloff

    def dump_feature(self, dump_path):
        AUDIO_TRAIN, AUDIO_TEST = train_test_split(self.AUDIO_PATHS, test_size=0.2, random_state=42)
        CONVERTED_TRAIN_PATH = os.path.join(dump_path, 'train')

        if not os.path.exists(CONVERTED_TRAIN_PATH):
            os.makedirs(CONVERTED_TRAIN_PATH)
            mfcc = defaultdict(np.array)
            zcr = defaultdict(np.array)
            chroma_stft = defaultdict(np.array)
            spectral_centroid = defaultdict(np.array)
            spectral_rolloff = defaultdict(np.array)
            y = self.genre_data[self.genre_data.file_path.isin(AUDIO_TRAIN)].genre.values
            for small_path in AUDIO_TRAIN:
                try:
                    mfcc[small_path] = self.compute_mfcc(small_path)
                    zcr[small_path] = self.compute_zcr(small_path)
                    chroma_stft[small_path] = self.compute_chroma_stft(small_path)
                    spectral_centroid[small_path] = self.compute_spectral_centroid(small_path)
                    spectral_rolloff[small_path] = self.compute_spectral_rolloff(small_path)
                except:
                    print("{} - corrupt".format(small_path))
            pickle.dump(mfcc, open(CONVERTED_TRAIN_PATH + "mfcc.p", "wb"))
            pickle.dump(zcr, open(CONVERTED_TRAIN_PATH + "zcr.p", "wb"))
            pickle.dump(chroma_stft, open(CONVERTED_TRAIN_PATH + "chroma_stft.p", "wb"))
            pickle.dump(spectral_centroid, open(CONVERTED_TRAIN_PATH + "spectral_centroid.p", "wb"))
            pickle.dump(spectral_rolloff, open(CONVERTED_TRAIN_PATH + "spectral_rolloff.p", "wb"))

        CONVERTED_TEST_PATH = './data/pickle/test/'

        if not os.path.exists(CONVERTED_TEST_PATH):
            os.makedirs(CONVERTED_TEST_PATH)
            mfcc_test = defaultdict(np.array)
            zcr_test = defaultdict(np.array)
            chroma_stft_test = defaultdict(np.array)
            spectral_centroid_test = defaultdict(np.array)
            spectral_rolloff_test = defaultdict(np.array)
            y = self.genre_data[self.genre_data.file_path.isin(AUDIO_TEST)].genre.values
            for small_path in AUDIO_TEST:
                try:
                    mfcc_test[small_path] = Fea.compute_mfcc(small_path)
                    zcr_test[small_path] = Fea.compute_zcr(small_path)
                    chroma_stft_test[small_path] = Fea.compute_chroma_stft(small_path)
                    spectral_centroid_test[small_path] = Fea.compute_spectral_centroid(small_path)
                    spectral_rolloff_test[small_path] = Fea.compute_spectral_rolloff(small_path)
                except:
                    print("{} - corrupt".format(small_path))
            pickle.dump(mfcc_test, open(CONVERTED_TEST_PATH + "mfcc_test.p", "wb"))
            pickle.dump(zcr_test, open(CONVERTED_TEST_PATH + "zcr_test.p", "wb"))
            pickle.dump(chroma_stft_test, open(CONVERTED_TEST_PATH + "chroma_stft_test.p", "wb"))
            pickle.dump(spectral_centroid_test, open(CONVERTED_TEST_PATH + "spectral_centroid_test.p", "wb"))
            pickle.dump(spectral_rolloff_test, open(CONVERTED_TEST_PATH + "spectral_rolloff_test.p", "wb"))