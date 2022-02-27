import numpy as np

import tensorflow as tf
from musicviz.Emo_CNN import Emo
from musicviz.param import Param
tf.random.set_seed(123)

param = Param()

chromaStore = np.load(param.chroma)
centStore = np.load(param.centroid)
rolloffStore = np.load(param.rolloff)
mfccsStore = np.load(param.mfcc)
# timbre = np.vstack(np.load(param.timbre))
stats = np.loadtxt(param.stats, delimiter=',', skiprows=1)

Y_v = stats[:,-1]
Y_a = stats[:,1]
Y_a = (Y_a - Y_a.mean())/Y_a.std()

DATASET_SIZE = Y_v.shape[0]

BATCH_SIZE = 16

dataset_x = tf.data.Dataset.from_tensor_slices((chromaStore, mfccsStore, rolloffStore, centStore)) #, timbre))
dataset_yv = tf.data.Dataset.from_tensor_slices(Y_v)
dataset_ya = tf.data.Dataset.from_tensor_slices(Y_a)

Valence_pred = Emo()
Arousal_pred = Emo()


Valence_pred.fit(dataset_x, dataset_yv, DATASET_SIZE, b_size = BATCH_SIZE, ckpt = param.Valance_Path, pat = 10)
Arousal_pred.fit(dataset_x, dataset_ya, DATASET_SIZE, b_size = BATCH_SIZE, ckpt = param.Arousal_Path, pat = 30)
#
print(Valence_pred.model.predict([chromaStore[:10], mfccsStore[:10], rolloffStore[:10], centStore[:10]]))
print(Y_v[:10])

print(Arousal_pred.model.predict([chromaStore[:10], mfccsStore[:10], rolloffStore[:10], centStore[:10]]))
print(Y_a[:10])

