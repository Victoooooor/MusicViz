import librosa

import numpy as np

import random

import tensorflow.compat.v1 as tf

from pytorch_pretrained_biggan import truncated_noise_sample

import tensorflow_hub as hub
from tqdm import tqdm


class vid_biggan:
    def __init__(self, res=256):

        tf.disable_v2_behavior()

        self.model_name = 'biggan-deep-' + str(res)
        self.module = hub.Module('https://tfhub.dev/deepmind/{model_name}/1'.format(model_name=self.model_name))
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                       for k, v in self.module.get_input_info_dict().items()}
        self.output = self.module(self.inputs)
        self.input_z = self.inputs['z']
        self.input_y = self.inputs['y']
        self.input_trunc = self.inputs['truncation']

        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]

        initializer = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(initializer)

    def _new_jitters(self, jitter):
        jitters = np.zeros(128)
        for j in range(128):
            if random.uniform(0, 1) < 0.5:
                jitters[j] = 1
            else:
                jitters[j] = 1 - jitter
        return jitters

    # get new update directions
    def _new_update_dir(self, nv2, update_dir, truncation, tempo_sensitivity):
        for ni, n in enumerate(nv2):
            if n >= 2 * truncation - tempo_sensitivity:
                update_dir[ni] = -1

            elif n < -2 * truncation + tempo_sensitivity:
                update_dir[ni] = 1
        return update_dir

    # smooth class vectors
    def _smooth(self, class_vectors, smooth_factor):
        if smooth_factor == 1:
            return class_vectors

        class_vectors_terp = []
        for c in range(int(np.floor(len(class_vectors) / smooth_factor) - 1)):
            ci = c * smooth_factor
            cva = np.mean(class_vectors[int(ci):int(ci) + smooth_factor], axis=0)
            cvb = np.mean(class_vectors[int(ci) + smooth_factor:int(ci) + smooth_factor * 2], axis=0)

            for j in range(smooth_factor):
                cvc = cva * (1 - j / (smooth_factor - 1)) + cvb * (j / (smooth_factor - 1))
                class_vectors_terp.append(cvc)

        return np.array(class_vectors_terp)

    # normalize class vector between 0-1
    def _normalize_cv(self, cv2):
        min_class_val = min(i for i in cv2 if i != 0)
        for ci, c in enumerate(cv2):
            if c == 0:
                cv2[ci] = min_class_val
        cv2 = (cv2 - min_class_val) / np.ptp(cv2)

        return cv2

    def generate(self, y, sr,
                 frame_length: int = 512,
                 pitch_sensitivity: int = 200,
                 tempo_sensitivity: float = 0.25,
                 depth: float = 1.0,
                 num_classes: int = 12,  # set number of classes
                 jitter: float = 0.5,  # set jitter
                 truncation: float = 1.0,  # set truncation
                 batch_size: int = 30,  # set batch size
                 smooth_factor: int = 20,
                 classes=None,
                 sort_classes_by_power=False):

        pitch_sensitivity = (300 - pitch_sensitivity) * 512 / frame_length
        tempo_sensitivity = tempo_sensitivity * frame_length / 512

        # set smooth factor
        if smooth_factor > 1:
            smooth_factor = int(smooth_factor * 512 / frame_length)

        # set duration
        frame_lim = int(np.floor(len(y) / sr * 22050 / frame_length / batch_size))

        # create spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=frame_length)

        # get mean power at each time point
        specm = np.mean(spec, axis=0)

        # compute power gradient across time points
        gradm = np.gradient(specm)

        # set max to 1
        gradm = gradm / np.max(gradm)

        # set negative gradient time points to zero
        gradm = gradm.clip(min=0)

        # normalize mean power between 0-1
        specm = (specm - np.min(specm)) / np.ptp(specm)

        # create chromagram of pitches X time points
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

        # sort pitches by overall power
        chromasort = np.argsort(np.mean(chroma, axis=1))[::-1]

        if classes:

            if len(classes) not in [12, num_classes]:
                raise ValueError(
                    "The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")

        else:  # select 12 random classes
            cls1000 = list(range(1000))
            random.shuffle(cls1000)
            classes = cls1000[:12]

        if sort_classes_by_power:
            classes = [classes[s] for s in np.argsort(chromasort[:num_classes])]

        # initialize first class vector
        cv1 = np.zeros(1000)
        for pi, p in enumerate(chromasort[:num_classes]):

            if num_classes < 12:
                cv1[classes[pi]] = chroma[p][np.min([np.where(chrow > 0)[0][0] for chrow in chroma])]
            else:
                cv1[classes[p]] = chroma[p][np.min([np.where(chrow > 0)[0][0] for chrow in chroma])]

        # initialize first noise vector
        nv1 = truncated_noise_sample(truncation=truncation)[0]

        # initialize list of class and noise vectors
        class_vectors = [cv1]
        noise_vectors = [nv1]

        # initialize previous vectors (will be used to track the previous frame)
        cvlast = cv1
        nvlast = nv1

        # initialize the direction of noise vector unit updates
        update_dir = np.zeros(128)
        for ni, n in enumerate(nv1):
            if n < 0:
                update_dir[ni] = 1
            else:
                update_dir[ni] = -1

        # initialize noise unit update
        update_last = np.zeros(128)

        print('\nGenerating input vectors \n')

        for i in tqdm(range(len(gradm))):

            # print progress
            pass

            # update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
            if i % 200 == 0:
                jitters = self._new_jitters(jitter)

            # get last noise vector
            nv1 = nvlast

            # set noise vector update based on direction, sensitivity, jitter, and combination of overall power and
            # gradient of power
            update = np.array([tempo_sensitivity for k in range(128)]) * (gradm[i] + specm[i]) * update_dir * jitters

            # smooth the update with the previous update (to avoid overly sharp frame transitions)
            update = (update + update_last * 3) / 4

            # set last update
            update_last = update

            # update noise vector
            nv2 = nv1 + update

            # append to noise vectors
            noise_vectors.append(nv2)

            # set last noise vector
            nvlast = nv2

            # update the direction of noise units
            update_dir = self._new_update_dir(nv2, update_dir, truncation, tempo_sensitivity)

            # get last class vector
            cv1 = cvlast

            # generate new class vector
            cv2 = np.zeros(1000)
            for j in range(num_classes):
                cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i]) / (pitch_sensitivity))) / (
                        1 + (1 / ((pitch_sensitivity))))

            # if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
            if num_classes > 6:
                cv2 = self._normalize_cv(cv2)
            else:
                cv2 = cv2 / np.max(cv2)

            # adjust depth
            cv2 = cv2 * depth

            # this prevents rare bugs where all classes are the same value
            if np.std(cv2[np.where(cv2 != 0)]) < 0.0000001:
                cv2[classes[0]] = cv2[classes[0]] + 0.01

            # append new class vector
            class_vectors.append(cv2)

            # set last class vector
            cvlast = cv2

        # interpolate between class vectors of bin size [smooth_factor] to smooth frames
        class_vectors = self._smooth(class_vectors, smooth_factor)

        print('\n\nGenerating frames \n')

        self.frames = []

        for i in tqdm(range(frame_lim)):

            # print progress
            pass

            if (i + 1) * batch_size > len(class_vectors):
                # torch.cuda.empty_cache()
                break
            # torch.cuda.empty_cache()
            # get batch
            noise_vector = noise_vectors[i * batch_size:(i + 1) * batch_size]
            class_vector = class_vectors[i * batch_size:(i + 1) * batch_size]
            noise_vector = np.stack(noise_vector, axis=0)
            class_vector = np.stack(class_vector, axis=0)
            # # Generate images
            # with torch.no_grad():
            #     output = model(noise_vector, class_vector, truncation)
            #
            # output_cpu = output.cpu().data.numpy()

            feed_dict = {self.input_z: noise_vector, self.input_y: class_vector, self.input_trunc: truncation}
            sample = self.sess.run(self.output, feed_dict=feed_dict)

            # sample = module(dict(y=class_vector, z=noise_vector, truncation=truncation))
            # convert to image array and add to frames
            for out in sample:
                out = np.clip(((out + 1) / 2.0) * 256, 0, 255)
                out = np.uint8(out)
                self.frames.append(out)

        return self.frames
