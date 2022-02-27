import librosa
import numpy as np
from tqdm import tqdm

from musicviz.param import Param
from musicviz.Emo_CNN import Emo
from musicviz.visual_gan import vid_biggan
from musicviz.style_transfer import style_tf

import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
import multiprocessing

class visualize(object):

    def __init__(self):
        self.sr = None
        self.y = None

    def get_score(self, model):
        chroma_stft = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)

        chroma_stft = np.expand_dims(chroma_stft, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=-1)
        rolloff = rolloff[0]
        centroid = centroid[0]

        feature_len = chroma_stft.shape[1] - 200
        indices = np.arange(0, feature_len, 200)
        if indices[-1] != feature_len - 1:
            indices = np.append(indices, feature_len - 1)

        chroma = []
        cent = []
        roll = []
        mfc = []
        for ind in indices:
            chroma.append(chroma_stft[:, ind:(ind + 200)])
            mfc.append(mfcc[:, ind:ind + 200])

            cent.append(centroid[ind:(ind + 200)])
            roll.append(rolloff[ind:(ind + 200)])

        chroma = np.stack(chroma, axis=0)
        mfc = np.stack(mfc, axis=0)
        cent = np.stack(cent, axis=0)
        roll = np.stack(roll, axis=0)

        pred = model.predict([chroma, mfc, cent, roll])

        return pred, indices + 200

    def save(self, frames, outname,
             frame_length: int = 512):
        y = np.expand_dims(self.y, axis=1)
        aud = AudioArrayClip(y, fps=self.sr * 2)

        clip = mpy.ImageSequenceClip(frames, fps=self.sr / frame_length)
        clip = clip.set_audio(aud)
        clip.write_videofile(outname, audio_codec='aac')


    def run_vid(self, queue):

        print("Generate: Vid-Biggan", flush=True)
        inst = vid_biggan()
        frame = inst.generate(self.y, self.sr)
        print("generated: ", len(frame), flush=True)
        queue.put({"frames":frame})

    def run_emo(self, queue):
        print("Generate: Valence, Arousal", flush=True)

        param = Param()

        V_pred = Emo()
        V_pred.load(param.Valance_Path)

        A_pred = Emo()
        A_pred.load(param.Arousal_Path)

        V_score, index = self.get_score(V_pred.model, self.y, self.sr)
        A_score, _ = self.get_score(A_pred.model, self.y, self.sr)

        mapped = {"V": V_score,
                  "A": A_score,
                  "ind": index}
        queue.put(mapped)

    def run(self, audio_path, outname):
        self.y, self.sr = librosa.load(audio_path)

        frames = None
        V_score = None
        A_score = None
        index = None

        shared_queue = multiprocessing.Queue()

        # spawn new process for biggan for lazy execution using graph
        biggan = multiprocessing.Process(target=self.run_vid, args=(shared_queue,))
        emopred = multiprocessing.Process(target=self.run_emo, args=(shared_queue,))
        biggan.start()
        emopred.start()
        result_count = 0
        while result_count < 2:
            temp = shared_queue.get()
            if "frames" in temp:
                frames = temp["frames"]
            else:
                V_score = temp["V"]
                A_score = temp["A"]
                index = temp["ind"]
            result_count += 1

        biggan.join()
        emopred.join()

        print("Perform: Style Transfer", flush=True)

        coef = (len(frames) - 1) / index[-1]

        scaled = np.rint(index * coef).astype(int)

        window = 0
        V_cur = 0.0;
        A_cur = 0.0;

        stf = style_tf()
        c1 = stf.load_img("./c1.jpg")
        c2 = stf.load_img("./c2.jpg")
        c3 = stf.load_img("./c3.jpg")
        c4 = stf.load_img("./c4.jpg")

        stf.load_style([c1, c2, c3, c4])

        base_step = 100

        for ind, frame in enumerate(tqdm(frames)):
            if ind > index[window]:
                window += 1

            dif = index[window] - ind + 1
            V_cur += (V_score[window] - V_cur) / dif
            A_cur += (A_score[window] - A_cur) / dif

            stf.load_content(frame)
            if V_cur > 0:
                frames[ind] = np.array(stf.process(0, round(abs(V_cur[0]) * base_step)))
            elif V_cur < 0:
                frames[ind] = np.array(stf.process(1, round(abs(V_cur[0]) * base_step)))

            if A_cur > 0:
                frames[ind] = np.array(stf.process(2, round(abs(A_cur[0]) * base_step)))
            elif A_cur < 0:
                frames[ind] = np.array(stf.process(3, round(abs(A_cur[0]) * base_step)))

        self.save(frames, outname)
