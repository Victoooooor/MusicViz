import librosa
import pyfirmata
import time

import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer

n_channels = 5

y, sr = librosa.load('./out.mp4')
chroma = librosa.feature.chroma_cqt(y = y, sr = sr,n_chroma = n_channels, bins_per_octave = n_channels * 3) #n_chroma = 5

dura = librosa.get_duration(y = y, sr = sr)

fps = 28
length = dura * fps
chroma_len = chroma.shape[1] - 1
window_map = chroma_len / length

board = pyfirmata.Arduino('/dev/ttyACM0')
m1 = board.get_pin("d:13:s")
m2 = board.get_pin("d:12:s")
m3 = board.get_pin("d:11:s")
m4 = board.get_pin("d:10:s")
m5 = board.get_pin("d:9:s")

video_path="./out.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
wait = round(1000/fps)
print(fps)
player = MediaPlayer(video_path)
index = 0
while True:
    grabbed, frame = video.read()
    audio_frame, val = player.get_frame()
    if not grabbed:
        print("End of video")
        break
    if cv2.waitKey(wait) & 0xFF == ord("q"):
        break
    cv2.imshow("Video", frame)
    if val != 'eof' and audio_frame is not None:
        # audio
        img, t = audio_frame

    pos = round(index * window_map)
    pos %= chroma_len
    m1.write(180*chroma[0][pos])
    m2.write(180 * chroma[1][pos])
    m3.write(180 * chroma[2][pos])
    m4.write(180 * chroma[3][pos])
    m5.write(180 * chroma[4][pos])
    index += 1
video.release()
cv2.destroyAllWindows()

# while True:
#     m1.write(0)
#     time.sleep(1)
#     m1.write(90)
#     time.sleep(1)




