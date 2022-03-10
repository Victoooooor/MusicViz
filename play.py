import time

import librosa
import pyfirmata
import cv2
import numpy as np
from cv2 import WINDOW_NORMAL
from ffpyplayer.player import MediaPlayer

n_channels = 6
vic_name = './data/Generated-Videos/Symphony.mp4'
if __name__ == "__main__":
    board = pyfirmata.Arduino('/dev/ttyACM0')
    m1 = board.get_pin("d:13:s")
    m2 = board.get_pin("d:12:s")
    m3 = board.get_pin("d:11:s")
    m4 = board.get_pin("d:10:s")
    m5 = board.get_pin("d:9:s")
    m6 = board.get_pin("d:2:s")


    y, sr = librosa.load(vic_name)
    chroma = librosa.feature.chroma_cqt(y = y, sr = sr) #,n_chroma = n_channels, bins_per_octave = n_channels * 3) #n_chroma = 5

    dura = librosa.get_duration(y = y, sr = sr)
    print("dura",dura)
    video = cv2.VideoCapture(vic_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    wait = round(1000/fps)


    length = dura * fps
    chroma_len = chroma.shape[1] - 1
    window_map = chroma_len / length

    player = MediaPlayer(vic_name)
    index = 0
    val = ''

    dura *= 1000

    freq1 = 0.1
    freq2 = 0.2
    freq3 = 0.3
    freq4 = 0.4
    freq5 = 0.5
    freq6 = 0.6

    start_time = None

    while val != 'eof':
        frame, val = player.get_frame()
        if val != 'eof' and frame is not None:
            img, _ = frame
            w = img.get_size()[0]
            h = img.get_size()[1]
            arr = np.uint8(np.asarray(list(img.to_bytearray()[0])).reshape(h,w,3))
            arr2 = cv2.resize(arr,(1024, 1024), interpolation = cv2.INTER_AREA)
            # cv2.namedWindow('test', WINDOW_NORMAL)
            cv2.imshow('test', arr2)
            # cv2.resizeWindow('test', 600, 600)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            cur = time.time()
            if start_time == None:
                start_time = cur
            elapsed = cur - start_time
            elapsed *= 1000
            pos = round(elapsed / dura * chroma_len)
            pos %= chroma_len
            print(pos)
            m1.write(90 * (chroma[0][pos] + chroma[1][pos]))
            m2.write(90 * (chroma[2][pos] + chroma[3][pos]))
            m3.write(90 * (chroma[4][pos] + chroma[5][pos]))
            m4.write(90 * (chroma[6][pos] + chroma[7][pos]))
            m5.write(90 * (chroma[8][pos] + chroma[9][pos]))
            m6.write(90 * (chroma[10][pos]+ chroma[11][pos]))

            index += 1
    video.release()
    cv2.destroyAllWindows()

# while True:
#     m1.write(0)
#     time.sleep(1)
#     m1.write(90)
#     time.sleep(1)




