import librosa
import pyfirmata

import cv2
import numpy as np
from cv2 import WINDOW_NORMAL
from ffpyplayer.player import MediaPlayer

n_channels = 5
vic_name = './out.mp4'
if __name__ == "__main__":
    board = pyfirmata.Arduino('/dev/ttyACM0')
    m1 = board.get_pin("d:13:s")
    m2 = board.get_pin("d:12:s")
    m3 = board.get_pin("d:11:s")
    m4 = board.get_pin("d:10:s")
    m5 = board.get_pin("d:9:s")

    y, sr = librosa.load(vic_name)
    chroma = librosa.feature.chroma_cqt(y = y, sr = sr,n_chroma = n_channels, bins_per_octave = n_channels * 3) #n_chroma = 5

    dura = librosa.get_duration(y = y, sr = sr)

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
            pos = round(index * window_map)
            pos %= chroma_len
            m1.write(180*chroma[0][pos])
            m2.write(720 * chroma[1][pos])
            m3.write(180 * chroma[2][pos])
            m4.write(180 * chroma[3][pos])
            m5.write(720 * chroma[4][pos])
            index += 1

    video.release()
    cv2.destroyAllWindows()

# while True:
#     m1.write(0)
#     time.sleep(1)
#     m1.write(90)
#     time.sleep(1)




