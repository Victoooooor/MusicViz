import tensorflow as tf
import numpy as np
import PIL.Image as Image

from musicviz.style_transfer import style_tf

# content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

stf = style_tf()
img1 = Image.open('./t2.jpg')
numpydata1 = np.asarray(img1)
img2 = Image.open('./t1.jpg')
numpydata2 = np.asarray(img2)
# asarray() class is used to convert
# PIL images into NumPy arrays

# content_image = stf.load_img(content_path)
# print(type(content_image))
style_image = stf.load_img(style_path)
#
stf.load_style([style_image])

stf.load_content(numpydata1)
image = stf.process(0, 20)
image.save('./result1.jpg')
stf.load_content(numpydata2)
image = stf.process(0, 20)
image.save('./result2.jpg')

