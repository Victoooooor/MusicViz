import tensorflow as tf
import numpy as np
import PIL.Image
import time

class style_tf():


    
    def _tensor_to_image(self,tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def load_img(self,path_to_img, max_dim = 256):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def _vgg_layers(self,layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def _clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def _style_content_loss(self, outputs, style_ind):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[style_ind][name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self, image, style_ind):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self._style_content_loss(outputs, style_ind)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self._clip_0_1(image))

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, Parent):
            self.parent = Parent
            super(self.parent.StyleContentModel, self).__init__()
            self.vgg = self.parent._vgg_layers(self.parent.style_layers + self.parent.content_layers)
            self.style_layers = self.parent.style_layers
            self.content_layers = self.parent.content_layers
            self.num_style_layers = len(self.style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                              outputs[self.num_style_layers:])

            style_outputs = [self.parent._gram_matrix(style_output)
                             for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name: value
                          for style_name, value
                          in zip(self.style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}

    def __init__(self, style_weight = 1e-2,
                 content_weight = 1e4):

        self.image = None
        self.content_targets = None
        self.content_image = None
        self.style_image = None

        self.content_layers = ['block5_conv2']

        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.style_extractor = self._vgg_layers(self.style_layers)
        self.extractor = self.StyleContentModel(self)
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight=30

    def load_style(self, style_images):
        self.style_targets = []
        for style_image in style_images:
            self.style_targets.append(self.extractor(style_image)['style'])

    def load_content(self, content_array, max_dim = 256):
        content_image = tf.convert_to_tensor(content_array)
        img = tf.image.convert_image_dtype(content_image, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        self.content_image = img
        self.content_targets = self.extractor(self.content_image)['content']


    def process(self, style_ind, n_iters):
        if self.image is None:
            self.image = tf.Variable(self.content_image)
        else:
            self.image.assign(self.content_image)
        for m in range(n_iters):
            self.train_step(self.image, style_ind)

        return self._tensor_to_image(self.image)