import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

model_params = {
    'conv1': ['conv1'],
    'conv12': ['conv1', 'conv2'],
    'conv13': ['conv1', 'conv2', 'conv3'],
    'conv14': ['conv1', 'conv2', 'conv3', 'conv4'],
    'conv15': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    'all': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc'],
    'conv45+fc': ['conv4', 'conv5', 'fc'],
    'conv5+fc': ['conv5', 'fc'],
    'fc': ['fc'],
}

class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        _, _, _, channels = input_shape
        self.hidden_units = channels // self.ratio
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential([
            layers.Dense(self.hidden_units, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excitation = self.excitation(squeeze)
        excitation = tf.reshape(excitation, [-1, 1, 1, inputs.shape[-1]])
        return inputs * excitation

class ConvModule(layers.Layer):
    def __init__(self, num_res_layers, num_kernels, expand=False, activation='relu', weight_decay=0.0, **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.num_res_layers = num_res_layers
        self.num_kernels = num_kernels
        self.expand = expand
        self.activation = activation
        self.weight_decay = weight_decay

    def build(self, input_shape):
        if self.expand:
            self.expand_conv = layers.Conv2D(self.num_kernels[1], kernel_size=3, strides=1, padding='VALID',
                                             kernel_initializer=tf.initializers.GlorotUniform(),
                                             kernel_regularizer=regularizers.l2(self.weight_decay),
                                             use_bias=False)

    def call(self, inputs):
        if input_shape[-1] == self.num_kernels[0]:
            shortcut = inputs
        else:
            shortcut = layers.Conv2D(self.num_kernels[0], kernel_size=1, strides=1, padding='VALID',
                                     kernel_initializer=tf.initializers.GlorotUniform(),
                                     kernel_regularizer=regularizers.l2(self.weight_decay),
                                     use_bias=False)(inputs)

        for i in range(self.num_res_layers):
            net = layers.Conv2D(self.num_kernels[0], kernel_size=3, strides=1, padding='SAME',
                                kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.01),
                                kernel_regularizer=regularizers.l2(self.weight_decay),
                                use_bias=False)(inputs)
            if i % 2 == 1:
                net = SEBlock()(net)
                net = layers.add([net, shortcut])
                shortcut = net

        if self.expand:
            net = self.expand_conv(net)
            if self.activation == 'prelu':
                net = layers.PReLU()(net)  # Use the Keras PReLU activation
            else:
                net = layers.Activation(self.activation)(net)
            net = layers.MaxPooling2D(pool_size=2, strides=2, padding='VALID')(net)

        return net

def build_scope(images, bottleneck_layer_size, shared_modules, scope_name, shared_scope_name, weight_decay=0.0):
    get_scope = lambda x: shared_scope_name if x in shared_modules else scope_name
    with tf.name_scope(get_scope('conv1')):
        net = ConvModule(num_res_layers=0, num_kernels=[32, 64], activation='prelu', weight_decay=weight_decay)(images)
        print('module_1 shape:', [dim for dim in net.shape])
    with tf.name_scope(get_scope('conv2')):
        net = ConvModule(num_res_layers=2, num_kernels=[64, 128], activation='prelu', weight_decay=weight_decay)(net)
        print('module_2 shape:', [dim for dim in net.shape])
    with tf.name_scope(get_scope('conv3')):
        net = ConvModule(num_res_layers=4, num_kernels=[128, 256], activation='prelu', weight_decay=weight_decay)(net)
        print('module_3 shape:', [dim for dim in net.shape])
    with tf.name_scope(get_scope('conv4')):
        net = ConvModule(num_res_layers=10, num_kernels=[256, 512], expand=True, activation='prelu', weight_decay=weight_decay)(net)
        print('module_4 shape:', [dim for dim in net.shape])
    with tf.name_scope(get_scope('conv5')):
        net = ConvModule(num_res_layers=6, num_kernels=[512], activation='prelu', weight_decay=weight_decay)(net)
        print('module_5 shape:', [dim for dim in net.shape])
    with tf.name_scope(get_scope('fc')):
        net = layers.Flatten()(net)
        prelogits = layers.Dense(bottleneck_layer_size, activation=None, kernel_initializer=tf.initializers.GlorotUniform(),
                                 kernel_regularizer=regularizers.l2(weight_decay))(net)
    return prelogits

def inference(images_A, images_B, keep_probability=1.0, phase_train=True, bottleneck_layer_size=512,
              weight_decay=0.0, reuse=None, model_version=None):
    shared_modules = model_params[model_version]

    print('input shape:', [dim for dim in images_A.shape])

    prelogits_A = build_scope(images_A, bottleneck_layer_size,
                              shared_modules, "NetA", "SharedNet", weight_decay=weight_decay)
    prelogits_B = build_scope(images_B, bottleneck_layer_size,
                              shared_modules, "NetB", "SharedNet", weight_decay=weight_decay)

    return prelogits_A, prelogits_B