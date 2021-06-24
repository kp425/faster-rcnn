import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def base_layer(input_tensor = None, trainable=False):
    input_shape = (None, None, 3)
    if input_tensor == None:
        inp_layer = Input(shape= input_shape)
    else:
        inp_layer = Input(shape= input_shape, tensor= input_tensor)
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

    for i,lay in enumerate(base_model.layers):
        if i==0:
            x = lay(inp_layer)
        else:
            x = lay(x)
        if lay.name == "block4_pool":
            break
    return x
    
def rpn_layer(base_layer_out, n_anchors):
    rpn_init = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(base_layer_out)
    rpn_cls = layers.Conv2D(n_anchors, (1, 1), activation="sigmoid")(rpn_init)
    rpn_reg = layers.Conv2D(n_anchors * 4, (1, 1), activation="linear")(rpn_init)
    return [rpn_cls, rpn_reg, base_layer_out]


class RoiPooling(layers.Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
         (1, num_rois, pool_size, pool_size, channels)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.dim_ordering = 'tf'
        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]
        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            x = tf.cast(x, 'int32')
            y = tf.cast(y, 'int32')
            w = tf.cast(w, 'int32')
            h = tf.cast(h, 'int32')

            # rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.concat(outputs, axis=0)
        final_output = tf.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output


def classification_layer(base_layers, input_rois, n_rois, n_classes = 21, trainable=False):
    pooling_regions = 7
    out_roi_pool = RoiPooling(pooling_regions, n_rois)([base_layers, input_rois])
    out = layers.TimeDistributed(layers.Flatten())(out_roi_pool)
    out = layers.TimeDistributed(layers.Dense(4096, activation='relu'))(out)
    out = layers.TimeDistributed(layers.Dropout(0.5))(out)
    out = layers.TimeDistributed(layers.Dense(4096, activation='relu'))(out)
    out = layers.TimeDistributed(layers.Dropout(0.5))(out)

    out_class = layers.TimeDistributed(layers.Dense(n_classes, 
                                                    activation='softmax', 
                                                    kernel_initializer='zero'))(out)
    # note: no regression target for bg class
    out_regr = layers.TimeDistributed(layers.Dense(4 * (n_classes-1), 
                                                        activation='linear', 
                                                        kernel_initializer='zero'))(out)
    return [out_class, out_regr]
