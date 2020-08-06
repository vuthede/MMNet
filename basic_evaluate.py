import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from matting_nets.mmnet import MMNet, MMNet_arg_scope
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import time


def resize_and_padding(image, size = [256, 256], padded_max_size = 256, padding = True):
    # image: NxHxWxC tensor
    # size : [H, W] list

    def do_padding(image):
        image_dims = tf.shape(image)
        height = image_dims[1]
        width = image_dims[2]

        min_size = min(*size)
        width_aspect = tf.maximum(min_size, tf.cast(width * min_size / height, dtype=tf.int32))
        height_aspect = tf.maximum(min_size, tf.cast(height * min_size / width, dtype=tf.int32))

        image = tf.image.resize_bilinear(image, (height_aspect, width_aspect))
        image = image[:, :padded_max_size, :padded_max_size, :]

        # Pads the image on the bottom and right with zeros until it has dimensions target_height, target_width.
        image = tf.image.pad_to_bounding_box(
            image,
            offset_height=tf.maximum(padded_max_size-height_aspect, 0),
            offset_width=tf.maximum(padded_max_size-width_aspect, 0),
            target_height=padded_max_size,
            target_width=padded_max_size,
        )
        return image

    image = tf.cond(padding, lambda: do_padding(image), lambda: image)

    image = tf.image.resize_bilinear(image, (size[0], size[1]), align_corners=True)
    image = image / 255.
    return image


if __name__ == "__main__":
    image_ori = tf.placeholder(shape = [None, None, None, 3], dtype = tf.float32, name = "image_input")
    padding = tf.placeholder(tf.bool, name = "padding")

    image_size = 128
    width_multiplier = 1.0
    weight_decay = 4e-7
    use_fused_batchnorm = False
    dropout = 0.0
    padded_max_size = 400
    image_path = "/home/ubuntu/MMNet/Input.png"
    # model_path = "/home/ubuntu/MMNet/models/mmnet/MMNetModel-38000/MMNetModel-38000"
    model_path = "/home/ubuntu/MMNet/models/mmnet/MMNetModel-119000/MMNetModel-119000"

    with slim.arg_scope(MMNet_arg_scope(use_fused_batchnorm=use_fused_batchnorm,
                                                  weight_decay=weight_decay,
                                                  dropout=dropout)):
        image = resize_and_padding(image_ori, [image_size, image_size], padded_max_size, padding = padding)
        output, endpoints = MMNet(image, depth_multiplier = width_multiplier, is_training = False)
        output = tf.nn.softmax(output, name = "MMNet/output/softmax")
    sess = tf.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, model_path)
    saver.save(sess, "models/MMNetModel")

    rgb_image = cv2.imread(image_path)[:, :, ::-1].astype(np.float32)
    
    t1 = time.time()
    print(image.shape)

    res, inp = sess.run([output, image], feed_dict = {image_ori: [rgb_image], padding: False})

    print("Time: ", time.time()-t1)
    # anh Nhan's code
    res = np.squeeze(res[:, :, :, 1]) * 255
    res = np.clip(res, 0, 255).astype(np.uint8)
    res = np.tile(res[..., np.newaxis], [1, 1,3])
    inp = np.clip(inp * 255, 0, 255).astype(np.uint8)
    inp = np.squeeze(inp)[:, :, ::-1]
    res = np.concatenate([inp, res], axis = 1)
    cv2.imwrite("output_119000.png", res)


    # Author's code
    # prediction = np.squeeze(res[:, :, :, 1]) 
    # prediction_normed = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    # _output = np.squeeze(prediction_normed) * 255
    # _output = _output.round().astype(np.uint8)
    # cv2.imwrite("output_117500.png", _output)

