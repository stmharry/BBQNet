from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

_CHANNELS = 3
_EPSILON = 1e-9


def random_int(lower, upper):
    return tf.to_int32(lower + tf.random_uniform([]) * tf.to_float(upper - lower))


def increment_variable(init=0):
    num = tf.Variable(init, dtype=tf.float32, trainable=False)
    num_ = num + 1
    with tf.control_dependencies([num.assign(num_)]):
        return tf.identity(num_)


def moving_average(value, window):
    shape = value.get_shape()

    queue_init = tf.zeros(tf.TensorShape(window).concatenate(shape), dtype=tf.float32)
    total_init = tf.zeros(shape, dtype=tf.float32)
    num_init = tf.constant(0, dtype=tf.float32)

    queue = tf.FIFOQueue(window, [tf.float32], shapes=[shape])
    total = tf.Variable(total_init, trainable=False)
    num = tf.Variable(num_init, trainable=False)

    init = tf.cond(
        tf.equal(queue.size(), 0),
        lambda: tf.group(
            queue.enqueue_many(queue_init),
            total.assign(total_init),
            num.assign(num_init)),
        lambda: tf.no_op())

    with tf.control_dependencies([init]):
        total_ = total + value - queue.dequeue()
        num_ = num + 1
        value_averaged = total_ / (tf.minimum(num_, window) + _EPSILON)

        with tf.control_dependencies([queue.enqueue([value]), total.assign(total_), num.assign(num_)]):
            return tf.identity(value_averaged)


def exponential_moving_average(value, decay=0.9):
    shape = value.get_shape()

    value_averaged = tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)
    value_averaged_ = decay * value_averaged + (1 - decay) * value
    with tf.control_dependencies([value_averaged.assign(value_averaged_)]):
        return tf.identity(value_averaged_)


def to_rgb(value):
    shape = tf.shape(value)
    channel = shape[2]

    value = tf.cond(
        tf.equal(channel, 1),
        lambda: tf.image.grayscale_to_rgb(value),
        lambda: value)

    return value


def random_resize(value, size_range):
    new_shorter_size = random_int(size_range[0], size_range[1])

    shape = tf.shape(value)
    height = shape[0]
    width = shape[1]
    height_smaller_than_width = tf.less(height, width)

    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_size, new_shorter_size * width / height),
        lambda: (new_shorter_size * height / width, new_shorter_size))

    value = tf.expand_dims(value, 0)
    value = tf.image.resize_bilinear(value, tf.pack(new_height_and_width))
    value = tf.squeeze(value, [0])
    return value


def random_crop(value, size):
    shape = tf.shape(value)
    height = shape[0]
    width = shape[1]

    offset_height = random_int(0, height - size)
    offset_width = random_int(0, width - size)

    value = tf.slice(
        value,
        tf.pack([offset_height, offset_width, 0]),
        tf.pack([size, size, -1]))
    value.set_shape([size, size, _CHANNELS])
    return value


def random_flip(value):
    value = tf.image.random_flip_left_right(value)
    value = tf.image.random_flip_up_down(value)
    return value


def random_adjust(value, max_delta, contrast_lower, contrast_upper):
    value = tf.image.random_brightness(value, max_delta=max_delta)
    value = tf.image.random_contrast(value, lower=contrast_lower, upper=contrast_upper)
    return value


def remove_mean(value, mean):
    return value - mean


class Model(object):
    def __init__(self, global_step, sess=None):
        self.global_step = global_step
        self.sess = tf.get_default_session() if sess is None else sess

    def get_value(self, value, feed_dict=None):
        return self.sess.run(value, feed_dict=feed_dict)

    def get_callback(self, callbacks):
        for callback in callbacks:
            interval = callback.get('interval', 1)
            fetch = callback.get('fetch', {})
            func = callback.get('func', None)

            yield (interval, fetch, func)

    def is_run(self, step, interval, end_step):
        return (interval > 0 and (step + 1) % interval == 0) or (interval == -1 and (step + 1) == end_step)

    def train(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(self.get_value(self.global_step), iteration, feed_dict, callbacks)

    def test(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(0, iteration, feed_dict, callbacks)

    def feed(self, local_step, iteration, feed_dict, callbacks):
        for i in xrange(local_step, local_step + iteration):
            output_dict = {}
            for (interval, fetch, func) in self.get_callback(callbacks):
                if self.is_run(i, interval, local_step + iteration):
                    output_dict.update(fetch)

            values = self.sess.run(output_dict.values(), feed_dict=feed_dict)
            output_value_dict = dict(zip(output_dict.keys(), values))

            for (interval, fetch, func) in self.get_callback(callbacks):
                if self.is_run(i, interval, local_step + iteration) and callable(func):
                    func(
                        local_step=i,
                        end_step=local_step + iteration,
                        fetch=fetch,
                        output_value_dict=output_value_dict)

    def display(self, begin, end, local_step, end_step, fetch, output_value_dict):
        step_length = np.floor(np.log10(end_step) + 1).astype(np.int32)
        format_str = '%%s %%%dd/%%%dd' % (step_length, step_length)
        print(format_str % (begin, local_step + 1, end_step), end='')
        np.set_printoptions(precision=8)
        for key in fetch.keys():
            output_value_str = ('%.8f' if output_value_dict[key].ndim == 0 else '%s') % output_value_dict[key]
            print(', %s: %s' % (key, output_value_str), end='')
        print('', end=end)
        sys.stdout.flush()

    def summary(self, summary_writer, local_step, end_step, fetch, output_value_dict):
        for key in fetch.keys():
            summary_writer.add_summary(output_value_dict[key], global_step=self.get_value(self.global_step))

    def save(self, saver, save_path, local_step, end_step, fetch, output_value_dict):
        saver.save(self.sess, save_path, global_step=self.get_value(self.global_step))
