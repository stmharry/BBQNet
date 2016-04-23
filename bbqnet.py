from __future__ import print_function

import tensorflow as tf
import numpy as np

import data
import util

_BBQ = 2
_NET_SIZE = 256
_SIZE_RANGE = (320, 480)
_BATCH_SIZE = 32
_MEAN = [123.680, 116.779, 103.939]

ROOT = '/mnt/data/'
SAVE_DIR = ROOT + 'ImageNet/model/'
SAVE_NAME = 'model'
SUMMARY_PATH = ROOT + 'ImageNet/log/'


def value_pipeline(value, phase):
    _MAX_DELTA = 63
    _CONTRAST_LOWER = 0.5
    _CONTRAST_UPPER = 1.5

    value = util.to_rgb(value)
    value = util.random_resize(value, size_range=_SIZE_RANGE)
    value = util.random_crop(value, size=_NET_SIZE)
    value = util.random_flip(value)
    if phase == data._TRAIN:
        value = util.random_adjust(value, max_delta=_MAX_DELTA, contrast_lower=_CONTRAST_LOWER, contrast_upper=_CONTRAST_UPPER)
    value = util.remove_mean(value, mean=_MEAN)
    return value


def train_pipeline(values_list):
    _MIN_AFTER_DEQUEUE = 4096  #
    _EXCESSIVE = 128

    with tf.variable_scope('root'):
        values_list = [(
            values[0],
            value_pipeline(values[1], data._TRAIN),
            values[2]) for values in values_list]

        (key, value, label) = tf.train.shuffle_batch_join(
            values_list,
            batch_size=_BATCH_SIZE,
            capacity=_MIN_AFTER_DEQUEUE + _EXCESSIVE,
            min_after_dequeue=_MIN_AFTER_DEQUEUE)

    return (key, value, label)


def test_pipeline(values_list):
    with tf.variable_scope('root'):
        values_list = [(
            values[0],
            value_pipeline(values[1], data._VAL),
            values[2]) for values in values_list]

        (key, value, label) = tf.train.batch_join(
            values_list,
            batch_size=_BATCH_SIZE,
            capacity=_BATCH_SIZE)

    return (key, value, label)


def get_channel(value):
    return value.get_shape().as_list()[-1]


def avg_pool(value, name, size, stride, padding='VALID'):
    value = tf.nn.avg_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding)
    return value


def max_pool(value, name, size, stride, padding='SAME'):
    value = tf.nn.max_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding)
    return value


def conv(value, name, size, stride, out_channel, stddev=None, trainable=True, padding='SAME'):
    with tf.variable_scope(name):
        in_channel = get_channel(value)

        if stddev is None:
            stddev = np.sqrt(2. / in_channel)

        weight = tf.get_variable(
            'weight',
            shape=size + (in_channel, out_channel),
            initializer=tf.truncated_normal_initializer(stddev=stddev),
            trainable=trainable)

        if (stride[0] > size[0]) or (stride[1] > size[1]):
            weight_ = tf.pad(weight, ((0, max(0, stride[0] - size[0])), (0, max(0, stride[1] - size[1])), (0, 0), (0, 0)))
        else:
            weight_ = weight

        bias = tf.get_variable(
            'bias',
            shape=(out_channel,),
            initializer=tf.constant_initializer(0.1),
            trainable=trainable)

        value = tf.nn.conv2d(value, weight_, strides=(1,) + stride + (1,), padding=padding, name='conv')
        value = tf.add(value, bias, name='conv-add')

    # print('[[[ %s, size=%s, stride=%s' % (weight.name, size, stride))
    # print('[[[ %s, shape=%s' % (value.name, value.get_shape()))
    return (value, weight)


def conv_relu(value, name, size, stride, out_channel, stddev=None, trainable=True, padding='SAME'):
    (value, weight) = conv(value, name, size, stride, out_channel, stddev, trainable, padding)
    value = tf.nn.relu(value)
    tf.add_to_collection('lsuv', dict(value=value, weights=[weight]))
    return value


def bbq_unit(value_strides, name, stride, out_channel_small, out_channel_large, trainable=True):
    (last_value, last_stride) = value_strides[-1]

    values = []
    weights = []
    with tf.variable_scope(name):
        # print('[ %s ' % name)
        with tf.variable_scope('branch-0'):
            # print('[[ branch-0')
            value = conv_relu(last_value, 'conv-0', size=(3, 3), stride=stride, out_channel=out_channel_small)
            value = conv_relu(value, 'conv-1', size=(1, 1), stride=(1, 1), out_channel=out_channel_large)
            (value, weight) = conv(value, 'conv-2', size=(1, 1), stride=(1, 1), out_channel=out_channel_small)
            values += [value]
            weights += [weight]
        with tf.variable_scope('branch-1'):
            # print('[[ branch-1')
            this_stride = (last_stride[0] * stride[0], last_stride[1] * stride[1])
            for bbq in xrange(min(_BBQ, len(value_strides))):
                (bbq_value, bbq_stride) = value_strides[-(bbq + 1)]
                (bbq_value, bbq_weight) = conv(
                    bbq_value,
                    'conv-%d' % bbq,
                    size=(1, 1),
                    stride=(this_stride[0] / bbq_stride[0], this_stride[1] / bbq_stride[1]),
                    out_channel=out_channel_small,
                    stddev=1e-4)
                values += [bbq_value]
                weights += [bbq_weight]

    value = tf.add_n(values)
    value = tf.nn.relu(value)

    value_strides += [(value, this_stride)]
    tf.add_to_collection('lsuv', dict(value=value, weights=weights))
    return value_strides


def softmax(value, dim):
    value = tf.exp(value - tf.reduce_max(value, reduction_indices=dim, keep_dims=True))
    value = value / tf.reduce_sum(value, reduction_indices=dim, keep_dims=True)
    return value


def get_net():
    net = {}

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    phase = tf.placeholder(tf.int32)
    net.update(dict(global_step=global_step, phase=phase))

    train_values = train_pipeline(data.get_values(files[data._TRAIN]))
    test_values = test_pipeline(data.get_values(files[data._VAL]))

    (key, value, label) = tf.case([
        (tf.equal(phase, data._TRAIN), lambda: train_values),
        (tf.equal(phase, data._VAL), lambda: test_values)], default=lambda: train_values)

    net.update(dict(key=key, value=value, label=label))

    with tf.variable_scope('subsample-2x'):
        value = conv_relu(value, 'conv-0', size=(3, 3), stride=(2, 2), out_channel=32)
        value = conv_relu(value, 'conv-1', size=(3, 3), stride=(1, 1), out_channel=32)
        value = conv_relu(value, 'conv-2', size=(3, 3), stride=(1, 1), out_channel=96)
        value = conv_relu(value, 'conv-3', size=(3, 3), stride=(1, 1), out_channel=128)

    with tf.variable_scope('subsample-4x-0'):
        with tf.variable_scope('branch-0'):
            value0 = conv_relu(value, 'conv', size=(3, 3), stride=(2, 2), out_channel=128)
        with tf.variable_scope('branch-1'):
            value1 = max_pool(value, 'pool', size=(3, 3), stride=(2, 2))
        value = tf.concat(3, [value0, value1])

    with tf.variable_scope('subsample-4x-1'):
        with tf.variable_scope('branch-0'):
            value0 = conv_relu(value, 'conv-0', size=(1, 1), stride=(1, 1), out_channel=64)
            value0 = conv_relu(value0, 'conv-1', size=(3, 3), stride=(1, 1), out_channel=128)
        with tf.variable_scope('branch-1'):
            value1 = conv_relu(value, 'conv-0', size=(1, 1), stride=(1, 1), out_channel=64)
            value1 = conv_relu(value1, 'conv-1', size=(1, 7), stride=(1, 1), out_channel=64)
            value1 = conv_relu(value1, 'conv-2', size=(7, 1), stride=(1, 1), out_channel=64)
            value1 = conv_relu(value1, 'conv-3', size=(3, 3), stride=(1, 1), out_channel=128)
        value = tf.concat(3, [value0, value1])

    with tf.variable_scope('subsample-8x'):
        with tf.variable_scope('branch-0'):
            value0 = conv_relu(value, 'conv', size=(3, 3), stride=(2, 2), out_channel=256)
        with tf.variable_scope('branch-1'):
            value1 = max_pool(value, 'pool', size=(3, 3), stride=(2, 2))
        value = tf.concat(3, [value0, value1])

    bbq_list = [(value, (1, 1))]
    with tf.variable_scope('subsample-16x'):
        for i in xrange(4):
            stride = (2, 2) if i == 0 else (1, 1)
            bbq_list = bbq_unit(bbq_list, 'bbq-%d' % i, stride=stride, out_channel_small=128, out_channel_large=512)

    with tf.variable_scope('subsample-32x'):
        for i in xrange(32):
            stride = (2, 2) if i == 0 else (1, 1)
            bbq_list = bbq_unit(bbq_list, 'bbq-%d' % i, stride=stride, out_channel_small=256, out_channel_large=1024)

    with tf.variable_scope('subsample-64x'):
        for i in xrange(4):
            stride = (2, 2) if i == 0 else (1, 1)
            bbq_list = bbq_unit(bbq_list, 'bbq-%d' % i, stride=stride, out_channel_small=512, out_channel_large=2048)

    value = bbq_list[-1][0]
    with tf.variable_scope('fc'):
        value = avg_pool(value, 'pool', size=(4, 4), stride=(1, 1))
        value = conv(value, 'fc', size=(1, 1), stride=(1, 1), out_channel=1000)[0]
        value = tf.squeeze(value, squeeze_dims=(1, 2))
        value = softmax(value, dim=1)

    loss = - tf.reduce_mean(tf.reduce_sum(label * tf.log(value + util._EPSILON), 1), 0)
    correct = tf.to_float(tf.equal(tf.argmax(label, 1), tf.argmax(value, 1)))
    acc = tf.reduce_mean(correct)

    learning_rate = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=global_step,
        decay_steps=50000,
        decay_rate=0.9,
        staircase=True)

    train = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        decay=0.9,
        epsilon=1.0).minimize(loss, global_step=global_step)

    net.update(dict(loss=loss, acc=acc, train=train))
    return net


def get_stat():
    fields = ['loss', 'acc']

    stat = {}
    for phase in data._PHASES:
        if phase == data._TRAIN:
            iteration = sum([len(file) for file in files[data._TRAIN]]) / _BATCH_SIZE
        elif phase == data._VAL:
            iteration = sum([len(file) for file in files[data._VAL]]) / _BATCH_SIZE

        raw_averages = {field: (net[field], util.moving_average(net[field], iteration)) for field in fields}

        display = {}
        display.update({'%s_raw' % field: raw_averages[field][0] for field in fields})
        display.update({'%s_avg' % field: raw_averages[field][1] for field in fields})

        summaries = []
        summaries += [tf.scalar_summary('%s_%s_raw' % (data._NAME[phase], field), raw_averages[field][0]) for field in fields]
        summaries += [tf.scalar_summary('%s_%s_avg' % (data._NAME[phase], field), raw_averages[field][1]) for field in fields]
        summary = tf.merge_summary(summaries)

        stat[phase] = dict(
            iteration=iteration,
            display=display,
            summary=summary)

    return stat


def construct_lsuv():
    for lsuv in tf.get_collection('lsuv'):
        variance = tf.nn.moments(lsuv['value'], axes=(0, 1, 2, 3))[1]
        condition = tf.greater(tf.abs(variance - 1), 0.1)

        weight_assigns = [
            weight.assign(tf.cond(
                condition,
                lambda: weight / tf.sqrt(variance),
                lambda: weight)) for weight in lsuv['weights']]

        lsuv.update(dict(variance=variance, condition=condition, weight_assigns=weight_assigns))


def run_lsuv():
    print('[ LSUV ]')
    feed_dict = {net['phase']: data._TRAIN}
    for lsuv in tf.get_collection('lsuv'):
        print('Using layer %s to tune %s' % (lsuv['value'].name, [weight.name for weight in lsuv['weights']]))
        for iteration in xrange(10):
            (variance_value, condition) = sess.run(
                [lsuv['variance'], lsuv['condition']] + lsuv['weight_assigns'], feed_dict=feed_dict)[:2]
            print('Iteration %d, variance = %.4f' % (iteration, variance_value))
            if not condition:
                break


def get_param_count():
    all_params = 0
    for variable in tf.trainable_variables():
        variable_params = np.prod(variable.get_shape().as_list())
        all_params += variable_params

        print('%s: %d' % (variable.name, variable_params))
    print ('TOTAL: %d' % all_params)


files = {phase: data.get_files(phase) for phase in data._PHASES}
net = get_net()
stat = get_stat()

sess = tf.Session()
with sess.as_default():
    model = util.Model(net['global_step'])
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=32, keep_checkpoint_every_n_hours=2)
    summary_writer = tf.train.SummaryWriter(SUMMARY_PATH)

    checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)
    if checkpoint:
        print('[ Model restored from %s ]' % checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print('[ Model initialized ]')
        construct_lsuv()
        sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners()
    if not checkpoint:
        run_lsuv()


def val(**kwargs):
    model.test(
        iteration=stat[data._VAL]['iteration'],
        feed_dict={net['phase']: data._VAL},
        callbacks=[
            dict(fetch=stat[data._VAL]['display'],
                 func=lambda **kwargs: model.display(begin='\033[2K\rVal', end='', **kwargs)),
            dict(interval=-1,
                 func=lambda **kwargs: print('')),
            dict(interval=-1,
                 fetch={'summary': stat[data._VAL]['summary']},
                 func=lambda **kwargs: model.summary(summary_writer=summary_writer, **kwargs))])


def train(iteration):
    model.train(
        iteration=iteration,
        feed_dict={net['phase']: data._TRAIN},
        callbacks=[
            dict(fetch={'train': net['train']}),
            dict(fetch=stat[data._TRAIN]['display'],
                 func=lambda **kwargs: model.display(begin='Train', end='\n', **kwargs)),
            dict(interval=10,
                 fetch={'summary': stat[data._TRAIN]['summary']},
                 func=lambda **kwargs: model.summary(summary_writer=summary_writer, **kwargs)),
            dict(interval=2500,
                 func=lambda **kwargs: model.save(saver=saver, save_path=SAVE_DIR + SAVE_NAME, **kwargs)),
            dict(interval=10000,
                 func=val)])
