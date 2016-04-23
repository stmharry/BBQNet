import tensorflow as tf
import scipy.io
import os
import csv
import random

_ROOT = '/mnt/data/ImageNet/'
_PIPELINE_DIR = 'pipeline/'
_DTYPES = [tf.string, tf.int32]
_NUM_CLASS = 1000
_NUM_PARALLEL = 8

_TRAIN = 0
_VAL = 1
_TEST = 2
_PHASES = [_TRAIN, _VAL]
_NAME = {_TRAIN: 'train', _VAL: 'val'}


meta = scipy.io.loadmat(os.path.join(_ROOT, 'ILSVRC2012_devkit_t12/data/meta.mat'))
synset_ids = [key[0] for key in meta['synsets']['WNID'][:, 0]]
ilsvrc_ids = [key[0, 0] for key in meta['synsets']['ILSVRC2012_ID'][:, 0]]
synset_id_to_ilsvrc_id = dict(zip(synset_ids, ilsvrc_ids))


def get_file_object_names(phase):
    file_object_names = [os.path.join(_ROOT, _PIPELINE_DIR, '%s_files_%d.csv' % (_NAME[phase], f)) for f in xrange(_NUM_PARALLEL)]
    return file_object_names


def prefetch_files(phase):
    file_object_names = get_file_object_names(phase)
    if all([os.path.isfile(file_object_name) for file_object_name in file_object_names]):
        return

    if phase == _TRAIN:
        (file_names, labels) = ([], [])
        data_dir = os.path.join(_ROOT, 'ILSVRC2012_img_train/')
        for synset_id in sorted(os.listdir(data_dir)):
            synset_dir = os.path.join(data_dir, synset_id)
            this_file_names = sorted(os.listdir(synset_dir))

            file_names += [os.path.join(synset_dir, file_name) for file_name in this_file_names if file_name.endswith('.JPEG')]
            labels += [synset_id_to_ilsvrc_id[synset_id] - 1] * len(this_file_names)
            print('synset_id = %s' % synset_id)

    elif phase == _VAL:
        data_dir = os.path.join(_ROOT, 'ILSVRC2012_img_val')
        file_names = sorted([os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)])

        label_file = open(os.path.join(_ROOT, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'), 'r')
        labels = [int(label) - 1 for label in label_file.readlines()]
        label_file.close()

    file_objects = [open(file_object_name, 'w') for file_object_name in file_object_names]
    file_writers = [csv.writer(file_object) for file_object in file_objects]
    for (file_name, label) in zip(file_names, labels):
        file_writers[hash(file_name) % _NUM_PARALLEL].writerow([file_name, label])
    for file_object in file_objects:
        file_object.close()


def get_files(phase):
    prefetch_files(phase)

    file_object_names = get_file_object_names(phase)
    file_objects = [open(file_object_name, 'r') for file_object_name in file_object_names]
    file_readers = [csv.reader(file_object) for file_object in file_objects]

    files_list = [[(file_name, int(label)) for (file_name, label) in file_reader] for file_reader in file_readers]
    for file_object in file_objects:
        file_object.close()

    if phase == _TRAIN:
        for files in files_list:
            random.seed(1337)
            random.shuffle(files)

    return files_list


def get_values(files_list):
    values_list = []
    for files in files_list:
        file_attrs = zip(*files)
        file_queues = []
        for (file_attr, dtype) in zip(file_attrs, _DTYPES):
            queue = tf.FIFOQueue(32, dtype, shapes=[()])
            enqueue = queue.enqueue_many([list(file_attr)])
            queue_runner = tf.train.QueueRunner(queue, [enqueue])
            tf.train.add_queue_runner(queue_runner)
            file_queues += [queue]

        (file_name_queue, label_queue) = file_queues

        reader = tf.WholeFileReader()
        (key, value) = reader.read(file_name_queue)
        image = tf.image.decode_jpeg(value)
        image = tf.cast(image, tf.float32)

        label = label_queue.dequeue()
        label = tf.equal(label, tf.range(_NUM_CLASS))
        label = tf.cast(label, tf.float32)

        values_list += [(key, image, label)]
    return values_list
