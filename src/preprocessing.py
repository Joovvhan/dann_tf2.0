import numpy as np
import tensorflow as tf

TRAIN_NUM = 60000
TEST_NUM = 9000

def pad_image(x, y):
    
    paddings = tf.constant([[2, 2,], [2, 2]])
    
    new_x = tf.pad(x, paddings, "CONSTANT")
    
    return (new_x, y)

def duplicate_channel(x, y):

    new_x = tf.stack([x, x, x], axis = -1)
    
    return (new_x, y)

def cast(x, y):
    new_x = tf.cast(x, tf.float32)
    
    return new_x, y


def load_data(data_category):
    
    if (data_category == 'MNIST'):
        x_train = np.load('../data/mnist/x_train.npy')
        y_train = np.load('../data/mnist/y_train.npy')

        x_test = np.load('../data/mnist/x_test.npy')
        y_test = np.load('../data/mnist/y_test.npy')
                
    elif (data_category == 'SVHN'):
        x_train = np.load('../data/svhn/x_train.npy')
        y_train = np.load('../data/svhn/y_train.npy')

        x_test = np.load('../data/svhn/x_test.npy')
        y_test = np.load('../data/svhn/y_test.npy')
        
    elif (data_category == 'SYN'):
        x_train = np.load('../data/syn_num/x_train.npy')
        y_train = np.load('../data/syn_num/y_train.npy')

        x_test = np.load('../data/syn_num/x_test.npy')
        y_test = np.load('../data/syn_num/y_test.npy')


    x_train = x_train[:TRAIN_NUM] / 255.0
    y_train = y_train[:TRAIN_NUM]

    x_test = x_test[:TEST_NUM] / 255.0
    y_test = y_test[:TEST_NUM]

    return (x_train, y_train, x_test, y_test)

def data2dataset(x, y, data_category):

    if (data_category == 'MNIST'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(pad_image)
        dataset = dataset.map(duplicate_channel)
        dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))

    elif (data_category == 'SVHN' or data_category == 'SYN'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))

    return dataset


def prepare_dataset(source, target):

    (x_train, y_train, x_test, y_test) = load_data(source)

    (x_target, y_target, x_test_target, y_test_target) = load_data(target)

    source_train_dataset = data2dataset(x_train, y_train, source)
    source_test_dataset = data2dataset(x_test, y_test, source)
    target_dataset = data2dataset(x_target, y_target, target)
    target_test_dataset = data2dataset(x_test_target, y_test_target, target)

    source_train_dataset = source_train_dataset.batch(300)
    source_train_dataset = source_train_dataset.prefetch(30)

    source_test_dataset = source_test_dataset.batch(300)
    source_test_dataset = source_test_dataset.prefetch(30)

    target_dataset = target_dataset.batch(300)
    target_dataset = target_dataset.prefetch(30)

    target_test_dataset = target_test_dataset.batch(300)
    target_test_dataset = target_test_dataset.prefetch(30)

    return (source_train_dataset, source_test_dataset, target_dataset, target_test_dataset)

def prepare_dataset_single(data_category):

    (x_train, y_train, x_test, y_test) = load_data(data_category)

    train_dataset = data2dataset(x_train, y_train, data_category)
    test_dataset = data2dataset(x_test, y_test, data_category)

    train_dataset = train_dataset.batch(300)
    train_dataset = train_dataset.prefetch(1)

    test_dataset = test_dataset.batch(300)
    test_dataset = test_dataset.prefetch(1)

    return (train_dataset, test_dataset)


