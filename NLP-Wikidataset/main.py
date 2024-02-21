from imports import tensorflow as tf
from imports import tensorflow_datasets as tfds

def load_and_prep_dataset(batch_size):
    
    train, test = tfds.load('wikipedia', split = ['train', 'test'], shuffle_files = True)
    
    train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':

    BATCH_SIZE = 256

    train, test = load_and_prep_dataset(BATCH_SIZE)
