from imports import tensorflow as tf
from imports import tensorflow_datasets as tfds

def load_and_prep_dataset(batch_size):
    
    train, test = tfds.load('wikipedia', split = ['train', 'test'], shuffle_files = True)

    def tokenise():
        pass
    
    train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# do we want to get chars or rather words?????
def id_from_chr(char):
    return tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token = None)(char)

def char_from_id(id):
    return tf.keras.layers.StringLookup(vocabulary = list(vocab), invert = True, mask_token = None)(id)

def text_from_ids(ids):
    return tf.strings.reduce_join(char_from_id(ids), axis = -1)

if __name__ == '__main__':

    BATCH_SIZE = 256

    global vocab 
    vocab = set('a', 'b', 'c', 'd')

    train, test = load_and_prep_dataset(BATCH_SIZE)
