from imports import tensorflow as tf
from imports import tensorflow_datasets as tfds

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def load_and_prep_dataset(batch_size):
    
    text = tfds.load('wikipedia')

    # for elem in text['train']:
    #     print(elem)
    #     break
    # return

    
    tokenise = tf.keras.layers.TextVectorization()

    test = ['word', 'foo', 'bar', 'taz']
    for elem in text['train']:
        tokenise(elem['text'])
    
    print(tokenise.get_vocabulary())
    

    #train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return 

# do we want to get chars or rather words?????
    # I concluded this is uneccesary.
def id_from_chr(char):
    return tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token = None)(char)

def char_from_id(id):
    return tf.keras.layers.StringLookup(vocabulary = list(vocab), invert = True, mask_token = None)(id)

def text_from_ids(ids):
    return tf.strings.reduce_join(char_from_id(ids), axis = -1)

# rather, we want to train a model on "vocabulary"
# see tokenise

if __name__ == '__main__':

    BATCH_SIZE = 256

    global vocab 

    a = load_and_prep_dataset(BATCH_SIZE)
