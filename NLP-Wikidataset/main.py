from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from imports import np
from imports import os
import time

# tf.data.experimental.enable_debug_mode()
                         


def load_tutorial_dataset(batch_size):
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    encoder = tf.keras.layers.TextVectorization()
    new_train = train_dataset.map(lambda text, label: text)
    print(type(new_train))
    encoder.adapt(new_train)

    return encoder

def from_gpt(batch_size, max_tokens = 50000):
    start = time.time()
    print(f'Start: {start}')

    data = tfds.load('wikipedia')
    data = data['train']
    tf.print(f'After load: {time.time() - start}')

    vectorise = Tokeniser(max_tokens = max_tokens)

    def get_text(elem):
        return elem['text']

    text_data = data.map(get_text, num_parallel_calls = tf.data.AUTOTUNE)
    text_data = text_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    tf.print(f'After prefetch: {time.time() - start}')

    vectorise.adapt(text_data.take(5))
    tf.print(f'After Adapt: {time.time() - start}')

    # text_data = text_data.map(vectorise, num_parallel_calls = tf.data.AUTOTUNE) # no idea if that really works lel
    # print(f'After map: {time.time() - start}')
    
    vectorise.builder()
    # maybe do vectorisation with calass instead of in situ model
    breakpoint()

    return vectorise



if __name__ == '__main__':

    BATCH_SIZE = 512
    MAX_TOKENS = 100
    #c = load_tutorial_dataset(BATCH_SIZE)

    model = from_gpt(BATCH_SIZE)

    breakpoint()
