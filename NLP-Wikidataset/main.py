from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from imports import np
from imports import os
from imports import time
import func

# tf.data.experimental.enable_debug_mode()

def from_gpt(batch_size, max_tokens = 50000):
    start = time.time()
    print(f'Start: 0')

    # loading the dataset
    data = tfds.load('wikipedia')
    data = data['train']
    tf.print(f'After load: {time.time() - start}')

    # preparing the vectorisation layer
    tokenise = Tokeniser(max_tokens = max_tokens)

    # preparing the dataset, to make it workable


    text_data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)
    text_data = text_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    tf.print(f'After prefetch: {time.time() - start}')

    # feeding it the vectorisation layer
    tokenise.adapt(text_data.take(25))
    tf.print(f'After Adapt: {time.time() - start}')

    tokenise.builder()
    num_data = text_data.map(lambda x: tokenise(x), num_parallel_calls = tf.data.AUTOTUNE) 
    # that should turn the text into numbers - I think

    # print(f'After map: {time.time() - start}')
    

    embed = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = 10, mask_zero = True)
    embed_data = num_data.map(lambda x: embed(x), num_parallel_calls = tf.data.AUTOTUNE)
    target = num_data.map(lambda x: tf.roll(input = x, shift = -1, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)
    


    return tokenise, text_data

@tf.function
def embedding(max_tokens, text):
    # maybe padding before passing it to the embedding layer with
    #
    # padded_inputs = tf.keras.utils.pad_sequences(raw_inputs, padding="post") 
    # post is important, due to the CUDNN implementation
    embed = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = 10, mask_zero = True)(text)
    return embed




if __name__ == '__main__':

    #### split data 
    # increase swap

    BATCH_SIZE = 512
    BUFFER_SIZE = 1000
    MAX_TOKENS = 50000

    func.check_cwd()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    start = time.time()

    raw_data = func.get_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
    func.timer(start)

    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)
    tokeniser.adapt(raw_data.take(20000))
    tokeniser.builder()
    func.timer(start)

    # num_data, targets = func.targenise(raw_data, tokeniser)



    # tokenise_model, data = from_gpt(BATCH_SIZE, MAX_TOKENS)
    
    breakpoint()