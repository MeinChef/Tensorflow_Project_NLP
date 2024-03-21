from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from model import LSTM
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
    num_data = num_data.map(lambda x: embed(x), num_parallel_calls = tf.data.AUTOTUNE)
    target = num_data.map(lambda x: tf.roll(input = x, shift = -1, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)
    


    return tokenise, text_data

@tf.function
def embedding(max_tokens, text):
    # maybe padding before passing it to the PADding layer with
    #
    # padded_inputs = tf.keras.utils.pad_sequences(raw_inputs, padding="post") # this works only on lists, not on tensors :)
    # post is important, due to the CUDNN implementation
    he = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = 10, mask_zero = True)(text)
    return he




if __name__ == '__main__':

    #### split data 
    #### remove last non -null element to unk token
    #### write custom loss with punishment of -1 predictions, since we can't use masking with CUDA 
    
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # some very interesting setup stuff
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')


    if gpus:
        print("Num GPUs Available: ", len(gpus))
        try:
            strategy = tf.distribute.MirroredStrategy()
            # Allocate 6.5 GB of memory to each GPU
            for gpu in gpus:
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit = 6656)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialised
            print(e)

    if tf.__version__ != '2.15.0': print('You are not using Version 2.15.0 of Tensorflow. The pretokenised models are not avaliable for you :)')

    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()




    # here the actual program starts: setting Hyperparameters
    EPOCHS = 16
    BATCH_SIZE = 128 # 416 this is the highest value that works with 74154 (longest wikipedia) and embedding of 1
    BUFFER_SIZE = 1000
    MAX_TOKENS = 25000 # to use pre-tokenised model use 25k, 50k, 75k, or 100k
    PAD_SIZE = 20000

    # timer
    start = time.time()

    # get the data for the vocabulary
    raw_data = func.get_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)
    func.timer(start)

    ####### code to get tokens ######
    # with tf.device('/device:GPU0'):
    # tokeniser.adapt(raw_data)
    # tokeniser.builder()
    # tokeniser.save_to_file(f'full_{int(MAX_TOKENS/1000)}k.keras')

    ###### code to load pre-tokenised model ######
    tokeniser.load_from_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    
    func.timer(start)

    # targets could be created by rolling the vector one to the left (but that means that our last layer does need to be as large as the input)
    # or we one hot encode each token in the article (creating dimensions of [None, article_len, vocab_size]), and then reduce_max to get rid of the [article_len] dimension, 
    # leaving us with [None, vocab_size], which is recommended by like everyone - this would give us a vector with 1s at every position where the is a token in the article 
    #       (could use sum to get words that are being used e.g. twice represented as a two)
    
    num_data, targets = func.targenise(raw_data, tokeniser, MAX_TOKENS, PAD_SIZE, pad_val = 0)
    del raw_data
    func.timer(start)

    #### how we got the longest entry of the datset ####
    # longest = 0
    # for elem in num_data:
    #     if elem.shape[1] > longest: longest = elem.shape[1]
    # print(longest)
    # tf.keras.utils.split_dataset

    

    model = LSTM(layer_units = [64, 64], embed_size = PAD_SIZE, max_tokens = MAX_TOKENS, output_dim = 20)
    model.lazy_setter()
    model.info()
    tf.print()

    model.train_test(data = num_data, targets = targets, epochs = EPOCHS)


    breakpoint()