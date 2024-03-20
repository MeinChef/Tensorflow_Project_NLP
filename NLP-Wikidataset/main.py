from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from model import LSTM
from imports import np
from imports import os
from imports import time
import func

tf.data.experimental.enable_debug_mode()

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
    ## maybe padding


    EPOCHS = 16
    BATCH_SIZE = 256 # 416 this is the highest value that works with 74154 (longest wikipedia)
    BUFFER_SIZE = 1000
    MAX_TOKENS = 50000 # to use pre-tokenised model use 50k, 75k, or 100k
    PAD_SIZE = 45000


    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print("Num GPUs Available: ", len(gpus))
        try:
            strategy = tf.distribute.MirroredStrategy(gpus)
            # Allocate 6.5 GB of memory to each GPU
            for gpu in gpus:
                tf.config.set_logical_device_configuration(gpu, tf.config.LogicalDeviceConfiguration(memory_limit = 6656))
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialised
            print(e)

    # do some setup stuff
    if tf.__version__ != '2.15.0': print('You are not using Version 2.15.0 of Tensorflow, I haven\'t tested it with other versions, you are on your own :)')
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()

    # here the actual fun starts
    start = time.time()

    raw_data = func.get_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)
    func.timer(start)

    # for Mini11e trials
    # raw_data = raw_data.take(20)
    # list = raw_data.map(lamda x: whatever you do with x)

    ####### code to get tokens ######
    # with tf.device('/device:GPU0'):
    #     tokeniser.adapt(raw_data)
    # tokeniser.builder()
    # tokeniser.save_to_file(f'full_{int(MAX_TOKENS/1000)}k.keras')

    ###### code to load pre-tokenised model ######
    tokeniser.load_from_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    func.timer(start)
    
    num_data, targets = func.targenise(raw_data, tokeniser, PAD_SIZE, pad_val = 0)
    del raw_data
    func.timer(start)

    # longest = 0
    # for elem in num_data:
    #     if elem.shape[1] > longest: longest = elem.shape[1]
    # print(longest)
    # tf.keras.utils.split_dataset

    

    model = LSTM(layer_units = [5], embed_size = PAD_SIZE, max_tokens = MAX_TOKENS, output_dim = 50)
    model.lazy_setter()
    model.info()

    model.train_test(data = num_data, targets = targets, epochs = EPOCHS)


    breakpoint()