from imports import tf
from imports import np
from imports import os

import func
from tokeniser import Tokeniser
from model_sentence import LSTM


# for the sentence dataset, mean sentence len is 140, 95 percentile is 264, 99 percentile 356
# for the wikipedia dataset, mean article len is 3144, median 1421, 75 percentile 3293, 80 percentile is 4038, 85 percentile is 5139 (OOM), 90 percentile is 6988 (OOM), 95 percentile is 11184 (gives OOM error), 99 percentile 28025, max is 463680

# code used for this: (unbatched data)
# i = 0
# mean = np.empty(6672479)
# for elem in raw_data:
#     mean[i] = len(elem.numpy())
#     i += 1
# breakpoint() # execute here: np.median(mean), np.percentile(mean, perc), np.mean(mean)
def get_model(layers, dim, pad_size, max_tokens):
    
    model = LSTM(layer_units = layers, embed_size = pad_size, max_tokens = max_tokens, output_dim = dim)
    model.lazy_setter()
    model.info()
    tf.print()
        
    return model


if __name__ == '__main__':

    #### split data 
    #### remove last non -null element to unk token
    
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # some very interesting setup stuff
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_physical_devices('GPU')


    # if gpus:
    #     print("Num GPUs Available: ", len(gpus))
    #     try:
    #         strategy = tf.distribute.MirroredStrategy()
    #         for gpu in gpus:
    #             # Allocate 6.5 GB of memory to each GPU
    #             tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit = 7168)])
    #             
    #             # set memory_growth = True 
    #             # tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialised
    #         print(e)

    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()



    # here the actual program starts: setting Hyperparameters
    EPOCHS = 16
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    MAX_TOKENS = 10000
    PAD_SIZE = 264

    # since we're using uint16, for memory reasons, we must make sure max tokens doesn't exceed the max value of uint16
    if MAX_TOKENS > tf.uint16.max: raise ValueError(f'The variable \'MAX_TOKENS\' (value: {MAX_TOKENS}) exceeds the maximum value of a uint16 ({tf.uint16.max}).')

    # get the data for the vocabulary
    raw_data = func.get_sentence_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
    # initialise the Tokeniser
    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)

    ####### code to get tokeniser ######
    # with tf.device('GPU:0'):
    #     tokeniser.adapt(raw_data)
    # tokeniser.builder()
    # tokeniser.save_layer(f'NLP-Wikidataset/model/layer/sentence_{MAX_TOKENS}/')
    # tokeniser.save_to_file(f'sentence_{int(MAX_TOKENS/1000)}k.keras')
    
    ###### code to load pre-tokenised model ######
    tokeniser.load_layer(f'NLP-Wikidataset/model/layer/sentence_{MAX_TOKENS}/')
    tokeniser.load_from_file(f'sentence_{int(MAX_TOKENS/1000)}k.keras')
            
    num_data, targets = func.targenise(raw_data, tokeniser, max_tokens = MAX_TOKENS, padding = PAD_SIZE, pad_val = 0, batch_size = BATCH_SIZE)
    del raw_data    
    
    start = [16, 8, 0, 0]
    layer_units = [[64 for _ in range(6)], [64 for _ in range(6)], [64, 64, 64], [64, 64, 64]] # half and quarter of gpt2-hidden layers
    embed_dim = [384, 192, 384, 192] # half and quarter of gpt2-embedding
    index = 1
    
    ###################### continue training at 192, 64*6, epoch 8


           
    model = get_model(layer_units[index], embed_dim[index], PAD_SIZE, MAX_TOKENS)    
   
    model.load_from_file(name = f'Sentence_Epoch{start[index]-1}_{layer_units[index]}_{embed_dim[index]}', path = 'NLP-Wikidataset/model/LSTM/training_checkpoints')

    print(f'Continue training from Epoch {start[index]}...')

    with tf.device('GPU:0'):
        model.training(num_data, targets, EPOCHS, extension = f'{layer_units[index]}_{embed_dim[index]}', start = start[index], text_file = 'NLP-Wikidataset/model/LSTM/acc.txt')

    model.save_to_file(f'trained_sentence_{layer_units[index]}_{embed_dim[index]}')
  
    with open('NLP-Wikidataset/model/LSTM/acc.txt', 'a') as file:
        file.write(f'Model of struct {layer_units[index]}\nand Embeddings of {embed_dim[index]}:\n  Accuracy: {model.acc}\n  Loss: {model.loss}\n')
    
    
    breakpoint()

    # a = np.zeros((1,100), np.int32)
    # a[0][0] = 15
    # print(a)
    
    # result = model(a)
    # b = func.make_readable(result)
    
    # try:
    #     d = func.string_from_token(b[0], tokeniser.layer.get_vocabulary())
    # except:
    #     pass
    
    

    # breakpoint()