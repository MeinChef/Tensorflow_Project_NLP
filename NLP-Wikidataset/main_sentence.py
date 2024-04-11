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
# mean = np.empty(len(data))
# for elem in raw_data:
#     mean[i] = len(elem.numpy())
#     i += 1
# breakpoint() # execute here: np.median(mean), np.percentile(mean, perc), np.mean(mean)


def get_model(layers, dim, pad_size, max_tokens):
    
    model = LSTM(layer_units = layers, pad_size = pad_size, max_tokens = max_tokens, output_dim = dim)
    model.lazy_setter()
    model.info()
    tf.print()
        
    return model


if __name__ == '__main__':
    
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()


    # here the actual program starts: setting Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    MAX_TOKENS = 10000
    PAD_SIZE = 264

    # and initialising variables for training different networks
    start = [0, 0, 10, 0]
    layer_units = [[64 for _ in range(6)], [64 for _ in range(6)], [64, 64, 64], [64, 64, 64]] # half and quarter of gpt2-hidden layers
    embed_dim = [384, 192, 384, 192] # half and quarter of gpt2-embedding

    # since we're using uint16, for memory reasons, we must make sure max tokens doesn't exceed the max value of uint16
    if MAX_TOKENS > tf.uint16.max: raise ValueError(f'The variable \'MAX_TOKENS\' (value: {MAX_TOKENS}) exceeds the maximum value of a uint16 ({tf.uint16.max}).')

    # get the data for the vocabulary
    data = func.get_sentence_data(buff_size = BUFFER_SIZE, batch_size = 1024)
    # initialise the Tokeniser
    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)

    # adapt the text to tokens
    with tf.device('GPU:0'):
        tokeniser.adapt(data)
    tokeniser.builder()
    
    # save the tokeniser
    tokeniser.save_layer(f'NLP-Wikidataset/model/layer/sentence_{MAX_TOKENS}/')
    tokeniser.save_to_file(f'sentence_{int(MAX_TOKENS/1000)}k_ragged.keras')
    
    # load already adapted tokeniser
    tokeniser.load_layer(f'NLP-Wikidataset/model/layer/sentence_{MAX_TOKENS}/')
    tokeniser.load_from_file(f'sentence_{int(MAX_TOKENS/1000)}k_ragged.keras')
            
    # transform the sentences to tokens and create targets
    data = func.targenise(
        text_data = data, 
        tokeniser = tokeniser,
        max_tokens = MAX_TOKENS,
        padding = PAD_SIZE,
        pad_val = 0,
        batch_size = BATCH_SIZE,
        buff_size = BUFFER_SIZE
        )
        

    # the training loop for the different networks
    for index in range(len(start)):
        
        # initialise model with hyperparameters
        model = get_model(layer_units[index], embed_dim[index], PAD_SIZE, MAX_TOKENS)
        
        # if there has been a trained model, continue training from there   
        if start[index] != 0:
            model.load_from_file(
                name = f'Sentence_Epoch{start[index]-1}_{layer_units[index]}_{embed_dim[index]}', 
                path = 'NLP-Wikidataset/model/LSTM/training_checkpoints'
                )
            
            print(f'Continue training from Epoch {start[index]}...')
            
        # and do the actual training
        with tf.device('GPU:0'):
            model.training(
                data, 
                EPOCHS, 
                extension = f'{layer_units[index]}_{embed_dim[index]}', 
                start = start[index], 
                text_file = 'NLP-Wikidataset/model/LSTM/acc_over_one_epoch.txt'
                )
            
        # save the model
        model.save_to_file(f'trained_sentence_{layer_units[index]}_{embed_dim[index]}')
        
        # and the accuracies etc
        with open('NLP-Wikidataset/model/LSTM/acc.txt', 'a') as file:
            file.write(f'Model of struct {layer_units[index]}\nand Embeddings of {embed_dim[index]}:\n  Accuracy: {model.acc}\n  Loss: {model.loss}\n')
        

    breakpoint()
