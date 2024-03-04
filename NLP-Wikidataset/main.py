from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from imports import np
from imports import nltk
import time

# tf.data.experimental.enable_debug_mode()



def load_and_prep_dataset(batch_size = 64, max_tokens = 100000):
    

    data = tfds.load('wikipedia')

    data = data['train']
    tokenise = tf.keras.layers.TextVectorization(max_tokens = max_tokens)



    for elem in data:
        
        breakpoint() # useful for in code debugging

        # eager Tensor is just tensor
        tokenise.adapt(elem['text'])

    data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return tokenise

            
    # meh = data.map(lambda dictionary: dictionary['text'])

    # tokenise = Tokeniser(vocab = data['train'])

    # halp = tf.convert_to_tensor(data)
    # tokenise.adapt(halp)
    # breakpoint()

    # more = tf.convert_to_tensor(elem['text'])
    # hopf_words = text.map(lambda words: words)
    # print(type(hopf_words))    
    # tokenise.adapt_it(more)

    
    def article_to_text(text):
        return np.array([char for char in text.numpy().decode('utf-8')])

    # Converting each dataset item to a string ('text') instead of a dictionary ({'text', 'title'}).
    data_text = data.map(lambda article: tf.py_function(func=article_to_text, inp=[article['text']], Tout=tf.string))


    for text in data_text.take(2):
        print(text.numpy())
        print('\n')


    #  #### text = [example['text'].numpy().decode('utf-8') for example in text['train']] # this step seems utterly ridicolous, converting a dataset into a list???? like wtf
    # let's not do that, RAM overflow :)

    # date = data['train']
    # print(type(date))

    # text = date.batch(batch_size).prefetch(tf.data.AUTOTUNE)
           




    
    


    # test = ['word', 'foo', 'bar', 'taz']
    # for body, title in text:
    #     tokenise(body)
# 
    # print(tokenise.get_vocabulary())
    
    
    #train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return 
                         

def load_and_nltk(batch_size):
    dataset = tfds.load('wikipedia')

    text = []

    for data in dataset['train']:
        text.extend(nltk.tokenize.sent_tokenize(data['text']))
    
    print(len(text))

# use the old tokeniser instead of the new layer, maybe that works better :MaiShrug:
def load_and_tokenise(batch_size):
    data = tfds.load('wikipedia')

    i = 0
    tokeniser = tf.keras.preprocessing.text.Tokenizer()

    # data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    for elem in data['train']:
        was_anderes = tf.cast(elem['text'], dtype = tf.string)
        # print(type(elem['text']))     

        
        tokeniser.fit_on_texts(was_anderes)

        if i == 20: break
        i += 1

    print(tokeniser.word_index)

    return tokeniser


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
    print(f'After load: {time.time() - start}')

    vectorise = tf.keras.layers.TextVectorization(max_tokens = max_tokens, output_mode = 'int')

    def get_text(elem):
        return elem['text']

    text_data = data.map(get_text, num_parallel_calls = tf.data.AUTOTUNE)
    text_data = text_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print(f'After prefetch: {time.time() - start}')
    vectorise.adapt(text_data.take(5))
    print(f'After Adapt: {time.time() - start}')

    text_data = text_data.map(vectorise, num_parallel_calls = tf.data.AUTOTUNE) # no idea if that really works lel
    print(f'After map: {time.time() - start}')
    breakpoint()
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape = (1,), dtype = tf.string))
    model.add(vectorise)
    breakpoint()
    # maybe do vectorisation with calass instead of in situ model

    return model



if __name__ == '__main__':

    BATCH_SIZE = 512
    #c = load_tutorial_dataset(BATCH_SIZE)

    a = from_gpt(BATCH_SIZE)
    breakpoint()
    print(type(a))
