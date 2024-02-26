from imports import tf
from imports import tfds
from tokeniser import Tokeniser
from imports import np

tf.data.experimental.enable_debug_mode()


def load_and_prep_dataset(batch_size):
    
    data = tfds.load('wikipedia', split = tfds.Split.TRAIN)
    
    def article_to_text(text):
        return np.array([char for char in text.numpy().decode('utf-8')])

# Converting each dataset item to a string ('text') instead of a dictionary ({'text', 'title'}).
    data_text = data.map(lambda article: tf.py_function(func=article_to_text, inp=[article['text']], Tout=tf.string))


    for text in data_text.take(2):
        print(text.numpy())
        print('\n')


    #  #### text = [example['text'].numpy().decode('utf-8') for example in text['train']] # this step seems utterly ridicolous, converting a dataset into a list???? like wtf
    # let's not do that, RAM overflow :)

    date = data['train']
    print(type(date))

    text = date.batch(batch_size).prefetch(tf.data.AUTOTUNE)



    tokenise = tf.keras.layers.TextVectorization()
    hopf_words = text.map(lambda words: words)
    print(type(hopf_words))

    tokenise.adapt(text)




    
    


    # test = ['word', 'foo', 'bar', 'taz']
    # for body, title in text:
    #     tokenise(body)
# 
    # print(tokenise.get_vocabulary())
    
    
    #train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return 


# use the old tokeniser instead of the new layer, maybe that works better :MaiShrug:
def load_and_tokenise(batch_size):
    text = tfds.load('wikipedia')

    tokeniser = tf.keras.preprocessing.text.Tokenizer()
    tokeniser.fit_on_sequences(text)

    print(type(tokeniser))


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






#   # do we want to get chars or rather words?????
#       # I concluded this is uneccesary.
#   def id_from_chr(char):
#       return tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token = None)(char)
#   
#   def char_from_id(id):
#       return tf.keras.layers.StringLookup(vocabulary = list(vocab), invert = True, mask_token = None)(id)
#   
#   def text_from_ids(ids):
#       return tf.strings.reduce_join(char_from_id(ids), axis = -1)

# rather, we want to train a model on "vocabulary"
# see tokenise

if __name__ == '__main__':

    BATCH_SIZE = 4
    #c = load_tutorial_dataset(BATCH_SIZE)

    a = load_and_tokenise(BATCH_SIZE)
    print(type(a))
