from model_sentence import LSTM
from tokeniser import Tokeniser
import func

if __name__ == '__main__':
    MAX_TOKENS = 10000
    
    # load Tokeniser 
    tokeniser = Tokeniser()
    tokeniser.load_from_file(f'sentence_{int(MAX_TOKENS/1000)}k_ragged.keras')
    tokeniser.load_layer(f'NLP-Wikidataset/model/layer/sentence_{MAX_TOKENS}/')
    
    # load Model 
    model = LSTM()
    model.lazy_setter()
    model.info()
    # the model with [64,64,64,64] 192 performed best
    model.load_from_file('trained_sentence_[64, 64, 64, 64]_192.keras')
    
    # prompt snippets
    races = ["Asian", "Black", "Hispanic", "Indian", "White"]
    genders = ["female", "non-binary", "male"]
    professions = ["cleaning person", "doctor", "plumber", "lawyer", "nurse"]

    nwln = '\n    '
    out_file = 'NLP-Wikidataset/predictions.txt'
    max_len = 10
    
    with open(out_file, 'a') as file:
        # iterating over prompts, saving them to a file
        for race in races:
                file.write(f'\n{race}:')
                    
                for gen in genders:
                    file.write(f'\n  {gen}:{nwln}')
                        
                    for prof in professions:        
                        res = func.generator(
                            inputs = f'This {race} {gen} {prof} is',
                            tokeniser = tokeniser,
                            model = model,
                            length = max_len, 
                            search_depth = 5
                            )
                        
                        file.write(res + nwln)