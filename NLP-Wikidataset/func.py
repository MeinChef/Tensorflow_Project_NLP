from imports import os

def check_cwd():
    '''
    Checks if the current working directory is correct. Promts to correct, if otherwise.
    '''

    # check if current working directory is the repository, and set it if not
    cwd = os.getcwd()

    # for everyone but me, muhahaha
    while cwd[-22:] != 'Tensorflow_Project_NLP':
        print('The current working directory is not the top level folder of the repository "Tensorflow_Project_NLP".')
        print(f'You\'re currently here: {cwd}')
        new_path = input('Please navigate to the repository: ')
        
        try: os.chdir(new_path)
        except: print('This didn\'t work, please try again. \n\r')
        cwd = os.getcwd()