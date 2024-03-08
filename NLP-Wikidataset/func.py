from imports import os

def check_cwd():
    '''
    Checks if the current working directory is correct. Promts to correct, if otherwise.
    '''

    # check if current working directory is the repository, and set it if not
    cwd = os.getcwd()

    # because I'm lazy and don't want to do the while loop every time I save the model:
    if 'Documents/Uni/Semester3/IANNwTF/Code' in cwd: 
        os.chdir('Tensorflow_Project_NLP')
        cwd = os.getcwd()

    # for everyone but me, muhahaha
    while not 'Tensorflow_Project_NLP' in cwd:
        print('The current working directory is not the repository "Tensorflow_Project_NLP".')
        print(f'You\'re currently here: {cwd}')
        new_path = input('Please navigate to the repository: ')
        
        try: os.chdir(new_path)
        except: print('This didn\'t work')
        cwd = os.getcwd()