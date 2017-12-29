import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word_id in test_set.get_all_Xlengths().keys():
        test_X, test_lengths = test_set.get_item_Xlengths(word_id)
        word_to_logl = {}
        for word, model in models.items():
            try:
                word_to_logl[word] = model.score(test_X, test_lengths)

            except:
                word_to_logl[word] = float("-inf")

        guess_word = max(word_to_logl, key = word_to_logl.get)

        probabilities.append(word_to_logl)
        guesses.append(guess_word)

    return probabilities, guesses
