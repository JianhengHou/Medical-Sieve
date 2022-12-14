import os
import random
import numpy as np

# PATH DEFINITION
PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATA_DIR = os.path.join(PACKAGE_ROOT, 'data')

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
EMBEDDING_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, 'embedding_model')

TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'medical_sieve_aspect_training_set.csv')

RAW_DATASET_DIR = os.path.join(DATA_DIR, 'patient-massage-board-raw-dataset')  
#RAW_DATASET_DIR = os.path.join(DATA_DIR, "raddit_message_board_data")

DATASET_TEXT_CORPUS_FILE_PATH = os.path.join(DATA_DIR, 'medical_sive_sentence_corpus.txt')
#DATASET_TEXT_CORPUS_FILE_PATH = os.path.join(DATA_DIR, 'raddit_sentence_corpus.txt')

TEST_DATA_PATH = os.path.join(DATA_DIR, 'line-level-patient-massage-board-dataset.jl')
#TEST_DATA_PATH = os.path.join(DATA_DIR, 'line-level-raddit-massage-board-dataset.jl')

RESULT_DIR = os.path.join(PACKAGE_ROOT, "result")
TRAINING_STAGE_OUTPUT_DIR = os.path.join(RESULT_DIR, "training_stage_output")
PREDICT_STAGE_OUTPUT_DIR = os.path.join(RESULT_DIR, "patient_aspect_prediction_stage_output")
# PREDICT_STAGE_OUTPUT_DIR = os.path.join(RESULT_DIR, "raddit_aspect_prediction_stage_output")
ASPECT_RESULT_FILE_NAME = f'aspect_prediction_{_version}.jl'
ASPECT_RESULT_PATH = os.path.join(RESULT_DIR, ASPECT_RESULT_FILE_NAME)

# Set a seed value
SEED_VALUE = 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED_VALUE)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED_VALUE)


#RADDIT  DATA FIELDS
'''
FIELDS = ["content_id", 
          "group",
          "post_type", 
          "poster",
          "timestamp",
          "text",
          "text_processed",
          "title",
          "url",
          "score",
          "num_comments_in_submission",
          "group_subscribers",
          "upvote_ratio"]
'''

#PATIENT DATA FIELDS
FIELDS = ["content_id", 
          "group",
          "post_type", 
          "poster",
          "timestamp",
          "text",
          "text_processed",
          "category",
          "url"]
   
FEATURES = "text_processed"

ASPECT_TARGET = ['access', 
                 'costs', 
                 'delays', 
                 'errors', 
                 'trust']

NB_WORDS = 500000
TOKENIZER_FILE_NAME = f'tokenizer_model_{_version}.pkl'
TOKENIZER_FILE_PATH = os.path.join(TRAINED_MODEL_DIR, TOKENIZER_FILE_NAME)

MAX_SEQUENCE_LEN = 128

EMBEDDING_DIM = 300
EMBEDDING_NAME = "cbow"
EMBEDDING_MODEL_FILE_NAME = f'{EMBEDDING_NAME}_embedding_model_{_version}.bin'
EMBEDDING_MODEL_PATH = os.path.join(EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_FILE_NAME)

EMBEDDING_MATRIX_FILE_NAME = f'{EMBEDDING_NAME}_embedding_matrix_{_version}.pkl'
EMBEDDING_MATRIX_PATH = os.path.join(EMBEDDING_MODEL_DIR, EMBEDDING_MATRIX_FILE_NAME)

# MODEL PERSISTING
MODEL1_NAME = "POOLED_RNN_MODEL"
MODEL2_NAME = "POOLED_RNN_TEXT_CNN_MODEL"
MODEL3_NAME = "BERT_MODEL"
MODEL4_NAME = "STACKING_MODEL"


# MODEL parameters
# Model 1 parameters
MODEL1_BATCH_SIZE = 128
MODEL1_EPOCHS = 100
MODEL1_FILE_NAME = f'{MODEL1_NAME}_{_version}.h5'
MODEL1_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL1_FILE_NAME)
MODEL1_TRAINING_STAGE_OUTPUT_NAME = f'{MODEL1_NAME}_training_stage_prediction_{_version}.jl'
MODEL1_TRAINING_STAGE_OUTPUT_PATH = os.path.join(TRAINING_STAGE_OUTPUT_DIR, MODEL1_TRAINING_STAGE_OUTPUT_NAME)
MODEL1_PREDICTION_NAME = f'{MODEL1_NAME}_prediction_{_version}.jl'
MODEL1_PREDICTION_PATH = os.path.join(PREDICT_STAGE_OUTPUT_DIR, MODEL1_PREDICTION_NAME)

# Model 2 parameters
MODEL2_BATCH_SIZE = 256
MODEL2_EPOCHS = 100
MODEL2_FILE_NAME = f'{MODEL2_NAME}_{_version}.h5'
MODEL2_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL2_FILE_NAME)
MODEL2_TRAINING_STAGE_OUTPUT_NAME = f'{MODEL2_NAME}_training_stage_prediction_{_version}.jl'
MODEL2_TRAINING_STAGE_OUTPUT_PATH = os.path.join(TRAINING_STAGE_OUTPUT_DIR, MODEL2_TRAINING_STAGE_OUTPUT_NAME)
MODEL2_PREDICTION_NAME = f'{MODEL2_NAME}_prediction_{_version}.jl'
MODEL2_PREDICTION_PATH = os.path.join(PREDICT_STAGE_OUTPUT_DIR, MODEL2_PREDICTION_NAME)

# Model 3 parameters
MODEL3_TRAINING_STAGE_OUTPUT_NAME = f'{MODEL3_NAME}_training_stage_prediction_{_version}.jl'
MODEL3_TRAINING_STAGE_OUTPUT_PATH = os.path.join(TRAINING_STAGE_OUTPUT_DIR, MODEL3_TRAINING_STAGE_OUTPUT_NAME)
MODEL3_PREDICTION_NAME = f'{MODEL3_NAME}_prediction_{_version}.jl'
MODEL3_PREDICTION_PATH = os.path.join(PREDICT_STAGE_OUTPUT_DIR, MODEL3_PREDICTION_NAME)

# Model 4 parameters
ENSEMBLE_INPUT_DIM = len(ASPECT_TARGET) * 3
MODEL4_BATCH_SIZE = 128
MODEL4_EPOCHS = 200
MODEL4_FILE_NAME = f'{MODEL4_NAME}_{_version}.h5'
MODEL4_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL4_FILE_NAME)
MODEL4_PREDICTION_NAME = f'{MODEL4_NAME}_prediction_{_version}.jl'
MODEL4_PREDICTION_PATH = os.path.join(RESULT_DIR, MODEL4_PREDICTION_NAME)
SOFTMAX_THRESHOLD = [0.4, 0.25, 0.3, 0.45, 0.3]

MODEL_TRAINING_STAGE_PREDICTION_MAPPING = {MODEL1_NAME:MODEL1_TRAINING_STAGE_OUTPUT_PATH,
                                           MODEL2_NAME:MODEL2_TRAINING_STAGE_OUTPUT_PATH,
                                           MODEL3_NAME:MODEL3_TRAINING_STAGE_OUTPUT_PATH}

MODEL_PREDICTION_MAPPING = {MODEL1_NAME:MODEL1_PREDICTION_PATH,
                            MODEL2_NAME:MODEL2_PREDICTION_PATH,
                            MODEL3_NAME:MODEL3_PREDICTION_PATH}

# Preprocesser - text preprocessing variable
CONTRACTION_MAPPING = {"ive":"i have",
                       "i've":"i have", 
                       "don't":"do not", 
                       "dont":"do not",
                       "doesn't":"does not", 
                       "doesnt":"does not",
                       "cant":"can not",
                       "can't":"can not",
                       "isn't":"is not", 
                       "isnt":"is not",
                       "ain't": "is not", 
                       "aren't": "are not", 
                       "'cause": "because", 
                       " cuz ": "because", 
                       "could've": "could have", 
                       "couldve": "could have", 
                       "couldn't": "could not", 
                       "couldnt": "could not", 
                       "didnt": "did not",
                       "didn't": "did not",  
                       "doesn't": "does not", 
                       "don't": "do not", 
                       "hadn't": "had not", 
                       "hasn't": "has not", 
                       "havent":"have not",
                       "haven't": "have not", 
                       "he'd": "he would",
                       "he'll": "he will", 
                       "he's": "he is", 
                       "he'd've": "he would have", 
                       "he'll've": "he will have",
                       "how'd": "how did", 
                       "how'd'y": "how do you", 
                       "how'll": "how will", 
                       "how's": "how is",  
                       "i'd": "i would", 
                       "i'd've": "i would have", 
                       "i'll": "i will", 
                       "i'll've": "i will have",
                       "i'm": "i am", 
                       " im " :" i am ", 
                       "i've": "i have", 
                       "i'd": "i would", 
                       "i'd've": "i would have", 
                       "i'll": "i will",  
                       "i'll've": "i will have",
                       "i'm": "i am", 
                       "i've": "i have", 
                       "isn't": "is not", 
                       "it'd": "it would", 
                       "it'd've": "it would have", 
                       "it'll": "it will", 
                       "it'll've": "it will have",
                       "it's": "it is", 
                       "let's": "let us",
                       "ma'am": "madam", 
                       "mayn't": "may not", 
                       "might've": "might have",
                       "mightn't": "might not",
                       "mightn't've": "might not have", 
                       "must've": "must have", 
                       "mustn't": "must not", 
                       "mustn't've": "must not have", 
                       "needn't": "need not", 
                       "needn't've": "need not have",
                       "o'clock": "of the clock",
                       "oughtn't": "ought not", 
                       "oughtn't've": "ought not have", 
                       "shan't": "shall not", 
                       "sha'n't": "shall not", 
                       "shan't've": "shall not have", 
                       "she'd": "she would", 
                       "she'd've": "she would have", 
                       "she'll": "she will", 
                       "she'll've": "she will have",
                       "she's": "she is", 
                       "should've": "should have", 
                       "shouldn't": "should not", 
                       "shouldn't've": "should not have", 
                       "so've": "so have",
                       "so's": "so as", 
                       "this's": "this is",
                       "that'd": "that would", 
                       "that'd've": "that would have", 
                       "that's": "that is", 
                       "there'd": "there would", 
                       "there'd've": "there would have", 
                       "there's": "there is", 
                       "here's": "here is",
                       "they'd": "they would",
                       "they'd've": "they would have", 
                       "they'll": "they will", 
                       "they'll've": "they will have", 
                       "they're": "they are", 
                       "they've": "they have", 
                       "to've": "to have",
                       "wasn't": "was not", 
                       "we'd": "we would", 
                       "we'd've": "we would have", 
                       "we'll": "we will",
                       "we'll've": "we will have", 
                       "we're": "we are", 
                       "we've": "we have",
                       "weren't": "were not", 
                       "what'll": "what will", 
                       "what'll've": "what will have", 
                       "what're": "what are",  
                       "what's": "what is", 
                       "what've": "what have", 
                       "when's": "when is", 
                       "when've": "when have", 
                       "where'd": "where did", 
                       "where's": "where is", 
                       "where've": "where have", 
                       "who'll": "who will", 
                       "who'll've": "who will have", 
                       "who's": "who is", 
                       "who've": "who have", 
                       "why's": "why is", 
                       "why've": "why have", 
                       "will've": "will have", 
                       "won't": "will not", 
                       "won't've": "will not have", 
                       "would've": "would have", 
                       "wouldn't": "would not", 
                       "wouldn't've": "would not have", 
                       "y'all": "you all", 
                       "y'all'd": "you all would",
                       "y'all'd've": "you all would have",
                       "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would",
                       "you'd've": "you would have", 
                       "you'll": "you will", 
                       "you'll've": "you will have", 
                       "you're": "you are", 
                       "you've": "you have",
                       " doesn t ": " does not ", 
                       " isn t ": " is not ", 
                       " wasn t ": " was not ", 
                       " don t ": " do not ", 
                       " haven t ": " have not " 
                       }
DONMAIN_TERM_MAPPING = {" dr.": " doctor ", 
                        " docs ":" doctor ", 
                        " doc s ": " doctor ", 
                        " dr ": " doctor ", 
                        " drs ": " doctor ", 
                        " doc ":" doctor ", 
                        " med's ": " medication ", 
                        " meds ": " medication ", 
                        " med ": " medication ", 
                        " appt ": "appointment", 
                        " appts ": "appointment"
                        }
