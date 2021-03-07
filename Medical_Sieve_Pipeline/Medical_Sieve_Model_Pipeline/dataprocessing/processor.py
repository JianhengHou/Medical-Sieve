import joblib
from ..config import config
from .. import models
import fasttext
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras import backend as K
from pathlib import Path

import logging

_logger = logging.getLogger(__name__)

def read_corpus_file(corpus_text_file_path):
    whole_text = []
    with open(corpus_text_file_path, 'r') as corpus_file:
        for each in corpus_file.readlines():
            whole_text.append(each.strip())
    return whole_text

def load_data(data_file_path):
    df = pd.read_csv(data_file_path)
    return df

def fit_tokenizer(num_words, corpus_text, tokenizer_path):
    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except:
        tokenizer = Tokenizer(num_words=num_words, lower=True, filters='"#()*+-/:;<=>?@[\\]^_`{|}~\t\n')
        # tokenizer = Tokenizer(num_words=num_words, lower=True)
        tokenizer.fit_on_texts(corpus_text)
        joblib.dump(tokenizer, tokenizer_path)
    return tokenizer

def train_embedding_model(corpus_text_file_path, embedding_dim, embedding_model, embedding_model_file_path):
    model_path = Path(embedding_model_file_path)
    if model_path.is_file():
        pass
    else:
        model = fasttext.train_unsupervised(input=corpus_text_file_path, 
                                            model=embedding_model,
                                            dim=embedding_dim)
        model.save_model(embedding_model_file_path)
    return

def build_embedding_matrix(word_index, embedding_dim, embedding_model_path, embedding_matrix_path):
    try:
        embedding_matrix = load_embedding_matrix(embedding_matrix_path)
    except:
        embedding_model = fasttext.load_model(embedding_model_path)
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_model.get_word_vector(word)
            except:
                embedding_matrix[i] = embedding_model.get_word_vector("unknown")
        joblib.dump(embedding_matrix, embedding_matrix_path)
    return embedding_matrix

def load_tokenizer(tokenizer_path):
    return joblib.load(tokenizer_path)

def load_embedding_matrix(embedding_matrix_path):
    return joblib.load(embedding_matrix_path)

def text_to_sequence_transformer(text_data, tokenizer):
    return tokenizer.texts_to_sequences(text_data)

def padding_sequence_transformer(sequence_text_data, max_sequence_len):
    return pad_sequences(sequence_text_data, maxlen=max_sequence_len)

def save_result(prob_prediction, prediction_file_path):
    result_df = pd.DataFrame(prob_prediction, columns=config.ASPECT_TARGET)
    result_df.to_csv(prediction_file_path, index=False)
    return

def training_part_features_generation_for_stacking(model_name, X_train, y_train, embedding_matrix):
    folds = 5
    batch_size = len(X_train) / float(folds)
    training_prediction = np.empty((0, len(config.ASPECT_TARGET)))
    for i in range(folds):
        _logger.info("Fold {} Model Traning...".format(i+1))

        X_5fold_test = X_train[int(round(batch_size * i)):int(round(batch_size * (i+1)))]
        y_5fold_test = y_train[int(round(batch_size * i)):int(round(batch_size * (i+1)))]

        X_5fold_train = np.concatenate((X_train[:int(round(batch_size * i))], 
                                        X_train[int(round(batch_size * (i+1))):]))
        y_5fold_train = np.concatenate((y_train[:int(round(batch_size * i))], 
                                        y_train[int(round(batch_size * (i+1))):]))
        K.clear_session()
        if model_name == config.MODEL1_NAME:
            model = models.pooled_rnn_aspect_clf_for_fold
        elif model_name == config.MODEL2_NAME:
            model = models.pooled_rnn_text_cnn_aspect_clf_for_fold
        model.set_params(embedding_matrix=embedding_matrix, 
                         validation_data=(X_5fold_test, y_5fold_test))
        model.fit(X_5fold_train, y_5fold_train)
        fold_prediction = model.predict_proba(X_5fold_test)
        training_prediction = np.concatenate((training_prediction, fold_prediction))
    return training_prediction

def test_part_features_generation_for_stacking(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    test_prediction = model.predict_proba(X_test)
    return test_prediction

def read_predictions(selected_model_list, file_mapping):
    predictions_to_be_concatenated = []
    for model_name in selected_model_list:
        df = pd.read_csv(file_mapping[model_name])
        new_column_name_mapping = {aspect:model_name + '_'+ aspect for aspect in config.ASPECT_TARGET}
        df = df.rename(columns=new_column_name_mapping)
        predictions_to_be_concatenated.append(df)
    full_prediction = pd.concat(predictions_to_be_concatenated, axis=1)
    return full_prediction

def transform_prediction(prob_prediction):
    predict_softmax = np.zeros(prob_prediction.shape, dtype=int)
    for row_index, row in enumerate(prob_prediction):
        for index, each in enumerate(row):
            if each >= config.SOFTMAX_THRESHOLD[index]:
                predict_softmax[row_index][index] = 1
    return predict_softmax
