import numpy as np
from Medical_Sieve_Model_Pipeline.dataprocessing import processor as proc
from Medical_Sieve_Model_Pipeline.config import config
from sklearn.model_selection import train_test_split
from Medical_Sieve_Model_Pipeline import models
from keras import backend as K
from argparse import ArgumentParser
import logging
from Medical_Sieve_Model_Pipeline import estimator
_logger = logging.getLogger("Medical_Sieve_Model_Pipeline")

# debbug
# import pandas as pd

def run_train():
    _logger.info("Start training process for the model: pooled_rnn_text_cnn_aspect_model")

    # generate embedding model
    _logger.info("Training tokenizer...")
    corpus = proc.read_corpus_file(config.DATASET_TEXT_CORPUS_FILE_PATH)
    tokenizer = proc.fit_tokenizer(config.NB_WORDS, corpus, config.TOKENIZER_FILE_PATH)

    _logger.info("Training fasttext embedding model...")
    proc.train_embedding_model(config.DATASET_TEXT_CORPUS_FILE_PATH, 
                               config.EMBEDDING_DIM, 
                               config.EMBEDDING_NAME,
                               config.EMBEDDING_MODEL_PATH) 
    embedding_matrix = proc.build_embedding_matrix(tokenizer.word_index, 
                                config.EMBEDDING_DIM, 
                                config.EMBEDDING_MODEL_PATH, 
                                config.EMBEDDING_MATRIX_PATH)
    
    # load training data
    _logger.info("Loading training data...")
    df = proc.load_data(config.TRAINING_DATA_PATH)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(df[config.FEATURES], 
                                                        df[config.ASPECT_TARGET],
                                                        shuffle=False,
                                                        test_size=0.05)

    _logger.info("Transforming data...")                   
    # transform train set
    X_train = proc.text_to_sequence_transformer(X_train, tokenizer)
    X_train = proc.padding_sequence_transformer(X_train, config.MAX_SEQUENCE_LEN)

    # transform test set
    X_test = proc.text_to_sequence_transformer(X_test, tokenizer)
    X_test = proc.padding_sequence_transformer(X_test, config.MAX_SEQUENCE_LEN)


    # 5-afold training and prediction for the level1 data
    _logger.info("5-fold training and do prediction for level1 training data...")
    training_prediction = proc.training_part_features_generation_for_stacking(config.MODEL2_NAME, X_train, y_train, embedding_matrix)
    
    # train the model on the training data and predict the test data
    _logger.info("Train the model on the training data and do prediction on the test data ... ")
    K.clear_session()
    model = models.pooled_rnn_text_cnn_aspect_clf
    model.set_params(embedding_matrix=embedding_matrix,
                     validation_data=(X_test, y_test))
    test_prediction = proc.test_part_features_generation_for_stacking(model, X_train, y_train, X_test, y_test)
        
    # model evaluation
    _logger.info(f"\n====== Model Evaluation=====")
    thresholds = [[0.2, 0.25, 0.3, 0.4, 0.45, 0.5] for each in range(5)]
    thresholds_set = estimator.combinations(thresholds)
    estimator.model_evaluation(test_prediction, y_test, thresholds_set)

    # save full probability prediction of this model for stacking stage
    _logger.info("Save full probability prediction for the model")
    full_prob_prediction = np.concatenate((training_prediction, test_prediction))
    proc.save_result(full_prob_prediction, config.MODEL2_TRAINING_STAGE_OUTPUT_PATH)
    _logger.info("Finished training process for the model: pooled_rnn_text_cnn_aspect_model")

def run_predict():
    _logger.info("Start predict process for the model: pooled_rnn_text_cnn_aspect_model")

    # generate embedding model
    _logger.info("Loading trained tokenizer and fasttest embedding")
    tokenizer = proc.load_tokenizer(config.TOKENIZER_FILE_PATH)
    embedding_matrix = proc.load_embedding_matrix(config.EMBEDDING_MATRIX_PATH)

    # load test data
    _logger.info("Loading test data")
    df = proc.load_data(config.TEST_DATA_PATH)
    X = df[config.FEATURES].apply(lambda x: str(x))
    X = proc.text_to_sequence_transformer(X, tokenizer)
    X = proc.padding_sequence_transformer(X, config.MAX_SEQUENCE_LEN)

    # make the prediction
    _logger.info("Predicting test data")
    model = models.pooled_rnn_text_cnn_aspect_model(embedding_matrix=embedding_matrix)
    model.load_weights(config.MODEL2_PATH)
    prob_prediction = model.predict(X)
    
    # save the probability prediction of this model for the stacking stage 
    _logger.info("Save the probability prediction result")
    proc.save_result(prob_prediction, config.MODEL2_PREDICTION_PATH)
    _logger.info(f"Finished predict process for the model: pooled_rnn_text_cnn_aspect_model")
    
    # debugging this individual model
    # prediction = proc.transform_prediction(prob_prediction)
    # aspects_df = pd.DataFrame(prediction, columns=config.ASPECT_TARGET)
    # result_df = pd.concat([df, aspects_df], axis=1)
    # result_df.to_csv(config.RESULT_DIR + "/pooled_rnn_text_cnn_aspect_prediction.csv", index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    args = parser.parse_args()

    if args.do_train:
        run_train()

    if args.do_predict:
        run_predict()


if __name__ == '__main__':
    main()
