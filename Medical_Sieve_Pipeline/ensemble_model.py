
import pandas as pd
from Medical_Sieve_Model_Pipeline.config import config
from Medical_Sieve_Model_Pipeline import models
from sklearn.model_selection import train_test_split
from Medical_Sieve_Model_Pipeline.dataprocessing import processor as proc
from argparse import ArgumentParser
from Medical_Sieve_Model_Pipeline import estimator
import logging

_logger = logging.getLogger("Medical_Sieve_Model_Pipeline")

def run_train():
    _logger.info(f"Start training process for the model: ensemble_aspect_model")

    base_models = [config.MODEL1_NAME, config.MODEL2_NAME, config.MODEL3_NAME]
    
    # read probability prediction results of three models
    _logger.info(f"Loading training data...")
    features_df = proc.read_predictions(base_models, 
                                        config.MODEL_TRAINING_STAGE_PREDICTION_MAPPING)

    # load original training data
    training_df = proc.load_data(config.TRAINING_DATA_PATH)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(features_df, 
                                                        training_df[config.ASPECT_TARGET],
                                                        shuffle=False,
                                                        test_size=0.05)

    # model fitting
    _logger.info(f"Model fitting...")
    model = models.stacking_model_aspect_clf
    model.set_params(validation_data=(X_test, y_test))
    model.fit(X_train, y_train)
    prob_prediction = model.predict_proba(X_test)
    _logger.info(f"Finished training process for the model: ensemble_model")
    
    # model evaluation
    _logger.info(f"\n====== Model Evaluation=====")
    thresholds = [[0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55] for each in range(5)]
    thresholds_set = estimator.combinations(thresholds)
    estimator.model_evaluation(prob_prediction, y_test, thresholds_set)

def run_predict():
    _logger.info(f"Start predict process for the model: ensemble_model")

    base_models = [config.MODEL1_NAME, config.MODEL2_NAME, config.MODEL3_NAME]

    # read probability prediction results of three models
    _logger.info(f"Loading test data...")
    features_df = proc.read_predictions(base_models, 
                                        config.MODEL_PREDICTION_MAPPING)
    print("Feature dataframe shape:", features_df.shape)
    print("Feature columns:", features_df.columns)
    # load original training data
    df = proc.load_data(config.TEST_DATA_PATH)

    # load pretrained model
    model = models.ensemble_aspect_model()
    model.load_weights(config.MODEL4_PATH)

    # make prediction
    _logger.info(f"Predicting test data...")
    prob_prediction = model.predict(features_df)

    # fransform probability prediction to real prediction
    prediction = proc.transform_prediction(prob_prediction)
    print("Prediction length", len(prediction))

    # concatenate aspects with the original dataframe
    aspects_df = pd.DataFrame(prediction, columns=config.ASPECT_TARGET)
    result_df = pd.concat([df, aspects_df], axis=1)
    print("final dataframe shape", result_df.shape)

    # save result df
    _logger.info(f"Save the aspect prediction result")
    # result_df.to_csv(config.ASPECT_RESULT_PATH, index=False)
    result_df.to_json(config.ASPECT_RESULT_PATH,orient='records', lines=True)
    _logger.info(f"Finished predict process for the model: ensemble_aspect_model")


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
