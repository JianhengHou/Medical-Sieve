from config import config
from dataprocessing import dataManager as dm
import shutil
import os

if __name__ == '__main__':
    dataset_dictionary_path = config.RAW_DATASET_DIR
    test_data_path = config.TEST_DATA_PATH
    dataset_text_corpus_file_path = config.DATASET_TEXT_CORPUS_FILE_PATH
    predict_stage_output_dir = config.PREDICT_STAGE_OUTPUT_DIR
    
    try:
        os.remove(test_data_path)
    except:
        pass
    try:
        os.remove(dataset_text_corpus_file_path)
    except:
        pass

    datamanager = dm.dataManager(config.CONTRACTION_MAPPING, 
                              config.DONMAIN_TERM_MAPPING, 
                              config.FIELDS)
    datamanager.run_separator(dataset_dictionary_path, 
                              test_data_path, 
                              dataset_text_corpus_file_path)
    try:
        shutil.rmtree(predict_stage_output_dir)
        os.mkdir(predict_stage_output_dir)
    except:
        os.mkdir(predict_stage_output_dir)
