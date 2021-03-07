from config import config
from dataprocessing import dataManager as dm


if __name__ == '__main__':
    dataset_dictionary_path = config.RAW_DATASET_DIR
    test_data_path = config.TEST_DATA_PATH
    dataset_text_corpus_file_path = config.DATASET_TEXT_CORPUS_FILE_PATH

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
