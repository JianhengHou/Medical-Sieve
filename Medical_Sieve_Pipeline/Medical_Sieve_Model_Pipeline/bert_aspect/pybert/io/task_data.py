import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from ..configs.basic_config import config
import numpy as np

class TaskData(object):
    def __init__(self):
        pass
    def train_val_split(self,X, y, valid_size, stratify=False,shuffle=False,save = True,
                        seed = None,data_name = None,data_dir = None):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        logger.info('split raw data into train and valid')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        else:
            data = []
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            save_pickle(data=train,file_path=train_path)
            save_pickle(data = valid,file_path=valid_path)
        return train, valid

    def train_val_split_by_fold(self, X, y, batch_size, i, save=True):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        logger.info('split raw data into train and valid for fold {}'.format(i))
        data = []
        for step,(data_x, data_y) in enumerate(zip(X, y)):
            data.append((data_x, data_y))
            pbar(step=step)
        del X, y
        data = np.array(data)
        valid = data[int(round(batch_size * (i-1))):int(round(batch_size * i))]
        train = np.concatenate((data[:int(round(batch_size * (i-1)))], 
                                data[int(round(batch_size * i)):]))
        return train, valid

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        # data = pd.read_csv(raw_data_path)
        data = pd.read_json(raw_data_path, lines=True)
        print("There are {} rows in the original date set".format(data.shape[0]))
        if is_train:
            targets = data[config['aspect_target']].values
        else:
            targets = np.zeros((data.shape[0], len(config['aspect_target'])))
        for row in data['text_processed'].values:
            sentence = str(row)
            sentences.append(sentence)
        print("There are {} sentences parsed".format(len(sentences)))
        return targets, sentences

    def save_result(self, prob_prediction, prediction_file_path):
        result_df = pd.DataFrame(prob_prediction, columns=config['aspect_target'])
        # result_df.to_csv(prediction_file_path, index=False)
        result_df.to_json(prediction_file_path, orient='records', lines=True)
        return

    def transform_prediction(self, prob_prediction):
        thresholds = [0.5, 0.25, 0.2, 0.5, 0.5]
        predict_softmax = np.zeros(prob_prediction.shape, dtype=int)
        for row_index, row in enumerate(prob_prediction):
            for index, each in enumerate(row):
                if each >= thresholds[index]:
                    predict_softmax[row_index][index] = 1
        return predict_softmax

