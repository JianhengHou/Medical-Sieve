import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from ..configs.basic_config import config
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
        print("Opening the aspect prediction file:", raw_data_path)
        # data = pd.read_csv(raw_data_path)
        data = pd.read_json(raw_data_path, lines=True)
        print(data.info())
        print("There are {} rows in the original date set".format(data.shape[0]))
        if is_train:
            targets = data[config['sentiment_target']].values
        else:
            targets = np.zeros((data.shape[0], len(config['sentiment_target'])))
        for row in data['text_processed'].values:
            sentence = str(row)
            sentences.append(sentence)
        print("There are {} sentences parsed".format(len(sentences)))
        return targets, sentences

    def save_result(self, prob_prediction, original_file_path, prediction_file_path):
        # main_df = pd.read_csv(original_file_path)
        main_df = pd.read_json(original_file_path, lines=True)
        result_df = pd.DataFrame(prob_prediction, columns=config['sentiment_target'])
        print("*"*20)
        print("There are {} instances of sentiment prediction".format(result_df.shape[0]))
        print("There are {} instances of aspect prediction".format(main_df.shape[0]))
        print("*"*20)
        main_df = pd.concat([main_df, result_df], axis=1)
        # main_df.to_csv(prediction_file_path, index=False)
        main_df.to_json(prediction_file_path, orient='records', lines=True)
        return
    
    def evaluate(self, val_preds, sentiment_gt, threshold_candidates):
        for threshold in threshold_candidates:
            sentiment_preds = [1 if each >= threshold else 0 for each in val_preds]
            print("="*10+str(threshold)+"="*10)
            print(confusion_matrix(sentiment_gt, sentiment_preds, labels=[0, 1]))
            print(classification_report(sentiment_gt, sentiment_preds, target_names=['non negative', 'negative']))
        return
