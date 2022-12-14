import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.callback.earlystopping import EarlyStopping
from pybert.train.metrics import F1Score, AUC, AccuracyThresh, MultiLabelReport
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
from pybert.io.task_data import TaskData
from sklearn.model_selection import train_test_split
import numpy as np
from pybert.test.predictor import Predictor
warnings.filterwarnings("ignore")
import os

def run_train(args):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    data = TaskData()
    targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                        preprocessor=EnglishPreProcessor(),
                                        is_train=True)
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        targets,
                                                        shuffle=False,
                                                        test_size=0.05)


    # Train model on the whole training data
    train_data, valid_data = data.train_val_split(X=X_train, y=y_train, shuffle=False, stratify=False,
                                                valid_size=args.valid_size, data_dir=config['data_dir'],
                                                data_name=args.data_name)
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                'data_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)
    earlystopping = EarlyStopping(patience=50)
    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args= args,model=model,logger=logger,criterion=BCEWithLogLoss(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=earlystopping,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[F1Score(search_thresh=True),
                                     AUC(average='macro', task_type='binary'),
                                     MultiLabelReport(id2label=id2label)])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)

    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                        logger=logger,
                        n_gpu=args.n_gpu)

    logger.info('creating test data')
    test_data = []
    for step,(data_x, data_y) in enumerate(zip(X_test, y_test)):
        test_data.append((data_x, data_y))
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                            'data_dir'] / f"cached_training_part_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                            'data_dir'] / "cached_training_part_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    logger.info('predicting test data')
    test_prediction = predictor.predict(data=test_dataloader)
    logger.info('evaluation of predicted test data')
    data.evaluate(test_prediction, y_test, config['threshold_candidates'])

def run_test(args):
    data = TaskData()
    targets, sentences = data.read_data(raw_data_path=config['test_data_path'],
                                        preprocessor=EnglishPreProcessor(),
                                        is_train=False)
    lines = list(zip(sentences, targets))
    logger.info('==============There are {} instances in total to be prediected=============='.format(len(lines)))
    
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    try: 
        os.remove(config['data_dir'] / f"cached_test_examples_{args.arch}")
        logger.info('The previous cached test examples have been successfully deleted....')
        os.remove(config['data_dir'] / "cached_test_features_{}_{}".format(args.eval_max_seq_len, args.arch))
        logger.info('The previous cached test features have been successfully deleted....')
    except:
        pass
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                            'data_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                            'data_dir'] / "cached_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    print('loaded the test_features')
    test_dataset = processor.create_dataset(test_features)
    
    print("created dataset")
    test_sampler = SequentialSampler(test_dataset)
    
    print("finished test_sampler")
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    print("finished test_dataloader")
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))
    
    print("model_loaded")
    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu)
    results = predictor.predict(data=test_dataloader)
    sentiment_preds = [1 if each >= config['threshold'] else 0 for each in results]
    data.save_result(sentiment_preds, config['test_data_path'], config['predict_output_path'])


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", default=True, action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='medical_sieve_sentiment', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    init_logger(log_file=config['log_dir'] / f'{args.arch}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
