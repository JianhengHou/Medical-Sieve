import os
from pathlib import Path

PWD = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(os.path.abspath(os.path.join(PWD, '..')))

PROJECT_ROOT = Path(BASE_DIR / "../..").resolve()
with open(os.path.join(PROJECT_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()


RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
PREDICTION_FILE_NAME = f'raddit_aspect_sentiment_prediction_{_version}_part2.jl'
PREDICTION_FILE_PATH = os.path.join(RESULT_DIR, PREDICTION_FILE_NAME)

SENTIMENT_TARGET = ['sentiment']
THRESHOLDS_SET = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
THRESHOLD = 0.3


config = {
    'raw_data_path':Path(BASE_DIR / "../../data/medical_sieve_sentiment_training_set.jl").resolve(),
    'test_data_path':Path(BASE_DIR / f"../../result/aspect_prediction_{_version}_part2.jl").resolve(),

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model',
    'predict_output_path': PREDICTION_FILE_PATH,

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

    'sentiment_target': SENTIMENT_TARGET,
    'threshold_candidates':THRESHOLDS_SET,
    'threshold': THRESHOLD,

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased',

    'albert_vocab_path': BASE_DIR / 'pretrain/albert/albert-base/30k-clean.model',
    'albert_config_file': BASE_DIR / 'pretrain/albert/albert-base/config.json',
    'albert_model_dir': BASE_DIR / 'pretrain/albert/albert-base'

}
