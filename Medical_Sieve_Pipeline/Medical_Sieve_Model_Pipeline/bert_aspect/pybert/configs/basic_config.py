import os
from pathlib import Path
PWD = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(os.path.abspath(os.path.join(PWD, '..')))

PROJECT_ROOT = Path(BASE_DIR / "../..").resolve()
with open(os.path.join(PROJECT_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()


RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
TRAINING_STAGE_OUTPUT_DIR = os.path.join(RESULT_DIR, "training_stage_output")
PREDICT_STAGE_OUTPUT_DIR = os.path.join(RESULT_DIR, "predict_stage_output")

MODEL3_NAME = "BERT_MODEL"
MODEL3_TRAINING_STAGE_OUTPUT_NAME = f'{MODEL3_NAME}_training_stage_prediction_{_version}.jl'
MODEL3_TRAINING_STAGE_OUTPUT_PATH = os.path.join(TRAINING_STAGE_OUTPUT_DIR, MODEL3_TRAINING_STAGE_OUTPUT_NAME)
MODEL3_PREDICTION_NAME = f'{MODEL3_NAME}_prediction_{_version}.jl'
MODEL3_PREDICTION_PATH = os.path.join(PREDICT_STAGE_OUTPUT_DIR, MODEL3_PREDICTION_NAME)

ASPECT_TARGET = ['access', 
                 'costs', 
                 'delays', 
                 'errors', 
                 'trust']

config = {
    'raw_data_path':Path(BASE_DIR / "../../data/medical_sieve_aspect_training_set.jl").resolve(),
    'test_data_path': Path(BASE_DIR / "../../data/line-level-raddit-massage-board-dataset.jl").resolve(),
    'result_path': RESULT_DIR,

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'training_stage_output_path': MODEL3_TRAINING_STAGE_OUTPUT_PATH,
    'predict_stage_output_path': MODEL3_PREDICTION_PATH,

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

    'aspect_target': ASPECT_TARGET,

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased',

    'albert_vocab_path': BASE_DIR / 'pretrain/albert/albert-base/30k-clean.model',
    'albert_config_file': BASE_DIR / 'pretrain/albert/albert-base/config.json',
    'albert_model_dir': BASE_DIR / 'pretrain/albert/albert-base'
}


