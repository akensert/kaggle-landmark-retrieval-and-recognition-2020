

config_1 = {
    'backbone': 'efficientnet-b0',
    'pretrained_weights': 'imagenet',
    'save_path': '../output/weights/model-efficientnet-b0-1.h5',
    'optimizer': 'sgd',              # 'sgd' (momentum=0.9) or 'adam'
    'learning_rate': {
        'max': 3e-3,
        'min': 3e-4,
        'warmup_epochs': 1,
        'decay_epochs': 10,
        'power': 1
    },
    'loss': {                        # only for margin loss (not triplet loss)
        'type': 'cosface',           # arcface or cosface
        'scale': 30,
        'margin': 0.3,
    },
    'input_path': '../input/landmark-retrieval-2020/',
    'n_epochs': 100,
    'batch_size': 26,
    'input_size': (384, 384, 3),
    'tt_scaling': [1.00],
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.2,
    'regularization_factor': 1e-10,  # not utilized
    'K': 1                           # only for create_triplet_dataset
}

config_2 = {}

config_3 = {}

config_4 = {}

config_5 = {}

config_6 = {}
