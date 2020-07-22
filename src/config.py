

config_1 = {
    'backbone': 'resnet-50',
    'pretrained_weights': 'imagenet',
    'save_path': '../output/weights/model-resnet-50-1.h5',
    'optimizer': 'sgd',              # 'sgd' (momentum=0.9) or 'adam'
    'learning_rate': {
        'max': 1e-3,
        'min': 1e-4,
        'warmup_epochs': 1,
        'decay_epochs': 100,
        'power': 1
    },
    'loss': {                        # only for margin loss (not triplet loss)
        'type': 'arcface',           # arcface or cosface
        'scale': 30,
        'margin': 0.3,
    },
    'input_path': '../input/landmark-retrieval-2020/',
    'n_epochs': 100,
    'batch_size': 48,
    'input_size': (256, 256, 3),
    'tt_scaling': [0.5, 0.75, 1.00, 1.25, 1.5],
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.0,
    'regularization_factor': 1e-10,  # not utilized
    'K': 1                           # only for create_triplet_dataset
}

config_2 = {}

config_3 = {}

config_4 = {}

config_5 = {}

config_6 = {}
