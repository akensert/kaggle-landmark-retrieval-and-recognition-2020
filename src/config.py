

config = {
    'optimizer': 'sgd',              # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-3,
        'min': 1e-4,
        'warmup_epochs': 1,
        'decay_epochs': 100,
        'power': 1
    },
    'loss': {
        'type': 'arcface',           # arcface or cosface
        'scale': 30,
        'margin': 0.3,
    },
    'input_path': '../input/landmark-retrieval-2020/',
    'n_epochs': 100,
    'batch_size': 32,
    'input_size': (320, 320, 3),
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.0,
    'regularization_factor': 1e-10,  # not utilized
    'pad_on_resize': True,
    'K': 1
}
