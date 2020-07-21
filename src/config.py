

config = {
    'optimizer': 'sgd',              # 'sgd' (momentum=0.9) or 'adam'
    'learning_rate': {
        'max': 1e-3,
        'min': 1e-4,
        'warmup_epochs': 10,
        'decay_epochs': 200,
        'power': 1
    },
    'loss': {                        # only for margin loss (not triplet loss)
        'type': 'arcface',           # arcface or cosface
        'scale': 30,
        'margin': 0.3,
    },
    'input_path': '../input/landmark-retrieval-2020/',
    'n_epochs': 200,
    'batch_size': 24,
    'input_size': (384, 384, 3),
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.2,
    'regularization_factor': 1e-10,  # not utilized
    'pad_on_resize': False,
    'K': 1                           # only for create_triplet_dataset
}
