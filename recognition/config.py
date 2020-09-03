
config = {
    'checkpoint_weights': None,#'../output/weights/resnet50.h5',
    'optimizer': {
        'learning_rate_start': 1e-2,
        'learning_rate_end': 1e-3,
        'momentum': 0.9,
    },
    'loss': {
        'type': 'arcface', # arcface or cosface
        'scale': 32,
        'margin': 0.3,
    },
    'n_epochs': 16,
    'batch_size': 32,
    'input_dim': 256,
    'dense_units': 1024,
}
