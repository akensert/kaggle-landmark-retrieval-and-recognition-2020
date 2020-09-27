
config = {
    'checkpoint_weights': True,
    'optimizer': {
        'learning_rate_start': 1e-3,
        'learning_rate_end': 1e-4,
        'momentum': 0.9,
    },
    'loss': {
        'type': 'arcface', # arcface or cosface
        'scale': 32,
        'margin': 0.3,
    },
    'n_epochs': 16,
    'batch_size': 6,
    'input_dim': 512,
    'dense_units': 1024,
}
