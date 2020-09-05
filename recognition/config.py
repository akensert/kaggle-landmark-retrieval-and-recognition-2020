
config = {
    'checkpoint_weights': False,
    'optimizer': {
        'learning_rate_start': 1e-3,
        'learning_rate_end': 1e-4,
        'momentum': 0.9,
    },
    'loss': {
        'type': 'cosface', # arcface or cosface
        'scale': 32,
        'margin': 0.3,
    },
    'n_epochs': 64,
    'batch_size': 32,
    'input_dim': 256,
    'dense_units': 1024,
}
