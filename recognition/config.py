
config = {
    'checkpoint_weights': True,
    'optimizer': {
        'learning_rate_start': 1e-4,
        'learning_rate_end': 1e-4,
        'momentum': 0.9,
    },
    'loss': {
        'type': 'cosface', # arcface or cosface
        'scale': 32,
        'margin': 0.3,
    },
    'n_epochs': 64,
    'batch_size': 24,
    'input_dim': 384,
    'dense_units': 1024,
}
