config_1 = {
    'backbone': 'efficientnet-b0',
    'checkpoint_weights': None,
    'save_path': '../output/weights/model-efficientnet-b0-2',
    'load_path': '../output/weights/model-efficientnet-b0-2_0', # specific to serving
    'optimizer': 'sgd', # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-2,
        'min': 1e-3,
        'steps_per_epoch': 17_960, # rough approximation (will differ between phases)
        'warmup_epochs': 1,
        'decay_epochs': 36,
        'power': 1
    },
    'loss': {
        'type': 'arcface', # arcface or cosface
        'scale': 30, # 32
        'margin': 0.4,
    },
    'gem_p': 1.0,
    'clip_grad': 10.0,
    'n_epochs': 48,
    'phases': [36, ], # batch number
    'batch_size': [32, 12],
    'input_size': [256, 512],
    'served_input_size': 384, # specific to serving
    'n_classes': 81313,
    'dense_units': 1024,
    'dropout_rate': 0.0,
    'data_sampling': {
        'alpha': 0.5,
        'frac': 0.25,
    },
}

config_2 = {
    'backbone': 'efficientnet-b5',
    'checkpoint_weights': None,
    'save_path': '../output/weights/model-efficientnet-b5-1',
    'load_path': '../output/weights/model-efficientnet-b5-1_1', # specific to serving
    'optimizer': 'sgd', # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-2,
        'min': 1e-3,
        'steps_per_epoch': 20_000, # rough approximation (will differ between phases)
        'warmup_epochs': 1,
        'decay_epochs': 16,
        'power': 1
    },
    'loss': {
        'type': 'cosface', # arcface or cosface
        'scale': 30,
        'margin': 0.3,
    },
    'gem_p': 1.0,
    'clip_grad': 10.0,
    'n_epochs': 48,
    'phases': [12, 24, 28, ], # batch number # TESTING
    'batch_size': [24, 10, 6, 4],
    'input_size': [(256, 256, 3), (384, 384, 3), (512, 512, 3), (640, 640, 3)],
    'served_input_size': (384, 384, 3), # specific to serving
    'n_classes': 81313,
    'dense_units': 1024,
    'dropout_rate': 0.0,
    'data_sampling': {
        'alpha': 0.5,
        'frac': 0.25,
    },
}

config_3 = {}

config_4 = {}

config_5 = {}

config_6 = {}
