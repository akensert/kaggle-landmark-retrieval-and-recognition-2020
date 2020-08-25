config_1 = {
    'backbone': 'efficientnet-b0',
    'checkpoint_weights': None,
    'save_path': '../output/weights/model-efficientnet-b0-1',
    'load_path': '../output/weights/model-efficientnet-b0-1', # specific to serving
    'optimizer': 'sgd', # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-2,
        'min': 1e-3,
        'steps_per_epoch': 19756, # rough approximation (will differ between phases)
        'warmup_epochs': 0,
        'decay_epochs': 16,
        'power': 1
    },
    'loss': {
        'type': 'arcface', # softmax, arcface or cosface
        'scale': 32, # only if arcface or cosface
        'margin': 0.3, # only if arcface or cosface
    },
    'gem_p': 1.0,
    'clip_grad': 10.0,
    'n_epochs': 24,
    'phases': [36,], # batch number
    'batch_size': [20, 12],
    'input_size': [384, 512],
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
    'backbone': 'efficientnet-b3',
    'checkpoint_weights': None,
    'save_path': '../output/weights/model-efficientnet-b3-2',
    'load_path': '../output/weights/model-efficientnet-b3-2', # specific to serving
    'optimizer': 'sgd', # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-2,
        'min': 1e-3,
        'steps_per_epoch': 19756, # rough approximation (will differ between phases)
        'warmup_epochs': 0,
        'decay_epochs': 28,
        'power': 1
    },
    'loss': {
        'type': 'cosface', # softmax, arcface or cosface
        'scale': 32, # only if arcface or cosface
        'margin': 0.3, # only if arcface or cosface
    },
    'gem_p': 1.0,
    'clip_grad': 10.0,
    'n_epochs': 30,
    'phases': [28,], # batch number
    'batch_size': [20, 12,],
    'input_size': [384, 512,],
    'served_input_size': 512, # specific to serving
    'n_classes': 81313,
    'dense_units': 1024,
    'dropout_rate': 0.0,
    'data_sampling': {
        'alpha': 0.5,
        'frac': 0.25,
    },
}
