config = {
    'backbone': 'resnet-50',
    'checkpoint_weights': None,#'../output/weights/model-resnet-50-1',
    'save_path': '../output/weights/model-resnet-50-2',
    'finetuned_weights': '../output/weights/model-resnet-50-2.h5',
    'optimizer': 'sgd', # 'sgd' or 'adam'
    'learning_rate': {
        'max': 1e-2,
        'min': 1e-3,
        'steps_per_epoch': 17_960, # rough approximation (will differ between phases)
        'warmup_epochs': 0,
        'decay_epochs': 36,
        'power': 1
    },
    'loss': {
        'type': 'arcface', # arcface or cosface
        'scale': 32,
        'margin': 0.3,
    },
    'gem_p': 1.0,
    'clip_grad': 10.0,
    'n_epochs': 48,
    'batch_size': 18,
    'input_dim': 384,
    'n_classes': 81313,
    'dense_units': 1024,
    'dropout_rate': 0.0,
    'undersample': 0.25,
}
