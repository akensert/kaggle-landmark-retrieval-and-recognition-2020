
config = {
    'checkpoint_weights': None,#'../output/weights/model-resnet-50',
    'save_path': '../output/weights/model-resnet-101',
    'finetuned_weights': '../output/weights/model-resnet-101.h5',
    'optimizer': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'momentum': 0.9,
    },
    'loss': {
        'type': 'cosface', # arcface or cosface
        'scale': 32,
        'margin': 0.2,
    },
    'n_epochs': 64,
    'batch_size': 16,
    'input_dim': 384,
    'dense_units': 1024,
}
