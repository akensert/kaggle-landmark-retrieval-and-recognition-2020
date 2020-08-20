## landmark retrieval and recognition (2020)
#### Datasets
Datasets can be found [here](https://www.kaggle.com/c/landmark-retrieval-2020/data) (retrieval) and/or [here](https://www.kaggle.com/c/landmark-recognition-2020/data) (recognition). Note that the train image folder (which is most of the data) is the same for both competitions (thus only need to be downloaded once, from either 'retrieval' or 'recognition'). Downloaded dataset(s) (.zip file(s)) should be moved into `inputs/` and then unzipped (e.g. `unzip landmark-recognition-2020.zip`)

#### Train
##### Retrieval
Navigate into `retrieval/scripts/` and run from terminal: `python modify_trainfile.py`. This only has to be run once, but may take an hour to run. After it has finished, navigate outside the `retrieval/scripts/` folder (`../`) and run from terminal: `python main.py`. `main.py` will train the model according to `config_1` in `config.py`. When the training is done (which will take a couple of days on an RTX 20XX GPU card), run `python served_model.py` to save the model for later use.
##### Recoginition
Work in progress
