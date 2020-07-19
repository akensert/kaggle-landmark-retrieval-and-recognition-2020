import pandas as pd
import glob

def read_data(input_path):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    data = pd.read_csv(input_path + 'train.csv')
    data['path'] = data['id'].map(mapping)
    return (
        data
        .groupby('landmark_id')['path']
        .agg(lambda x: ','.join(x)) # 'path1,path2,path3,path4,...'
        #.apply(list) # ['path1', 'path2', 'path3', 'path4', ...]
        .reset_index()
    )
