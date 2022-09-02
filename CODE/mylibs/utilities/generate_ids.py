import os

def genearate_ids(path):
    ID_list_train = []
    path_train = os.path.join(path, 'train')
    for folder in os.listdir(path_train):
        for file_name in os.listdir(os.path.join(path_train, folder)):
            ID_list_train.append(os.path.join(folder, file_name))

    ID_list_validation = []
    path_validation = os.path.join(path, 'validation')
    for folder in os.listdir(path_validation):
        for file_name in os.listdir(os.path.join(path_validation, folder)):
            ID_list_validation.append(os.path.join(folder, file_name))

    return ID_list_train, ID_list_validation