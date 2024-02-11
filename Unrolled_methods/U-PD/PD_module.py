import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class MyDataset(torch.utils.data.Dataset):
    """"
    Loads the dataset.
    """

    def __init__(self, folder, initial_x0, need_names):
        super(MyDataset, self).__init__()
        self.need_names = need_names
        self.initial_x0 = initial_x0

        self.folder_Gr = os.path.join(folder, "Groundtruth")
        self.folder_De = os.path.join(folder, "Degraded")
        self.file_names_Gr = os.listdir(self.folder_Gr)
        self.file_names_De = os.listdir(self.folder_De)

        self.file_list_Gr = [os.path.join(
            self.folder_Gr, i) for i in self.file_names_Gr if not i.startswith('.')]
        self.file_list_De = [os.path.join(
            self.folder_De, i) for i in self.file_names_De if not i.startswith('.')]

        self.folder_H = os.path.join(folder, "H")
        self.file_names_H = os.listdir(self.folder_H)
        self.file_list_H = [os.path.join(
            self.folder_H, i) for i in self.file_names_H if not i.startswith('.')]

    def __getitem__(self, index):

        X_true = np.load(self.file_list_Gr[index], allow_pickle=True)
        Degraded_path = (self.file_list_Gr[index].replace(
            'Groundtruth', 'Degraded')).replace('Gr_', 'De_')
        X_degraded = np.load(Degraded_path, allow_pickle=True)

        H_path = (self.file_list_Gr[index].replace(
            'Groundtruth', 'H')).replace('Gr_', '').replace('x', 'H')
        H = np.load(H_path, allow_pickle=True)

        if self.need_names == 'no':
            return H, X_true, X_degraded
        if self.need_names == "yes":
            name = os.path.splitext(self.file_names_Gr[index])[0]
            return name, H, X_true, X_degraded

    def __len__(self):
        return len(self.file_list_Gr)
