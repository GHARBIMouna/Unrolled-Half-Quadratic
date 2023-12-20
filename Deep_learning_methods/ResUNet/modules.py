import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class MyDataset(torch.utils.data.Dataset):
    """"
    Loads the dataset.
    """

    def __init__(self, folder, need_names):
        super(MyDataset, self).__init__()
        self.folder_Gr = os.path.join(folder, "Groundtruth")
        self.folder_De = os.path.join(folder, "Degraded2000")
       
        self.file_names_Gr = os.listdir(self.folder_Gr)
        self.file_names_De = os.listdir(self.folder_De)
                
        self.file_list_Gr = [os.path.join(self.folder_Gr, i) for i in self.file_names_Gr if not i.startswith('.')]
        self.file_list_De = [os.path.join(self.folder_De, i) for i in self.file_names_De if not i.startswith('.')]
        self.need_names = need_names
  

    def __getitem__(self, index):
        X_true = np.load(self.file_list_Gr[index], allow_pickle=True)
        Degraded_path = (self.file_list_Gr[index].replace('Groundtruth', 'Degraded2000')).replace('Gr_', 'De_')
        X_degraded = np.load(Degraded_path, allow_pickle=True)
  
        if self.need_names == 'no':
            return X_true, X_degraded
        elif self.need_names =="yes":
            name=os.path.splitext(self.file_names_Gr[index])[0]
            return name, X_true, X_degraded



    def __len__(self):
        return len(self.file_list_Gr)

    
