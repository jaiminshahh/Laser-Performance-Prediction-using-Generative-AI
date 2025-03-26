from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch

class SpatioTemporalDataset(Dataset):
    def __init__(self, config=None, mode=None, folpath=None):
        super(SpatioTemporalDataset, self).__init__()
        self.mode = mode
        # Gets the data in [input image, input pulse, output image, output pulse] format
        self.data = self.getallfilelist(folpath)

    def __getitem__(self, index):
        img_inp, pls_inp, img_out, pls_out = self.data[index]
        # Read the input image and pulse files
        input_image_data = torch.tensor(np.array(pd.read_csv(img_inp, header=None)))
        input_pulse_data = torch.tensor(np.array(pd.read_csv(pls_inp, header=None))).squeeze(0)[59:]
        # Temporal scaling to create video
        input_3d_tensor = torch.stack([input_image_data * factor for factor in input_pulse_data]).unsqueeze(0)
        # Read the output image and pulse
        output_image_data = torch.tensor(np.array(pd.read_csv(img_out, header=None)))
        output_pulse_data = torch.tensor(np.array(pd.read_csv(pls_out, header=None))).squeeze(0)[59:]

        # output_pulse_data = output_pulse_data / torch.max(output_pulse_data)

        return input_3d_tensor.float(), output_image_data.float(), output_pulse_data.float(), input_pulse_data.float(), img_inp, img_out


    def __len__(self):
        return len(self.data)


    def getallfilelist(self, pth):
        X = []; Y = []
        for imgpth, plspth in pth:
            # Get all the files from each folder
            if os.path.isdir(imgpth) and os.path.isdir(plspth):
                input_image_files = list(sorted([f for f in os.listdir(imgpth) if
                               "Inj_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
                input_pulse_files = list(sorted([f for f in os.listdir(plspth) if
                                     "_InjPulsePower_InjEnergyFactor_" in f and f.endswith(".csv")]))
                output_image_files = list(sorted([f for f in os.listdir(imgpth) if
                                "UV_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
                output_pulse_files = list(sorted([f for f in os.listdir(plspth) if
                                      "_UVDBS_Power_InjEnergyFactor_" in f and f.endswith(".csv")]))
                # Concatenate to form input and output list
                for i in range(len(input_image_files)):
                    X.append([os.path.join(imgpth, input_image_files[i]), os.path.join(plspth, input_pulse_files[i])])
                    Y.append([os.path.join(imgpth, output_image_files[i]), os.path.join(plspth, output_pulse_files[i])])

        # Join the input and output list
        final_data = [x + y for x, y in zip(X,Y)]
        return final_data


def get_loader(config, fol_path, mode=None):
    # Dataset class
    dataset = SpatioTemporalDataset(config=config, mode=mode, folpath=fol_path)
    # x,y,z = dataset[0]
    # Callt he dataloader
    if mode == 'train' or mode == 'valid':
        dataloader = DataLoader(dataset, batch_size=config['training_parameters']['batch_size'], shuffle=True,
                                num_workers=config['training_parameters']['num_workers'])
    else:
        dataloader = DataLoader(dataset, batch_size=1,
                                num_workers=config['training_parameters']['num_workers'])

    return dataset, dataloader