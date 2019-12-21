from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

"""
gtzanspecgram.py

Contains the custom-made GTZAN Spectrogram dataset with labels
"""

# Change string labels to numeric values
label_dict = {'blues'     : 0,
              'classical' : 1,
              'country'   : 2,
              'disco'     : 3,
              'hiphop'    : 4,
              'jazz'      : 5,
              'metal'     : 6,
              'pop'       : 7,
              'reggae'    : 8,
              'rock'      : 9}

class GTZANSpecgram(Dataset):
    def __init__(self, csv_path):
        """
        csv_path (string): path to csv file
        drop the header row which contains metadata info
        """
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_info.drop(self.data_info.head(1).index, inplace=True)

        # image files are located in index 1
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # labels are located in index 3
        self.label_arr = np.asarray(self.data_info.iloc[:, 3].map(label_dict))

        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):

        imgitem = self.image_arr[index]
        img = Image.open(imgitem)
        imgtensor = self.to_tensor(img)

        imglabel = self.label_arr[index]

        return (imgtensor, imglabel)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Just for testing purposes
    data =  GTZANSpecgram('metadata_test.csv')
    single = data.__getitem__(0)
    print(single)
