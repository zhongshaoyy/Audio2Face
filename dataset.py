from torch.utils.data import Dataset
import numpy as np

class BlendshapeDataset(Dataset):

    def __init__(self, feature_file, target_file):
        self.wav_feature = np.load(feature_file)
        # reshape to avoid automatic conversion to doubletensor
        self.blendshape_target = np.loadtxt(target_file) / 100.0

        self._align()

    def __len__(self):
        return len(self.wav_feature)

    def _align(self):
        """
            align audio feature with blendshape feature
            generally, number of audio feature is less
        """

        n_audioframe, n_videoframe = len(self.wav_feature), len(self.blendshape_target)
        print('Current dataset -- n_videoframe: {}, n_audioframe:{}'.format(n_videoframe, n_audioframe))
        assert n_videoframe - n_audioframe <= 40
        if n_videoframe != n_audioframe:
            start_videoframe = 16
            self.blendshape_target = self.blendshape_target[start_videoframe : start_videoframe+n_audioframe]

    def __getitem__(self, index):
        return self.wav_feature[index], self.blendshape_target[index]
