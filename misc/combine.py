import numpy as np
import os

dataroot = '/home/liyachun/data/audio2bs'
feature_path = os.path.join(dataroot, 'sandbox/feature-lpc/')
target_path = os.path.join(dataroot, 'sandbox/blendshape/')

test_ind = ['06', '17']

def combine(feature_path, target_path):
    feature_files = sorted(os.listdir(feature_path))
    print('feature: ', feature_files)

    blendshape_files = sorted(os.listdir(target_path))
    print('bs:      ', blendshape_files)

    for i in range(len(feature_files)):
        # skip test files
        if blendshape_files[i].split('.')[0].split('_')[1] in test_ind:
            continue

        if i == 0:
            feature = np.load(feature_path+feature_files[i])
            feature_combine_file = feature_files[i].split('_')[0] + '.npy'

            # blendshape is shorter, need cut
            blendshape = np.loadtxt(target_path+blendshape_files[i])
            blendshape = cut(feature, blendshape)

            blendshape_combine_file = blendshape_files[i].split('_')[0] + '.txt'

        else:
            feature_temp = np.load(feature_path+feature_files[i])
            feature = np.concatenate((feature, feature_temp), 0)

            # blendshape is shorter
            blendshape_temp = np.loadtxt(target_path+blendshape_files[i])
            blendshape_temp = cut(feature_temp, blendshape_temp)

            blendshape = np.concatenate((blendshape, blendshape_temp), 0)

        print(i, blendshape_files[i], feature.shape, blendshape.shape)

    np.save(os.path.join(feature_path, feature_combine_file), feature)
    np.savetxt(os.path.join(target_path, blendshape_combine_file), blendshape, fmt='%.8f')

def cut(wav_feature, blendshape_target):
    n_audioframe, n_videoframe = len(wav_feature), len(blendshape_target)
    print('--------\n', 'Current dataset -- n_audioframe: {}, n_videoframe:{}'.format(n_audioframe, n_videoframe))
    assert n_videoframe - n_audioframe == 32
    start_videoframe = 16
    blendshape_target = blendshape_target[start_videoframe : start_videoframe+n_audioframe]

    return blendshape_target

def main():
    combine(feature_path, target_path)

if __name__ == '__main__':
    main()
