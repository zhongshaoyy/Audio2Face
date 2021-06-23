import python_speech_features as psf
import numpy as np
# import scipy.signal as signal
import scipy.io.wavfile as wav
import os
from tqdm import tqdm


# data path
dataroot = '/home/liyachun/data/audio2bs'
wav_path = os.path.join(dataroot, 'test/wav/')
feature_path = os.path.join(dataroot, 'test/feature-mfcc39/')

if not os.path.isdir(feature_path): os.mkdir(feature_path)

def audio_mfcc(wav_file, feature_file=None):
    # load wav
    (rate, sig) = wav.read(wav_file)
    print(rate)

    # parameterss
    videorate = 30
    winlen = 1./videorate # time
    winstep = 0.5/videorate # 2* overlap
    # numcep = 32 # number of cepstrum to return, 0--power 1:31--features, nfilt caide
    numcep = 13 # typical value
    winfunc = np.hanning

    mfcc = psf.mfcc(sig, rate, winlen=winlen, winstep=winstep,
                    numcep=numcep, nfilt=numcep*2, nfft=int(rate/videorate), winfunc = winfunc)
    # print(mfcc_feature[0:5])
    mfcc_delta = psf.base.delta(mfcc, 2)
    mfcc_delta2 = psf.base.delta(mfcc_delta, 2)

    mfcc_all = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=1)
    print(type(mfcc_all), mfcc_all.shape)

    win_size = 64
    win_count = int((len(mfcc_all)-win_size)/2)+1
    output = np.zeros(shape=(win_count, win_size, numcep*3))
    # print(output.shape, mfcc_feature.shape)
    for win in tqdm(range(win_count)):
        output[win] = mfcc_all[2*win : 2*win+win_size]

    if feature_file:
        np.save(feature_file, output)
        
    print("MPCC extraction finished {}".format(output.shape))

    return output


def main():
    wav_files = os.listdir(wav_path)
    print(wav_files)
    for wav_file in wav_files:
        feature_file = wav_file.split('.')[0] + '-mfcc39.npy'
        print('-------------------\n', feature_file)
        audio_mfcc(os.path.join(wav_path, wav_file),
            os.path.join(feature_path, feature_file))

if __name__ == '__main__':
    main()
