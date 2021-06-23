from audiolazy import lpc
# from audiolazy import lazy_lpc
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import os
from tqdm import tqdm
from multiprocessing import Pool

_num_worker = 8

# data path
dataroot = '/home/liyachun/data/audio2bs'
wav_path = os.path.join(dataroot, 'sandbox/wav/')
feature_path = os.path.join(dataroot, 'sandbox/feature-lpc-mp/')
if not os.path.isdir(feature_path): os.mkdir(feature_path)

def audio_lpc(wav_file, feature_file):
    (rate, sig) = wav.read(wav_file)
    print(rate)

    videorate = 30
    nw = int(rate/videorate) # time
    inc = int(nw/2) # 2* overlap
    # assert type(inc) == int
    winfunc = np.hanning(nw)

    def enframe(signal, nw, inc, winfunc):
        """turn audio signal to frame.
        parameters:
        signal: original audio signal
        nw: length of each frame(audio samples = audio sample rate * time interval)
        inc: intervals of consecutive frames
        """
        signal_length=len(signal) #length of audio signal
        if signal_length<=nw:
            nf=1
        else: #otherwise, compute the length of audio frame
            nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
        pad_length=int((nf-1)*inc+nw) #length of flatten all frames
        zeros=np.zeros((pad_length-signal_length,)) #
        pad_signal=np.concatenate((signal,zeros)) #after padding
        #nf*nw matrix
        indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T
        indices=np.array(indices,dtype=np.int32) #turn indices to frames
        frames=pad_signal[indices] #get frames
        win=np.tile(winfunc,(nf,1))  #window function
        return frames*win   #return frame matrix

    sig = signal.detrend(sig, type= 'constant')

    frame = enframe(sig, nw, inc, winfunc)
    assert len(frame) >= 64

    win_size = 64
    K = 32 # number of coefficients
    win_count = int((len(frame)-win_size)/2)+1 # len(frame) or frame.shape[0]
    lpc_feature = np.zeros(shape=(frame.shape[0], K))
    output = np.zeros(shape=(win_count, win_size, K))
    # print(output.shape, mfcc_feature.shape)

    # pbar = tqdm(total=len(frame))
    pool = Pool(_num_worker)

    # for i in range(len(frame)):
    #     filt = lpc.nautocor(frame[i], order=K)
    #     lpc_feature[i] = filt.numerator[1:]
    #
    #     if i > 0 and i % 10 == 0:
    #         pbar.update(10)
    filt_coef = pool.map(lpc_K, tqdm(frame))
    lpc_feature[:] = filt_coef

    # pbar.close()
    print(type(lpc_feature), lpc_feature.shape)

    for win in range(win_count):
        output[win] = lpc_feature[2*win : 2*win+win_size]

    np.save(feature_file, output)
    print("LPC extraction finished {}".format(output.shape))

def lpc_K(frame, order=32):
    filt = lpc.nautocor(frame, order=order)
    return filt.numerator[1:] # List of coefficients

def main():
    wav_files = os.listdir(wav_path)
    print(wav_files)
    for wav_file in wav_files:
        feature_file = wav_file.split('.')[0] + '-lpc' + '.npy'
        print('-------------------\n', feature_file)
        audio_lpc(wav_path+wav_file, feature_path+feature_file)

if __name__ == '__main__':
    main()
