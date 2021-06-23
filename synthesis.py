import torch
import time
import argparse
import os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
from misc.audio_mfcc import audio_mfcc
from misc.audio_lpc import audio_lpc
from models import FullyLSTM, NvidiaNet
from models_testae import *

# parser arguments
parser = argparse.ArgumentParser(description='Synthesize wave to blendshape')
parser.add_argument('wav', type=str, help='wav to synthesize')
parser.add_argument('--smooth', type=bool, default=True)
parser.add_argument('--pad', type=bool, default=True)
parser.add_argument('--control', type=bool, default=False)

args = parser.parse_args()
print(args)

n_blendshape = 51
ckp = './checkpoint-lstmae-2distconcat_kl0/checkpoint-epoch500.pth.tar'
result_path = './synthesis'
result_file = 'test'+args.wav.split('/')[-1].split('.')[0]+'-lstmae-2distconcat_kl0.txt'

def main():
    global result_file

    start_time = time.time()
    ## process audio
    feature = torch.from_numpy(audio_mfcc(args.wav))
    # print('Feature extracted ', feature.shape)
    target = torch.from_numpy(np.zeros(feature.shape[0]))

    ## load model
    model = LSTMAE2dist(is_concat=True)

    # restore checkpoint model
    print("=> loading checkpoint '{}'".format(ckp))
    checkpoint = torch.load(ckp)
    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['eval_loss']))

    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    # evaluation for audio feature
    model.eval()

    ## build dataset
    test_loader = DataLoader(TensorDataset(feature, target),
                    batch_size=100, shuffle=False, num_workers=2)

    for i, (input, target) in enumerate(test_loader):
        # target = target.cuda(async=True)
        input_var = Variable(input.float(), volatile=True).cuda()
        # target_var = Variable(target.float(), volatile=True)

        # compute output
        if args.control:
            bs_z = Variable(torch.Tensor([0.5, 0.5]), volatile=True).cuda() # control variable
            # print('Control with', bs_z.data[0])
            output = model.decode_audio(input_var, bs_z)
        else:
            output = model(input_var)

        if i == 0:
            output_cat = output.data
        else:
            output_cat = torch.cat((output_cat, output.data), 0)
        # print(type(output_cat.cpu().numpy()), output_cat.cpu().numpy().shape)

    # convert back *100
    output_cat = output_cat.cpu().numpy()*100.0

    if args.smooth:
        #smooth3--savgol_filter
        win = 9; polyorder = 3
        for i in range(n_blendshape):
            power = output_cat[:,i]
            power_smooth = savgol_filter(power, win, polyorder, mode='nearest')
            output_cat[:, i] = power_smooth
        result_file = 'smooth-' + result_file

    # pad blendshape
    if args.pad:
        output_cat = pad_blendshape(output_cat)
        result_file = 'pad-' + result_file

    # count time for synthesis
    past_time = time.time() - start_time
    print("Synthesis finished in {:.4f} sec! Saved in {}".format(past_time, result_file))

    with open(os.path.join(result_path, result_file), 'wb') as f:
        np.savetxt(f, output_cat, fmt='%.6f')

def pad_blendshape(blendshape):
    return np.pad(blendshape, [(16, 16), (0, 0)], mode='constant', constant_values=0.0)

if __name__ == '__main__':
    main()
