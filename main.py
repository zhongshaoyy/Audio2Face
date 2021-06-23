'''
    virtualenv: 63server pytorch
    author: Yachun Li (liyachun@outlook.com)
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil
import time
from datetime import datetime

from dataset import BlendshapeDataset
from models import NvidiaNet, LSTMNvidiaNet, FullyLSTM
from models_testae import *

# gpu setting
gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# hyper-parameters
n_blendshape = 51
learning_rate = 0.0001
batch_size = 100
epochs = 500

print_freq = 20
best_loss = 10000000

# data path
dataroot = '/home/liyachun/data/audio2bs'
# data_path = os.path.join(dataroot, audio2bs)
data_path = dataroot
checkpoint_path = './checkpoint-lstmae-2distconcat_kl001/'
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print('loss percent: MSE %.4f(%.6f), KLD %.4f, total %.4f'
    #     % (MSE.data[0], MSE.data[0]/(MSE.data[0]+KLD.data[0]), KLD.data[0], MSE.data[0]+KLD.data[0]))

    return MSE + 0.01*KLD

def main():
    global best_loss
    model = LSTMAE2dist(is_concat=True)
    print(model)
    # model = nn.DataParallel(model)

    # get data
    train_loader = torch.utils.data.DataLoader(
                    BlendshapeDataset(feature_file=os.path.join(data_path, 'feature/0201train-mfcc39.npy'),
                                    target_file=os.path.join(data_path, 'blendshape/0201train.txt')),
                    batch_size=batch_size, shuffle=True, num_workers=2
                    )
    val_loader = torch.utils.data.DataLoader(
                    BlendshapeDataset(feature_file=os.path.join(data_path, 'test/feature/0201_06-1min-mfcc39.npy'),
                                    target_file=os.path.join(data_path, 'test/blendshape/0201_06-1min.txt')),
                    batch_size=batch_size, shuffle=False, num_workers=2
                    )

    # define loss and optimiser
    criterion = nn.MSELoss() #??.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()

    # training
    print('------------\n Training begin at %s' % datetime.now())
    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(async=True)
            input_var = autograd.Variable(input.float()).cuda()
            target_var = autograd.Variable(target.float())

            # compute model output
            # audio_z, bs_z, output = model(input_var, target_var)
            # loss = criterion(output, target_var)
            audio_z, bs_z, output, mu, logvar = model(input_var, target_var) # method2: loss change
            loss = loss_function(output, target_var, mu, logvar)

            train_loss += loss.data[0]

            # compute gradient and do the backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % print_freq == 0:
            #     print('Training -- epoch: {} | iteration: {}/{} | loss: {:.6f} \r'
            #             .format(epoch+1, i, len(train_loader), loss.data[0]))

        train_loss /= len(train_loader)
        print('Glance at training   z: max/min of hidden audio(%.4f/%.4f), blendshape(%.4f/%.4f)'
            % (max(audio_z.data[0]), min(audio_z.data[0]), max(bs_z.data[0]), min(bs_z.data[0])))

        model.eval()
        eval_loss = 0.
        for input, target in val_loader:
            target = target.cuda(async=True)
            input_var = autograd.Variable(input.float(), volatile=True).cuda()
            target_var = autograd.Variable(target.float(), volatile=True)

            # compute output temporal?!!
            # audio_z, bs_z, output = model(input_var, target_var)
            # loss = criterion(output, target_var)
            audio_z, bs_z, output, mu, logvar = model(input_var, target_var) # method2: loss change
            loss = loss_function(output, target_var, mu, logvar)

            eval_loss += loss.data[0]

        eval_loss /= len(val_loader)

        # count time of 1 epoch
        past_time = time.time() - start_time

        print('Glance at validating z: max/min of hidden audio(%.4f/%.4f), blendshape(%.4f/%.4f)'
            % (max(audio_z.data[0]), min(audio_z.data[0]), max(bs_z.data[0]), min(bs_z.data[0])))

        # print('Evaluating -- epoch: {} | loss: {:.6f} \r'.format(epoch+1, eval_loss/len(val_loader)))
        print('epoch: {:03} | train_loss: {:.6f} | eval_loss: {:.6f} | {:.4f} sec/epoch \r'
            .format(epoch+1, train_loss, eval_loss, past_time))

        # save best model on val
        is_best = eval_loss < best_loss
        best_loss = min(eval_loss, best_loss)
        if is_best:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'eval_loss': best_loss,
                }, checkpoint_path+'model_best.pth.tar')

        # save models every 100 epoch
        if (epoch+1) % 100 == 0:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'eval_loss': eval_loss,
                }, checkpoint_path+'checkpoint-epoch'+str(epoch+1)+'.pth.tar')

    print('Training finished at %s' % datetime.now())

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, checkpoint_path+filename)
#     if is_best:
#         shutil.copyfile(checkpoint_path+filename, checkpoint_path+'model_best.pth.tar')

if __name__ == '__main__':
    main()
