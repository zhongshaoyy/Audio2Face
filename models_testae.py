import torch.nn as nn
import torch
from torch.autograd import Variable

## LSTM-AE model
# concat
class LSTMAEsigm(nn.Module):

    def __init__(self, num_features=39, num_audiof=128, num_bsf=2, num_blendshapes=51, is_concat=True):
        super(LSTMAEsigm, self).__init__()
        # assign
        self.is_concat = is_concat

        ## encoder
        # audio part with LSTM
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.audio_fc = nn.Linear(256*2, num_audiof)

        # blendshape part with fc
        self.bs_fc = nn.Sequential(
            nn.Linear(num_blendshapes, 24),
            nn.ReLU(True),

            nn.Linear(24, num_bsf),
            nn.Sigmoid() # constrain to 0~1, control variable
        )

        ## decoder?
        if self.is_concat:
            self.decoder_fc = nn.Sequential(
                nn.Linear(num_audiof+num_bsf, 64),
                nn.ReLU(True),
                nn.Linear(64, num_blendshapes),
                nn.Sigmoid()
            )
        else:
            self.bilinear = nn.Bilinear(num_bsf, num_audiof, num_audiof)
            self.decoder_fc = nn.Sequential(
                nn.Linear(num_audiof, 64),
                nn.ReLU(True),
                nn.Linear(64, num_blendshapes),
                nn.Sigmoid()
            )

    def fuse(self, audio_z, bs_z):
        # concat or bilinear
        if self.is_concat:
            return torch.cat((audio_z, bs_z), dim=1)
        else:
            return self.bilinear(bs_z, audio_z)

    def decode(self, z):
        return self.decoder_fc(z)

    def decode_audio(self, audio, bs_z):
        audio_rnn, _ = self.rnn(audio)
        audio_z = self.audio_fc(audio_rnn[:, -1, :])
        bs_z = bs_z.repeat(audio_z.size()[0], 1) # to batch size

        z = self.fuse(audio_z, bs_z)
        return self.decode(z)

    def forward(self, audio, blendshape):
        # encode
        audio_rnn, _ = self.rnn(audio)
        audio_z = self.audio_fc(audio_rnn[:, -1, :])

        bs_z = self.bs_fc(blendshape)

        # decode
        z = self.fuse(audio_z, bs_z)
        output = self.decode(z)

        return audio_z, bs_z, output

class LSTMAEdist(nn.Module):

    def __init__(self, num_features=39, num_audiof=128, num_bsf=2, num_blendshapes=51, is_concat=True):
        super(LSTMAEdist, self).__init__()
        # assign
        self.is_concat = is_concat

        ## encoder
        # audio part with LSTM
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.audio_fc = nn.Linear(256*2, num_audiof)

        # blendshape part with fc
        self.bs_fc1 = nn.Sequential(
            nn.Linear(num_blendshapes, 24),
            nn.ReLU(True),
        )
        self.bs_fc21 = nn.Linear(24, num_bsf)
        self.bs_fc22 = nn.Linear(24, num_bsf)

        ## decoder?
        if self.is_concat:
            self.decoder_fc = nn.Sequential(
                nn.Linear(num_audiof+num_bsf, 64),
                nn.ReLU(True),
                nn.Linear(64, num_blendshapes),
                nn.Sigmoid()
            )
        else:
            self.bilinear = nn.Bilinear(num_bsf, num_audiof, num_audiof)
            self.decoder_fc = nn.Sequential(
                nn.Linear(num_audiof, 64),
                nn.ReLU(True),
                nn.Linear(64, num_blendshapes),
                nn.Sigmoid()
            )

    def encode(self, audio, blendshape):
        audio_rnn, _ = self.rnn(audio)
        audio_z = self.audio_fc(audio_rnn[:, -1, :])

        bs_h1 = self.bs_fc1(blendshape)
        return audio_z, self.bs_fc21(bs_h1), self.bs_fc22(bs_h1)

    def fuse(self, audio_z, bs_z):
        # concat or bilinear
        if self.is_concat:
            return torch.cat((audio_z, bs_z), dim=1)
        else:
            return self.bilinear(bs_z, audio_z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            # eps = torch.randn_like(std)
            # eps = torch.randn(std.size(), dtype=std.dtype, layout=std.layout, device=std.device)
            eps = Variable(torch.randn(std.size())).cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder_fc(z)

    def decode_audio(self, audio, bs_z):
        audio_rnn, _ = self.rnn(audio)
        audio_z = self.audio_fc(audio_rnn[:, -1, :])
        bs_z = bs_z.repeat(audio_z.size()[0], 1)

        z = self.fuse(audio_z, bs_z)
        return self.decode(z)

    def forward(self, audio, blendshape):
        # encode
        audio_z, bs_mu, bs_logvar = self.encode(audio, blendshape)
        bs_z = self.reparameterize(bs_mu, bs_logvar)

        # decode
        z = self.fuse(audio_z, bs_z)
        output = self.decode(z)

        return audio_z, bs_z, output, bs_mu, bs_logvar

class LSTMAE2dist(nn.Module):

    def __init__(self, num_features=39, num_audiof=128, num_bsf=2, num_blendshapes=51, is_concat=True):
        super(LSTMAE2dist, self).__init__()
        # assign
        self.is_concat = is_concat
        self.num_audiof = num_audiof
        self.num_bsf = num_bsf

        ## encoder
        # audio part with LSTM
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.audio_fc11 = nn.Linear(256*2, num_audiof)
        self.audio_fc12 = nn.Linear(256*2, num_audiof)

        # blendshape part with fc
        self.bs_fc1 = nn.Sequential(
            nn.Linear(num_blendshapes, 24),
            nn.ReLU(True),
        )
        self.bs_fc21 = nn.Linear(24, num_bsf)
        self.bs_fc22 = nn.Linear(24, num_bsf)

        ## decoder?
        if self.is_concat:
            self.decoder_fc = nn.Sequential(
                nn.Linear(num_audiof+num_bsf, 64),
                nn.ReLU(True),
                nn.Linear(64, num_blendshapes),
                nn.Sigmoid()
            )
        else:
            # self.bilinear = nn.Bilinear(num_bsf, num_audiof, num_audiof)
            # self.decoder_fc = nn.Sequential(
            #     nn.Linear(num_audiof, 64),
            #     nn.ReLU(True),
            #     nn.Linear(64, num_blendshapes),
            #     nn.Sigmoid()
            # )
            print('NO bilinear combination in 2dist model')

    def encode(self, audio, blendshape):

        audio_rnn, _ = self.rnn(audio)
        audio_h = audio_rnn[:, -1, :]

        bs_h1 = self.bs_fc1(blendshape)

        return self.audio_fc11(audio_h), self.audio_fc12(audio_h), self.bs_fc21(bs_h1), self.bs_fc22(bs_h1)

    # def fuse(self, audio_z, bs_z):
    #     # concat or bilinear
    #     if self.is_concat:
    #         return torch.cat((audio_z, bs_z), dim=1)
    #     else:
    #         return self.bilinear(bs_z, audio_z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            # eps = torch.randn_like(std)
            # eps = torch.randn(std.size(), dtype=std.dtype, layout=std.layout, device=std.device)
            eps = Variable(torch.randn(std.size())).cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder_fc(z)

    def decode_audio(self, audio, bs_z):
        audio_rnn, _ = self.rnn(audio)
        audio_h = audio_rnn[:, -1, :]

        audio_mu = self.audio_fc11(audio_h)
        audio_logvar = self.audio_fc12(audio_h)

        audio_z = self.reparameterize(audio_mu, audio_logvar)

        bs_z = bs_z.repeat(audio_z.size()[0], 1)

        z = torch.cat((audio_z, bs_z), dim=1)
        return self.decode(z)

    def forward(self, audio, blendshape):
        # encode
        audio_mu, audio_logvar, bs_mu, bs_logvar = self.encode(audio, blendshape)
        mu = torch.cat((audio_mu, bs_mu), dim=1)
        logvar = torch.cat((audio_logvar, bs_logvar), dim=1)

        z = self.reparameterize(mu, logvar)

        # decode
        # z = self.fuse(audio_z, bs_z)
        output = self.decode(z)

        return z[:, :self.num_audiof], z[:, self.num_audiof:], output, mu, logvar
