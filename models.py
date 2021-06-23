import torch.nn as nn
import torch

# n_blendshape = 51

# audio2blendshape model
class A2BNet(nn.Module):

    def __init__(self, base_model='lstm'):
        super(A2BNet, self).__init__()

        self.base_model = base_model
        print('Initialising with %s' % (base_model))


    def _prepare_model(self):

        if self.base_model == 'lstm':
            self.base_model = FullyLSTM()

        elif self.base_model == 'nvidia':
            self.base_model = NvidiaNet()

        # lstm concat before nvidia network
        elif self.base_model == 'lstm-nvidia':
            self.base_model = LSTMNvidiaNet()

    def forward(self, input):
        output = self.base_model(input)

        return output

# LSTM model
class FullyLSTM(nn.Module):

    def __init__(self, num_features=32, num_blendshapes=51):
        super(FullyLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=256, num_layers=2,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.out = nn.Linear(256*2, num_blendshapes)

    def forward(self, input):
        # self.rnn.flatten_parameters()
        output, _ = self.rnn(input)
        output = self.out(output[:, -1, :])
        return output

# LSTM-AE model
class LSTMAE(nn.Module):

    def __init__(self, num_timeseries=32, num_audiof=128, num_bsf=2, num_blendshapes=51):
        super(LSTMAE, self).__init__()
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
            nn.Sigmoid()
        )

        ## decoder?
        self.decoder_fc = nn.Sequential(
            nn.Linear(num_audiof+num_bsf, 64),
            nn.ReLU(True),
            nn.Linear(64, num_blendshapes),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            # eps = torch.randn_like(std)
            # eps = torch.randn(std.size(), dtype=std.dtype, layout=std.layout, device=std.device)
            eps = torch.randn(std.size())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder_fc(z)


    def forward(self, audio, blendshape):
        # encoder
        audio_rnn, _ = self.rnn(audio)
        audio_fc = self.audio_fc1(audio_rnn[:, -1, :])

        bs_fc = self.bs_fc(blendshape)

        z = torch.cat((audio_fc, bs_fc), dim=1)
        output = self.decode(z)

        return audio_fc, bs_fc, output


# nvidia model
class NvidiaNet(nn.Module):

    def __init__(self, num_blendshapes=51):
        super(NvidiaNet, self).__init__()
        # formant analysis network
        self.num_blendshapes = num_blendshapes
        self.formant = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # articulation network
        self.articulation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4,1), stride=(4,1)),
            nn.ReLU()
        )

        # output network
        self.output = nn.Sequential(
            nn.Linear(256, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # (-1, channel, height, width)
        # or x = x.view(-1, 1, 64, 32)

        # convolution
        x = self.formant(x)
        x = self.articulation(x)

        # fully connected
        x = x.view(-1, num_flat_features(x))
        x = self.output(x)

        return x

class LSTMNvidiaNet(nn.Module):

    def __init__(self, num_blendshapes=51, num_emotions=16):
        super(LSTMNvidiaNet, self).__init__()

        self.num_blendshapes = num_blendshapes
        self.num_emotions = num_emotions

        # emotion network with LSTM
        self.emotion = nn.LSTM(input_size=32, hidden_size=128, num_layers=1,
                        batch_first=True, dropout=0.5, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(128*2, 150),
            nn.ReLU(),
            nn.Linear(150, self.num_emotions)
        )


        # formant analysis network
        self.formant = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # articulation network
        self.conv1 = nn.Conv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv2 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.conv5 = nn.Conv2d(256+self.num_emotions, 256, kernel_size=(4,1), stride=(4,1))
        self.relu = nn.ReLU()

        # output network
        self.output = nn.Sequential(
            nn.Linear(256+self.num_emotions, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(150, self.num_blendshapes)
        )

    def forward(self, x):
        # extract emotion state
        e_state, _ = self.emotion(x[:, ::2]) # input features are 2* overlapping
        e_state = self.dense(e_state[:, -1, :]) # last
        e_state = e_state.view(-1, self.num_emotions, 1, 1)

        x = torch.unsqueeze(x, dim=1)
        # convolution
        x = self.formant(x)

        # conv+concat
        x = self.relu(self.conv1(x))
        x = torch.cat((x, e_state.repeat(1, 1, 32, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 16, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 8, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 4, 1)), 1)

        x = self.relu(self.conv5(x))
        x = torch.cat((x, e_state), 1)

        # fully connected
        x = x.view(-1, num_flat_features(x))
        x = self.output(x)

        return x




def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
