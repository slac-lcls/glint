"""1-D convolutional autoencoder for powder profiles + a head that predicts (cell, crystal system).

encoder: I(q) [1x1024] -> conv stack -> latent z (LATENT dims)
decoder: z -> conv-transpose stack -> reconstructed I(q)
head:    z -> MLP -> cell (6 normalised params) + system logits (7)

The latent is the shared representation: reconstruction/denoising (unsupervised) trains the encoder-decoder,
the head (supervised) makes z predict the unit cell + system -- a learned indexer feeding off the same code.
"""
import torch
import torch.nn as nn

NBINS = 1024
LATENT = 32
N_SYS = 7


class PowderAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        # encoder: 1024 -> 512 -> 256 -> 128 -> 64, channels 1->32->64->128->128
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, 2, 2), nn.ReLU(),
        )
        self.enc_fc = nn.Linear(128 * 64, latent)
        self.dec_fc = nn.Linear(latent, 128 * 64)
        # decoder: mirror, 64 -> 128 -> 256 -> 512 -> 1024
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 4, 2, 1),
        )
        self.head = nn.Sequential(nn.Linear(latent, 64), nn.ReLU())
        self.cell_out = nn.Linear(64, 6)              # normalised cell params in [0,1]
        self.sys_out = nn.Linear(64, N_SYS)           # crystal-system logits

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 64)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        h = self.head(z)
        return recon, torch.sigmoid(self.cell_out(h)), self.sys_out(h), z
