import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import torch.optim as optim

import sentencepiece as spm

import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sentence import vocab_make


import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

SEED = 1234

random.seed(SEED)

vocab_make()

df = pd.read_excel('conversation.xlsx', engine='openpyxl')
df = df.loc[:, '원문':'번역문']

#encoder vocab
sp_e = spm.SentencePieceProcessor()
vocab_file = "translate_korea.model"
sp_e.load(vocab_file)

#decoder_vocab
sp = spm.SentencePieceProcessor()
vocab_file = "translate_english.model"
sp.load(vocab_file)

#encoder, decoder input, output
encoder = list(df.원문)
decoder = list(df.번역문)

#data to embedding
max_encoder = max([len(sp_e.encode(line, out_type=int)) for line in encoder])
max_decoder = max([len(sp.encode(line, out_type=int)) for line in decoder]) + 1

encoder_index=[]
for str in encoder:
    temp = sp_e.encode(str, out_type=int)
    i = len(temp)
    while i < max_encoder:
        temp.append(0)
        i += 1
    encoder_index.append(temp)

decoder_input_index=[]
sp.SetEncodeExtraOptions('bos')
for str in decoder:
    temp = sp.encode(str, out_type=int)
    i = len(temp)
    while i < max_decoder:
        temp.append(0)
        i += 1
    decoder_input_index.append(temp)

decoder_output_index=[]
sp.SetEncodeExtraOptions('eos')
for str in decoder:
    temp = sp.encode(str, out_type=int)
    i = len(temp)
    while i < max_decoder:
        temp.append(0)
        i += 1
    decoder_output_index.append(temp)


class CustomDataset(Dataset):
    def __init__(self, x, y, z):
        self.x_data = x
        self.y_data = y
        self.z_data = z

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])
        y = torch.tensor(self.y_data[idx])
        z = torch.tensor(self.z_data[idx])
        return x, y, z

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True, bidirectional = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True, bidirectional = True)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)

        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

class Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.9):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device=self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

#trainning
def train(gpu, args):
    ############################################################
    print("gpu",gpu)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    device = gpu

    INPUT_DIM = sp_e.GetPieceSize()
    OUTPUT_DIM = sp.GetPieceSize()
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 1024
    N_LAYERS = 2
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)


    model = Model(enc, dec, device)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    loss_fn = nn.CrossEntropyLoss(ignore_index = 0).cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

#######################################################################################
    # checkpoint= torch.load('amp_checkpoint.pt', map_location=torch.device('cpu'))
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    model = DDP(model)
#######################################################################################
    dataset = CustomDataset(encoder_index, decoder_input_index, decoder_output_index)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    dataloader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

    if gpu == 0:
        print("start")

    for epoch in range(args.epochs):
        loss_epoch = 0
        for encoder_input, decoder_input, decoder_output in dataloader:

            encoder_input = encoder_input.cuda()
            decoder_input = decoder_input.cuda()
            decoder_output = decoder_output.cuda()

            output = model(encoder_input, decoder_input)
            output=torch.transpose(output, 1, 2)
            loss = loss_fn(output, decoder_output)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            loss_epoch += loss

        loss_epoch = loss_epoch / len(dataloader)
        if gpu == 0:
            print('Epoch:', '%04d' % (epoch + 1), ' cost =', '{:.6f}'.format(loss_epoch))
        #torch.save(model.module, 'weights_only.pth')
        #torch.save(model.module.state_dict(), 'weights_only.pth')
            checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '127.0.0.1'              #
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)         #
    #########################################################


################################translate test########################################
def translate(model, sequence = "", sequence2 = ""):
    encoder_index=sp_e.encode(sequence, out_type=int)

    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode(sequence2, out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()

    output = model(encoder_index, decoder_input_index, teacher_forcing_ratio = 0)
    output = torch.argmax(output, dim=2)
    target = output.tolist()[0]

    print("teacher_ratio = 0, 번역문 : ",sp.DecodeIds(target))

def translate_test(model, sequence = "", sequence2 = ""):
    encoder_index=sp_e.encode(sequence, out_type=int)

    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode(sequence2, out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()

    output = model(encoder_index, decoder_input_index, teacher_forcing_ratio = 1)
    output = torch.argmax(output, dim=2)
    target = output.tolist()[0]
    print("teacher_ratio = 1, 번역문 : ",sp.DecodeIds(target))

def beam_search(model, sequence = ""):
    encoder_index=sp_e.encode(sequence, out_type=int)

    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sequence2 = ""
    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode(sequence2, out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()

    output = model(encoder_index, decoder_input_index, teacher_forcing_ratio = 0)
    output = torch.argmax(output, dim=2)
    target = output.tolist()[0]
    print("teacher_ratio = 1, 번역문 : ",sp.DecodeIds(target))

def test():

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:8888',
        # init_method='env://',
        world_size=1,
        rank=0
    )
    torch.manual_seed(0)
    device = torch.device("cuda")
    INPUT_DIM = sp_e.GetPieceSize()
    OUTPUT_DIM = sp.GetPieceSize()
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 1024
    N_LAYERS = 2
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Model(enc, dec, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    checkpoint= torch.load('amp_checkpoint.pt')

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])


    sequence = "씨티은행에서 일하세요?"
    sequence2 = "Do you work at a City bank?"
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence, sequence2)


    sequence = "푸리토의 베스트셀러는 해외에서 입소문만으로 4차 완판을 기록하였다."
    sequence2 = "PURITO's bestseller, which recorded 4th rough -cuts by words of mouth from abroad."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence, sequence2)


    sequence = "마치 목욕탕 창구처럼 보일까말까 한 작은 구멍으로 내가 돈을 주면 그 여자가 교통카드를 충전시켜주었던 기억이 납니다."
    sequence2 = "I remember that the person recharged my transportation card when I gave her money through a tiny hole, just like the ticket office for the public bath."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence, sequence2)


    sequence = "우리는 언제나 서로를 사랑하려고 노력해요."
    sequence2 = "We always try and show our love for each other."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence, sequence2)

    sequence = "인권위가 설립된 20년 전 평화적 정권교체로 정치적 자유가 크게 신장됐다."
    sequence2 = ""
    print("원문 : ",sequence)
    translate(model, sequence, sequence2)

    print(max_encoder)
    print(max_decoder)
##########################################################################################################################

if __name__=="__main__":
    main()
    test()
