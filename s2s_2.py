import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import torch.optim as optim

import sentencepiece as spm
import csv

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

#encoder, decoder input, out
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
        temp.append(1)
        i += 1
    encoder_index.append(temp)

decoder_input_index=[]
sp.SetEncodeExtraOptions('bos')
for str in decoder:
    temp = sp.encode(str, out_type=int)
    i = len(temp)
    while i < max_decoder:
        temp.append(1)
        i += 1
    decoder_input_index.append(temp)

decoder_output_index=[]
sp.SetEncodeExtraOptions('eos')
for str in decoder:
    temp = sp.encode(str, out_type=int)
    i = len(temp)
    while i < max_decoder:
        temp.append(1)
        i += 1
    decoder_output_index.append(temp)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer_encoder = nn.Embedding(num_embeddings = sp_e.GetPieceSize(), embedding_dim = 128, padding_idx = 1)
        self.embedding_layer_decoder = nn.Embedding(num_embeddings = sp.GetPieceSize(), embedding_dim = 128, padding_idx = 1)
        self.W1 = nn.LSTM(input_size=128, hidden_size=1024, dropout=0.1, batch_first=True, num_layers = 2, bidirectional = True)
        self.W2 = nn.LSTM(input_size=128, hidden_size=1024, dropout=0.1, batch_first=True, num_layers = 2, bidirectional = True)
        self.W3 = nn.Linear(1024 * 2, sp.GetPieceSize())
        self.hn = None
        self.cn = None

    def forward(self, X = None, Y = None):
        if X != None:
            X = self.embedding_layer_encoder(X)
            out, (h0, c0) = self.W1(X)
            self.hn = h0
            self.cn = c0
        Y = self.embedding_layer_decoder(Y)
        out, (self.hn, self.cn) = self.W2(Y, (self.hn, self.cn))
        out = self.W3(out)
        return out


#trainning
batch_size = 400
model = Model()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.cuda()
model.train()

loss_fn = nn.CrossEntropyLoss(ignore_index = 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

encoder_index = torch.tensor(encoder_index)
decoder_input_index = torch.tensor(decoder_input_index)
decoder_output_index = torch.tensor(decoder_output_index)

print("start")
print("encoder shape : ",encoder_index.shape)
print("decoder shape : ",decoder_output_index.shape)
print(max_encoder)
print(max_decoder)

batch_num = int(encoder_index.shape[0] / batch_size)

for epoch in range(500):
    loss_epoch=0
    for i in range(batch_num):
        output = model(encoder_index[batch_size*i:batch_size*(i+1),:].cuda(), decoder_input_index[batch_size*i:batch_size*(i+1),:].cuda())
        output = torch.transpose(output, 1, 2)
        loss = loss_fn(output, decoder_output_index[batch_size*i:batch_size*(i+1),:].cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss

    loss_epoch = loss_epoch / batch_num
    print('Epoch:', '%04d' % (epoch + 1), ' cost =', '{:.6f}'.format(loss_epoch))
    torch.save(model.module.state_dict(), 'weights_only.pth')




#translate test
device = torch.device("cuda")
model = Model()
model.load_state_dict(torch.load('weights_only.pth'))
model.to(device)
model.eval()

def translate(sequence = ""): #teacher forcing x
    encoder_index=sp_e.encode(sequence, out_type=int)
    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode("Do", out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()
    target = []
    stop = 0

    output = model(X = encoder_index, Y = decoder_input_index)
    output = torch.argmax(output, dim=2)
    print(decoder_input_index)
    print(output)
    
    decoder_input_index = output
    stop = output.tolist()[0][0]
    target.append(stop)

    while stop != 3:
        output = model(Y = decoder_input_index)
        output = torch.argmax(output, dim=2)
        decoder_input_index = output
        stop = output.tolist()[0][0]
        target.append(stop)

        if len(target) == 65:
            break
        
    print("번역문 : ",sp.DecodeIds(target))

def translate_test(sequence = "", sequence2 = ""): #teacher forcing o
    encoder_index=sp_e.encode(sequence, out_type=int)
    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode(sequence2, out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()
    stop = 0

    output = model(X = encoder_index, Y = decoder_input_index)
    output = torch.argmax(output, dim=2)
    stop = output.tolist()[0]

    print("teacher forcing 번역문 : ",sp.DecodeIds(stop))

def log(number):
    return torch.log(number + 1e-10).tolist()

def beam_search(sequence = ""):
    encoder_index=sp_e.encode(sequence, out_type=int)
    encoder_index = torch.tensor(encoder_index).unsqueeze(0).to(device)

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode("", out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).to(device)

    row=3 #beam_search_number
    beam_list = [[] for i in range(row)]
    stop=[]
    num = 0
    softmax = nn.Softmax(dim=2)

    output = model(encoder_index, decoder_input_index)
    output = softmax(output)
    sorted, indices = torch.sort(output, 2)

    for i in range(row):#parent tree
        temp=[[2], 1.0]
        temp[0].append(indices[0,0,-(i+1)].tolist())
        temp[1]=temp[1] * log(sorted[0,0,-(i+1)])
        beam_list[i].append(temp)

    len=0
    while num != row and len < max_decoder:#beam search for beam_number and english_vocab_len
        len += 1
        for i in range(row): #child tree
            decoder_input_index = torch.tensor(beam_list[i][-1][0]).unsqueeze(0).to(device)
            output = model(Y = decoder_input_index)
            output = softmax(output)
            sorted, indices = torch.sort(output, 2)

            temp=beam_list[i][-1].copy()
            temp[0].append(indices[0,0,-(1)].tolist())
            temp[1]=temp[1] * log(sorted[0,0,-(1)])
            if temp[1] == 3 or len == max_decoder:
                num += 1
                stop.append(temp)

                temp=beam_list[i][-1].copy()
                temp[0].append(indices[0,0,-(2)].tolist())
                temp[1]=temp[1] * log(sorted[0,0,-(2)])
                beam_list[i].append(temp)
            else:
                beam_list[i].append(temp)


    stop.sort(key=lambda x: x[1])
    print("beam search 번역문 : ",sp.DecodeIds(stop[0][0]))


sequence = "씨티은행에서 일하세요?"
sequence2 = "Do you work at a City bank?"
print("원문 : ",sequence)
print("번역 원문: ",sequence2)
translate_test(sequence, sequence2)
translate(sequence)


sequence = "푸리토의 베스트셀러는 해외에서 입소문만으로 4차 완판을 기록하였다."
sequence2 = "PURITO's bestseller, which recorded 4th rough -cuts by words of mouth from abroad."
print("원문 : ",sequence)
print("번역 원문: ",sequence2)
translate_test(sequence, sequence2)
translate(sequence)
#beam_search(sequence)


sequence = "마치 목욕탕 창구처럼 보일까말까 한 작은 구멍으로 내가 돈을 주면 그 여자가 교통카드를 충전시켜주었던 기억이 납니다."
sequence2 = "I remember that the person recharged my transportation card when I gave her money through a tiny hole, just like the ticket office for the public bath."
print("원문 : ",sequence)
print("번역 원문: ",sequence2)
translate_test(sequence, sequence2)
translate(sequence)
#beam_search(sequence)


sequence = "우리는 언제나 서로를 사랑하려고 노력해요."
sequence2 = "We always try and show our love for each other."
print("원문 : ",sequence)
print("번역 원문: ",sequence2)
translate_test(sequence, sequence2)
translate(sequence)
#beam_search(sequence)


sequence = "인권위가 설립된 20년 전 평화적 정권교체로 정치적 자유가 크게 신장됐다."
print("원문 : ",sequence)
translate(sequence)
#beam_search(sequence)
