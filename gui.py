import tkinter as tk
from collections import Counter

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TRAIN_TEXT_FILE_PATH = 'train_text.txt'

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
# 

with open(TRAIN_TEXT_FILE_PATH, encoding="utf8") as text_file:
    text_sample = text_file.readlines()
text_sample = ' '.join(text_sample)

def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key = lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])
    
    return sequence, char_to_idx, idx_to_char

sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)

SEQ_LEN = 256
BATCH_SIZE = 16

def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start: batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)
    
def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text
    
    _, hidden = model(train, hidden)
        
    inp = train[-1].view(-1, 1, 1)
    
    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char
    
    while predicted_char!="." and predicted_char!="!":
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char
        
    return predicted_text
    
class TextRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))
               
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=5, 
    verbose=True, 
    factor=0.5
)

n_epochs = 50000
loss_avg = []

model.load_state_dict(torch.load("model.pt"))
model.eval()

def generate():
    text_box_out.delete("1.0", tk.END)
    #text_box_out.insert("1.0", "dfsdfsdf")
    text = text_box_in.get("1.0", tk.END)

    if text.find("олот"):
        Z = "золот"
    else: 
        Z = "серебр"

    if text.find("ерьг"):
        S = "серьги"
        K = "серьги"
        k = "украшение"
    else: 
        S = "кольцо"
        K = "кольца"
        k = "кольцо"

    if text.find("квама"):
        J = "аквамарины"
        j = "аквамарин"
        t = "аквамарином"
    elif text.find("етис"):
        J = "аметитсты"
        j = "аметист"
        t = "аметистом"
    elif text.find("етри"):
        J = "аметриныы"
        j = "аметрин"
        t = "аметрином"
    elif text.find("рюз"):
        J = "бирюзы"
        j = "бирюза"
        t = "бирюзой"
    elif text.find("иллиан"):
        J = "бриллианты"
        j = "бриллиант"
        t = "бриллиантом"
    elif text.find("орн"):
        J = "горные хрустали"
        j = "горный хрусталь"
        t = "горным хрусталём"
    elif text.find("ранат"):
        J = "гранаты"
        j = "гранат"
        t = "гранатом"
    elif text.find("емчу"):
        J = "жемчуга"
        j = "жемчуг"
        t = "жемчугом"
    elif text.find("зумру"):
        J = "изумруды"
        j = "изумруд"
        t = "изумрудом"
    elif text.find("олит"):
        J = "иолиты"
        j = "иолит"
        t = "иолитом"
    elif text.find("варц"):
        J = "кварцы"
        j = "кварц"
        t = "кварцом"
    elif text.find("орал"):
        J = "кораллы"
        j = "коралл"
        t = "кораллом"
    elif text.find("орунд"):
        J = "корунды"
        j = "корунд"
        t = "корундом"
    elif text.find("алахи"):
        J = "малахиты"
        j = "малахит"
        t = "малахиом"
    elif text.find("унн"):
        J = "лунные камни"
        j = "лунный камень"
        t = "лунным камнем"
    elif text.find("органи"):
        J = "морганиты"
        j = "морганит"
        t = "морганиом"
    elif text.find("никс"):
        J = "ониксы"
        j = "оникс"
        t = "ониксом"
    elif text.find("пал"):
        J = "опалы"
        j = "опал"
        t = "опалом"
    elif text.find("ерламут"):
        J = "перламутры"
        j = "перламутр"
        t = "перламутром"
    elif text.find("разиоли"):
        J = "празиолиты"
        j = "празиолит"
        t = "празиолитом"
    elif text.find("рени"):
        J = "прениты"
        j = "пренит"
        t = "пренитом"
    elif text.find("убин"):
        J = "рубины"
        j = "рубин"
        t = "рубином"
    elif text.find("апфи"):
        J = "сапфиры"
        j = "сапфир"
        t = "сапфиом"
    elif text.find("анзани"):
        J = "танзаниты"
        j = "танзанит"
        t = "танзанитом"
    elif text.find("опаз"):
        J = "топазы"
        j = "топаз"
        t = "топазом"
    elif text.find("урмали"):
        J = "турмалины"
        j = "турмалин"
        t = "турмалином"
    elif text.find("иниф"):
        J = "финифты"
        j = "финифт"
        t = "финифтом"
    elif text.find("алцедо"):
        J = "халцедоны"
        j = "халцедон"
        t = "халцедоном"
    elif text.find("ризоли"):
        J = "хризолиты"
        j = "хризолит"
        t = "хризолитом"
    elif text.find("итрин"):
        J = "цитрины"
        j = "цитрин"
        t = "цитрином"
    elif text.find("пинел"):
        J = "шпинелями"
        j = "шпинель"
        t = "шпинелью"
    elif text.find("маль"):
        J = "эмали"
        j = "эмаль"
        t = "эмалью"
    elif text.find("нтар"):
        J = "янтари"
        j = "янтарь"
        t = "янтарью"
    else: 
        J = "сияния"
        j = "блеск"
        t = "изяществом"
        #model.load_state_dict(torch.load("model_no_kamen.pt"))
        #model.eval()

    s = evaluate(
        model, 
        char_to_idx, 
        idx_to_char, 
        temp=0.3, 
        prediction_len=200, 
        start_text=text
        )
    s=s.replace("J",J)
    s=s.replace("j",j)
    s=s.replace("t",t)
    s=s.replace("Z",Z)
    s=s.replace("S",S)
    s=s.replace("K",K)
    s=s.replace("k",k)
    text_box_out.insert("1.0", s
    )
    


window = tk.Tk()
window.title("Description generator")

window.rowconfigure(0, minsize=50, weight=1)
window.columnconfigure([0, 1, 2], minsize=50, weight=1)

text_box_in = tk.Text()
text_box_in.grid(row=0, column=0, sticky="nsew")

text_box_out = tk.Text()
text_box_out.grid(row=0, column=2, sticky="nsew")

btn = tk.Button(master=window, text="Generate", command = generate)
btn.grid(row=0, column=1, sticky="nsew")
window.mainloop()