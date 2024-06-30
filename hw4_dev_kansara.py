from collections import defaultdict
import os
import operator
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import operator
import shutil
import gzip
from typing import List, Tuple

count_dict = defaultdict(int)
label_set = []

with open("data/train", "r") as f:
    for line in f:
        words = line.split()
        if words:
            count_dict[words[1]] += 1
            if words[2] not in label_set:
                label_set.append(words[2])

unkw = sum(val for val in count_dict.values() if val < 2)

sorted_count_list = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)

word_index = {word: i for i, (word, count) in enumerate(sorted_count_list, start=2) if count >= 2}
word_index['<PAD>'] = 0
word_index['<UNK>'] = 1


def process_file(file_path, has_tags=True):
    sentences = []
    tags = []
    curr_sent = ""
    curr_tags = ""

    with open(file_path, "r") as f:
        for line in f:
            get_line = line.split()
            if len(get_line) > 0:
                curr_sent += get_line[1] + " "
                if has_tags:
                    curr_tags += get_line[2] + " "
            else:
                sentences.append(curr_sent[:-1])
                if has_tags:
                    tags.append(curr_tags[:-1])
                curr_sent = ""
                curr_tags = ""

    sentences.append(curr_sent[:-1])
    if has_tags:
        tags.append(curr_tags[:-1])

    if has_tags:
        return pd.DataFrame({'sentences': sentences, 'tags': tags})
    else:
        return pd.DataFrame({'sentences': sentences})

train_data = process_file("./data/train")
dev_data = process_file("./data/dev")
test_data = process_file("./data/test", has_tags=False)


label_index = {}
i=0
for label in label_set:
    label_index[label] = i
    i+=1
label_index['pad_label'] = -1

index_word = {v: k for k, v in word_index.items()}
index_label = {v: k for k, v in label_index.items()}


class TrainDataBiLSTM:
    def __init__(self, sentences: pd.Series, tags: pd.Series, word_index: dict, label_index: dict):
        self.sentences = sentences
        self.tags = tags
        self.word_index = word_index
        self.label_index = label_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, ner_tag

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sentences, ner_tags = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    
    ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, lengths, ner_tags


class DevDataBiLSTM:
    def __init__(self, sentences: pd.Series, tags: pd.Series, word_index: dict, label_index: dict):
        self.sentences = sentences
        self.tags = tags
        self.word_index = word_index
        self.label_index = label_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, ner_tag

class TestDataBiLSTM:
    def __init__(self, sentences: pd.Series, word_index: dict):
        self.sentences = sentences
        self.word_index = word_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> torch.Tensor:
        sentence = self.sentences.iloc[i].split()
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
        sentence = torch.tensor(sentence)
        return sentence

def pad_collate_test(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    sentences = batch
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    return sentences, lengths


def create_data_loader(dataset, batch_size, collate_fn):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

batch_size = 16

train_dataset = TrainDataBiLSTM(train_data['sentences'], train_data['tags'], word_index, label_index)
train_loader = create_data_loader(train_dataset, batch_size, pad_collate)

dev_dataset = DevDataBiLSTM(dev_data['sentences'], dev_data['tags'], word_index, label_index)
dev_loader = create_data_loader(dev_dataset, batch_size, pad_collate)

test_dataset = TestDataBiLSTM(test_data['sentences'], word_index)
test_loader = create_data_loader(test_dataset, batch_size, pad_collate_test)


# Task 1


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_hidden_dim: int, lstm_dropout: float, linear_output_dim: int, num_tags: int):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.elu = nn.ELU(0.35)
        self.classifier = nn.Linear(linear_output_dim, num_tags)
    
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        linear_output = self.linear(output)
        elu_output = self.elu(linear_output)
        logits = self.classifier(elu_output)
        
        return logits

bilstm_model = BiLSTM(len(word_index.keys()), 100, 256, 0.33, 128, 9)
print(bilstm_model)


def create_optimizer_and_criterion(model, learning_rate, ignore_index):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return optimizer, criterion

optimizer, criterion = create_optimizer_and_criterion(bilstm_model, 0.33, -1)


def train_one_epoch(model, optimizer, criterion, data_loader):
    model.train()
    total_loss = 0.0
    
    for sentences, lengths, labels in data_loader:
        optimizer.zero_grad()
        output = model(sentences, lengths)
        output = output.permute(0,2,1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()*sentences.size(0)
    
    return total_loss / len(data_loader.dataset)

epochs = 30

for epoch in range(epochs):
    train_loss = train_one_epoch(bilstm_model, optimizer, criterion, train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))


torch.save(bilstm_model.state_dict(), 'blstm1.pt')


def load_model(model_class, model_path, word_index, embedding_dim, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_tags):
    model = model_class(len(word_index.keys()), embedding_dim, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_tags)
    model.load_state_dict(torch.load(model_path))
    return model

bilstm_model = load_model(BiLSTM, 'blstm1.pt', word_index, 100, 256, 0.33, 128, 9)


def getDevResults(model, dataloader):
    model.eval()
    
    with open("./data/dev","r") as f_read, open("dev1.out","w") as f_write:
        for sentences, lengths, labels in dataloader:
            output = model(sentences, lengths)
            max_values, max_indices = torch.max(output, dim=2)
            y = max_indices
            
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    read_line = f_read.readline().split()
                    if len(read_line)>0:
                        f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
                    else:
                        break
                    if j+1>=len(sentences[i]):
                        f_read.readline()
                if len(sentences)==batch_size or i<len(sentences)-1:
                    f_write.write("\n")


getDevResults(bilstm_model, dev_loader)


def getTestResults(model, dataloader):
    model.eval()
    
    with open("./data/test","r") as f_read, open("test1.out","w") as f_write:
        for sentences, lengths in dataloader:
            output = model(sentences, lengths)
            max_values, max_indices = torch.max(output, dim=2)
            y = max_indices
            
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    read_line = f_read.readline().split()
                    if len(read_line)>0:
                        f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
                    else:
                        break
                    if j+1>=len(sentences[i]):
                        f_read.readline()
                if len(sentences)==batch_size or i<len(sentences)-1:
                    f_write.write("\n")


getTestResults(bilstm_model, test_loader)

# Task 2

embed_vectors = []
embed_vocab = []

if not os.path.exists('glove.6B.100d'):
    with gzip.open('glove.6B.100d.gz', 'rb') as f_in, open('glove.6B.100d', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with open("glove.6B.100d","r") as file_embed:
    for line in file_embed:
        line = line.split()
        embed_vocab.append(line[0])
        embed_vectors.append(line[1:])


embed_vocab = np.asarray(embed_vocab)
embed_vectors = np.asarray(embed_vectors, dtype=np.float64)



pad_vector = np.zeros((1, embed_vectors.shape[1]))
unk_vector = np.mean(embed_vectors, axis=0, keepdims=True)

embed_vocab = np.concatenate([['<PAD>', '<UNK>'], embed_vocab])

embed_vectors = np.concatenate([pad_vector, unk_vector, embed_vectors], axis=0)


embed_vectors_tensor = torch.from_numpy(embed_vectors)
demo_embed: nn.Embedding = nn.Embedding.from_pretrained(embed_vectors_tensor, padding_idx=0)


#demo_embed(torch.LongTensor([2]))
#embed_vectors[2]
#embed_vocab


glove_word_index = {word: index for index, word in enumerate(embed_vocab)}

class TrainDataBiLSTMGlove:
    def __init__(self, sentences: List[str], tags: List[str], glove_word_index: dict, label_index: dict):
        self.sentences = sentences
        self.tags = tags
        self.glove_word_index = glove_word_index
        self.label_index = label_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, is_capital, ner_tag

def pad_collate_glove(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sentences, is_capitals, ner_tags = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    is_capitals = pad_sequence(is_capitals, batch_first=True, padding_value=-1)
    ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, is_capitals, lengths, ner_tags

class DevDataBiLSTMGlove:
    def __init__(self, sentences: List[str], tags: List[str], glove_word_index: dict, label_index: dict):
        self.sentences = sentences
        self.tags = tags
        self.glove_word_index = glove_word_index
        self.label_index = label_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, is_capital, ner_tag

class TestDataBiLSTMGlove:
    def __init__(self, sentences: List[str], glove_word_index: dict):
        self.sentences = sentences
        self.glove_word_index = glove_word_index
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
        
        return sentence, is_capital

def pad_collate_glove_test(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sentences, is_capitals = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    is_capitals = pad_sequence(is_capitals, batch_first=True, padding_value=-1)
    
    return sentences, is_capitals, lengths


def create_data_loader(dataset, batch_size, collate_fn):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

train_dataset = TrainDataBiLSTMGlove(train_data['sentences'], train_data['tags'], glove_word_index, label_index)
train_loader = create_data_loader(train_dataset, batch_size, pad_collate_glove)

dev_dataset = DevDataBiLSTMGlove(dev_data['sentences'], dev_data['tags'], glove_word_index, label_index)
dev_loader = create_data_loader(dev_dataset, batch_size, pad_collate_glove)

test_dataset = TestDataBiLSTMGlove(test_data['sentences'], glove_word_index)
test_loader = create_data_loader(test_dataset, batch_size, pad_collate_glove_test)


class BiLSTMGlove(nn.Module):
    def __init__(self, embedding_dim: int, lstm_hidden_dim: int, lstm_dropout: float, linear_output_dim: int, num_tags: int):
        super(BiLSTMGlove, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embed_vectors), padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim+1, hidden_size=lstm_hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_tags)
    
    def forward(self, inputs: torch.Tensor, is_capitals: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        concatenated_tensor = torch.cat((embedded, is_capitals.unsqueeze(-1)), dim=-1)
        packed_embedded = pack_padded_sequence(concatenated_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_embedded = packed_embedded.float()
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        linear_output = self.linear(output)
        elu_output = self.elu(linear_output)
        logits = self.classifier(elu_output)
        
        return logits

bilstm_glove_model = BiLSTMGlove(100, 256, 0.33, 128, 9)
print(bilstm_glove_model)


criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=-1)
optimizer: torch.optim.Optimizer = torch.optim.SGD(bilstm_glove_model.parameters(), lr=0.33)


def train_model(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, train_dataset: Dataset) -> float:
    train_loss = 0.0
    model.train()
    
    for sentences, is_capitals, lengths, labels in train_loader:
        optimizer.zero_grad()
        output = model(sentences, is_capitals, lengths)
        output = output.permute(0,2,1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*sentences.size(0)
    
    return train_loss / len(train_dataset)

epochs = 30

for epoch in range(epochs):
    train_loss = train_model(bilstm_glove_model, criterion, optimizer, train_loader, train_dataset)
    print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}')


torch.save(bilstm_glove_model.state_dict(), 'blstm2.pt')


bilstm_glove_model = BiLSTMGlove(100, 256, 0.33, 128, 9)
bilstm_glove_model.load_state_dict(torch.load('blstm2.pt'))


def get_dev_results_glove(model: nn.Module, dataloader: DataLoader) -> None:
    model.eval()
    
    with open("./data/dev", "r") as f_read, open("dev2.out", "w") as f_write:
        for sentences, is_capitals, lengths, labels in dataloader:
            output = model(sentences, is_capitals, lengths)
            _, max_indices = torch.max(output, dim=2)
            
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    read_line = f_read.readline().split()
                    if read_line:
                        f_write.write(f"{read_line[0]} {read_line[1]} {index_label[max_indices[i][j].item()]}\n")
                    else:
                        break
                    if j+1 >= len(sentences[i]):
                        f_read.readline()
                if len(sentences) == batch_size or i < len(sentences) - 1:
                    f_write.write("\n")

get_dev_results_glove(bilstm_glove_model, dev_loader)


def get_test_results_glove(model: nn.Module, dataloader: DataLoader) -> None:
    model.eval()
    
    with open("./data/test", "r") as f_read, open("test2.out", "w") as f_write:
        for sentences, is_capitals, lengths in dataloader:
            output = model(sentences, is_capitals, lengths)
            _, max_indices = torch.max(output, dim=2)
            
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    read_line = f_read.readline().split()
                    if read_line:
                        f_write.write(f"{read_line[0]} {read_line[1]} {index_label[max_indices[i][j].item()]}\n")
                    else:
                        break
                    if j+1 >= len(sentences[i]):
                        f_read.readline()
                if len(sentences) == batch_size or i < len(sentences) - 1:
                    f_write.write("\n")

get_test_results_glove(bilstm_glove_model, test_loader)