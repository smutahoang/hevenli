import torch.nn as nn
import random
import torch
import torch.optim as optim
from models import MonoLingualModel, SiameseModel, MultiLingualModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AdamW
from transformers import get_scheduler
import math

label_map = {
    'Entailment': 0, 'entailment': 0,
    'Contradiction': 1, 'contradiction': 1,
    'Neutral': 2, 'neutral': 2
}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_batch(batch, model, criterion, optimizer, scheduler, num_iterations=1):
    labels = torch.LongTensor([label_map[e['label']] for e in batch])
    labels = labels.to(model.device)
    if type(model) == MonoLingualModel:
        en_sentences = [e['en_sentence']['text'] for e in batch]
        vi_sentences = [e['vi_sentence']['tokenized_text'] for e in batch]
        for i in range(num_iterations):
            model.zero_grad()
            optimizer.zero_grad()
            output = model((en_sentences, vi_sentences))
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # print(f'output: ', output[0, 0].item())
            # print(f'linear_search:{i} \tloss: {loss.item()}')
        # print(f'loss: {loss.item()}')
        # print(f'lr: {get_lr(optimizer)}')
        return loss.item(), output
    elif type(model) == SiameseModel:
        en_sentences = [e['en_sentence']['text'] for e in batch]
        vi_sentences = [e['vi_sentence']['tokenized_text'] for e in batch]
        for _ in range(num_iterations):
            model.zero_grad()
            optimizer.zero_grad()
            output = model((en_sentences, vi_sentences))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        # print(f'loss = {loss.item()}')
        return loss.item(), output
    elif type(model) == MultiLingualModel:
        for _ in range(num_iterations):
            model.zero_grad()
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print(f'loss = {loss.item()}')
        return loss.item(), output
    else:
        pass


def eval_batch(batch, model, criterion):
    labels = torch.LongTensor([label_map[e['label']] for e in batch])
    labels = labels.to(model.device)
    if type(model) == MonoLingualModel:
        en_sentences = [e['en_sentence']['text'] for e in batch]
        vi_sentences = [e['vi_sentence']['tokenized_text'] for e in batch]
        output = model((en_sentences, vi_sentences))
        loss = criterion(output, labels)
        return loss.item(), output
    elif type(model) == SiameseModel:
        en_sentences = [e['en_sentence']['text'] for e in batch]
        # vi_sentences = [e['sentence_2']['text'] for e in batch]
        vi_sentences = [e['vi_sentence']['tokenized_text'] for e in batch]
        output = model((en_sentences, vi_sentences))
        loss = criterion(output, labels)
        return loss.item(), output
    elif type(model) == MultiLingualModel:
        output = model(batch)
        loss = criterion(output, labels)
        return loss.item(), output
    else:
        pass


def train_eval(model, train_data, test_data, num_epochs=20, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    test_labels = [label_map[e['label']] for e in test_data]
    performance_trace = []

    num_training_steps = num_epochs * math.ceil(len(train_data) / batch_size)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for e in range(num_epochs):
        model.train()
        random.shuffle(train_data)
        batches = []
        starting_index = 0
        while starting_index < len(train_data):
            batches.append(train_data[starting_index:min(starting_index + batch_size, len(train_data))])
            starting_index += batch_size
        train_loss = 0
        for bach in tqdm(batches):
            # for bach in batches:
            batch_loss, _ = train_batch(bach, model, criterion, optimizer, scheduler=lr_scheduler)
            train_loss += batch_loss
        train_loss /= len(batches)

        model.eval()
        with torch.no_grad():
            batches = []
            starting_index = 0
            while starting_index < len(test_data):
                batches.append(test_data[starting_index:min(starting_index + batch_size, len(train_data))])
                starting_index += batch_size
            test_loss = 0
            prediction = []
            for bach in tqdm(batches):
                batch_loss, output = eval_batch(bach, model, criterion)
                test_loss += batch_loss
                prediction.extend(list(torch.argmax(output, dim=1).detach().cpu().numpy()))
            test_loss /= len(batches)
        test_accuracy = accuracy_score(test_labels, prediction)
        print(f"epoch = {e} training_loss = {train_loss} test_loss = {test_loss} test_accuracy = {test_accuracy}")
        performance_trace.append({
            'epoch': e,
            'training_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })
    return performance_trace


def train(model, data, num_epochs=20, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    labels = [label_map[e['label']] for e in data]
    trace = []
    best_loss = float('inf')
    best_state = None
    for e in range(num_epochs):
        model.train_eval()
        random.shuffle(data)
        batches = []
        starting_index = 0
        while starting_index < len(data):
            batches.append(data[starting_index:min(starting_index + batch_size, len(data))])
            starting_index += batch_size
        train_loss = 0
        for bach in tqdm(batches):
            batch_loss, _ = train_batch(bach, model, criterion, optimizer)
            train_loss += batch_loss
        train_loss /= len(batches)

        model.eval()
        with torch.no_grad():
            batches = []
            starting_index = 0
            while starting_index < len(data):
                batches.append(data[starting_index:min(starting_index + batch_size, len(data))])
                starting_index += batch_size
            test_loss = 0
            prediction = []
            for bach in tqdm(batches):
                batch_loss, output = eval_batch(bach, model, criterion)
                test_loss += batch_loss
                prediction.extend(list(torch.argmax(output, dim=1).detach().cpu().numpy()))
            test_loss /= len(batches)
        accuracy = accuracy_score(labels, prediction)
        print(f"epoch = {e} training_loss = {train_loss} test_loss = {test_loss} test_accuracy = {accuracy}")
        trace.append({
            'epoch': e,
            'training_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': accuracy
        })
        if test_loss < best_loss:
            best_state = model.state_dict()

    return trace, best_state
