import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class MonoLingualModel(nn.Module):
    def __init__(self, en_pretrained_model='bert-base-uncased',
                 vi_pretrained_model='vinai/phobert-base',
                 num_classes=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.en_bert = AutoModel.from_pretrained(en_pretrained_model)
        self.en_tokenizer = AutoTokenizer.from_pretrained(en_pretrained_model)
        self.vi_bert = AutoModel.from_pretrained(vi_pretrained_model)
        self.vi_tokenizer = AutoTokenizer.from_pretrained(vi_pretrained_model)

        self.dropout = nn.Dropout(dropout)

        en_hidden_size = self.en_bert.config.hidden_size
        vi_hidden_size = self.vi_bert.config.hidden_size
        num_features = en_hidden_size + vi_hidden_size
        self.linear = nn.Linear(num_features, num_classes)

        self.device = device

    def forward(self, batch):
        en_sentences, vi_sentences = batch

        tokens = self.en_tokenizer(en_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokens = tokens.to(self.device)
        en_vec = self.en_bert(**tokens).pooler_output

        tokens = self.vi_tokenizer(vi_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokens = tokens.to(self.device)
        vi_vec = self.vi_bert(**tokens).pooler_output

        vec = torch.cat([en_vec, vi_vec], axis=1)
        output = self.linear(self.dropout(vec))
        return output


class SiameseModel(nn.Module):
    def __init__(self, model='base',
                 num_classes=3, dropout=0.1, device='cpu'):
        super().__init__()

        if model == 'base':
            en_pretrained_model = 'bert-base-uncased'
            vi_pretrained_model = 'vinai/phobert-base'
        else:
            en_pretrained_model = 'bert-large-uncased'
            vi_pretrained_model = 'vinai/phobert-large'

        self.en_bert = AutoModel.from_pretrained(en_pretrained_model)
        self.en_tokenizer = AutoTokenizer.from_pretrained(en_pretrained_model)
        self.vi_bert = AutoModel.from_pretrained(vi_pretrained_model)
        self.vi_tokenizer = AutoTokenizer.from_pretrained(vi_pretrained_model)

        self.dropout = nn.Dropout(dropout)

        en_hidden_size = self.en_bert.config.hidden_size
        vi_hidden_size = self.vi_bert.config.hidden_size

        assert en_hidden_size == vi_hidden_size, "The pre-trained models do not match in size"

        num_features = 4 * en_hidden_size
        self.linear = nn.Linear(num_features, num_classes)

        self.device = device

    def forward(self, batch):
        en_sentences, vi_sentences = batch

        tokens = self.en_tokenizer(en_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokens = tokens.to(self.device)
        en_vec = self.en_bert(**tokens).pooler_output

        tokens = self.vi_tokenizer(vi_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokens = tokens.to(self.device)
        vi_vec = self.vi_bert(**tokens).pooler_output

        vec = torch.cat([en_vec, vi_vec, torch.abs(en_vec - vi_vec), en_vec * vi_vec], axis=1)
        output = self.linear(self.dropout(vec))
        return output


class MultiLingualModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-multilingual-uncased',
                 num_classes=3, dropout=0.1, device='cpu'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        self.dropout = nn.Dropout(dropout)
        num_features = self.bert.config.hidden_size
        self.linear = nn.Linear(num_features, num_classes)
        self.device = device

    def forward(self, batch):
        sentences = [f"{e['en_sentence']['text']} [SEP] {e['vi_sentence']['text']}" for e in batch]

        tokens = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokens = tokens.to(self.device)

        output = self.bert(**tokens).last_hidden_state[:, 0, :]

        output = self.linear(self.dropout(output))
        return output
