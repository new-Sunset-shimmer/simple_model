"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

# from torchtext.legacy.data import BucketIterator
# from torchtext.legacy.data import Field
# from torchtext.legacy.datasets.language_modeling import WikiText2
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel.data_parallel import DataParallel

class DataLoader:

    def __init__(self, ext, init_token, eos_token, tokenizer, detokenizer):
        self.ext = ext
        self.init_token = init_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        print('dataset initializing start')

    def make_dataset(self):
        # if self.ext == ('.de', '.en'):
        #     self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
        #                         lower=True, batch_first=True)
        #     self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
        #                         lower=True, batch_first=True)

        # elif self.ext == ('.en', '.de'):
        #     self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
        #                         lower=True, batch_first=True)
        #     self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
        #                         lower=True, batch_first=True)

        # train_data, valid_data, test_data = WikiText2.splits(exts=self.ext, text_field=self.target, root = 'data')
        train_data = load_dataset("roneneldan/TinyStories", split='train')
        train_data = train_data.train_test_split(test_size=0.1)
        train_data, test_data = train_data['train'],train_data['test']
        valid_data = load_dataset("roneneldan/TinyStories", split='validation')
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, max_len):
        class trainer(Dataset):
        # class trainer(DataParallel):
            def __init__(self, texts, tokenizer, detokenizer, eos_token, labels=None, max_length=1024):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.detokenizer = detokenizer
                self.eos_token = eos_token
            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                
                encoding = self.tokenizer.encode(
                    text['text'] + self.eos_token,
                    allowed_special={self.eos_token}
                )
                if len(encoding) > self.max_length:     
                    encoding = encoding[:self.max_length]
                pad_len = self.max_length - len(encoding)
                encoding += [self.tokenizer.encode("<pad>"
                                ,allowed_special = {"<pad>"})[0]] * pad_len
                encoding = torch.tensor(encoding)
                return encoding

        train_iterator = trainer(train,self.tokenizer, self.detokenizer, eos_token=self.eos_token, max_length=max_len)
        valid_iterator = trainer(validate,self.tokenizer, self.detokenizer, eos_token=self.eos_token, max_length=max_len)
        test_iterator = trainer(test,self.tokenizer, self.detokenizer, eos_token=self.eos_token, max_length=max_len)

        # DataLoader for batching
        train_iterator = torch.utils.data.DataLoader(train_iterator, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_iterator = torch.utils.data.DataLoader(valid_iterator, batch_size=batch_size, shuffle=True, drop_last=True)
        test_iterator = torch.utils.data.DataLoader(test_iterator, batch_size=batch_size, shuffle=True, drop_last=True)        
        return train_iterator, valid_iterator, test_iterator
