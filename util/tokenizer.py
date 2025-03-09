"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

import tiktoken


class Tokenizer:

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.decoder = tiktoken.get_encoding("cl100k_base")
        self.encoder = tiktoken.Encoding(
                # If you're changing the set of special tokens, make sure to use a different name
                # It should be clear from the name what behaviour to expect.
                name="cl100k_im",
                pat_str=self.encoder._pat_str,
                mergeable_ranks=self.encoder._mergeable_ranks,
                special_tokens={
                    **self.encoder._special_tokens,
                    "<pad>": 100264,
                    "<sos>": 100265,
                    "<eos>": 100266,
                }
            )
    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
