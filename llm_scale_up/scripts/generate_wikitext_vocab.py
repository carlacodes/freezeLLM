import re
from collections import Counter, OrderedDict
from typing import Iterator, List, Tuple

import torch
from datasets import load_dataset


class Vocab:
    def __init__(self, token_to_index, specials):
        self.token_to_index = token_to_index
        self.index_to_token = {idx: token for token, idx in token_to_index.items()}
        self.specials = specials
        # Assuming <unk> is your unknown token.
        self.default_index = token_to_index.get("<unk>")

    def __len__(self):
        return len(self.token_to_index)

    def __getitem__(self, token):
        # Return the index for a token, or the default_index if not found.
        return self.token_to_index.get(token, self.default_index)

    def set_default_index(self, index):
        self.default_index = index


class WikiText2Dataset:
    DATASET_NAME = "Salesforce/wikitext"
    CONFIG_NAME = "wikitext-2-v1"

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _hf_dataset_to_line_iterator(self, hf_split_dataset) -> Iterator[str]:
        for item in hf_split_dataset:
            text_line = item.get("text", "").strip()
            if (
                text_line
                and not text_line.startswith(" = ")
                and not text_line.startswith("= ")
            ):
                yield text_line

    def __call__(self) -> Tuple[Iterator[str], Iterator[str], Iterator[str]]:
        train_hf_ds = load_dataset(
            self.DATASET_NAME, self.CONFIG_NAME, split="train", cache_dir=self.cache_dir
        )
        valid_hf_ds = load_dataset(
            self.DATASET_NAME,
            self.CONFIG_NAME,
            split="validation",
            cache_dir=self.cache_dir,
        )
        test_hf_ds = load_dataset(
            self.DATASET_NAME, self.CONFIG_NAME, split="test", cache_dir=self.cache_dir
        )
        return (
            self._hf_dataset_to_line_iterator(train_hf_ds),
            self._hf_dataset_to_line_iterator(valid_hf_ds),
            self._hf_dataset_to_line_iterator(test_hf_ds),
        )


def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b|[^\w\s]", text)


TOKENIZER = basic_english_tokenizer
UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN = (
    "<unk>",
    "<pad>",
    "<mask>",
    "<bos>",
    "<eos>",
)
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def yield_tokens(data_iter, tokenizer_fn):
    for text in data_iter:
        yield tokenizer_fn(text)


def build_vocab_from_iterator(iterator, min_freq, specials):
    token_counts = Counter()
    for tokens in iterator:
        token_counts.update(tokens)
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: (x[1], x[0]), reverse=True
    )
    filtered_tokens = [
        token for token, count in sorted_by_freq_tuples if count >= min_freq
    ]
    vocab_list = OrderedDict()
    for token in specials:
        vocab_list[token] = len(vocab_list)
    for token in filtered_tokens:
        if token not in vocab_list:
            vocab_list[token] = len(vocab_list)
    token_to_index = {token: idx for idx, token in enumerate(vocab_list.keys())}

    # Now, instead of defining the class here, we just create an instance of it.
    return Vocab(token_to_index, specials)


def build_vocab_from_wikitext2(min_freq=5):
    dataset_loader = WikiText2Dataset()
    train_iter, _, _ = dataset_loader()
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, TOKENIZER),
        min_freq=min_freq,
        specials=SPECIAL_TOKENS,
    )
    return vocab


if __name__ == "__main__":
    print("Deterministically rebuilding vocabulary from WikiText-2...")

    # This will build the exact same vocab object as before
    vocab = build_vocab_from_wikitext2(min_freq=5)

    VOCAB_SIZE = len(vocab)
    print(f"Successfully built vocabulary. Size: {VOCAB_SIZE}")

    # Define the path where you want to save it
    VOCAB_PATH = "../models/wikitext2_vocab.pth"

    # Save the vocabulary object
    torch.save(vocab, VOCAB_PATH)
    print(f"Vocabulary saved to {VOCAB_PATH}")
    print("You can now use this file in your fine-tuning script. Exiting.")
