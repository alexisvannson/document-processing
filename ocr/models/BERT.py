from transformers import BertLMHeadModel, BertTokenizerFast


def build_bert_decoder(name="bert-base-cased"):
    """
    Pretrained BERT turned into a causal (left-to-right) decoder. The cross-attention layers
    that attend to the ViT patch tokens don't exist in BERT, so they start randomly initialised.
    """
    return BertLMHeadModel.from_pretrained(name, is_decoder=True, add_cross_attention=True)


class CharTokenizer:
    """
    Character-level tokenizer over BERT's vocabulary.

    WordPiece splits "16,500" into "16 , 500" and decoding can't tell whether there were
    spaces in between, so every character gets its own token instead. BERT's cased vocab
    has a token for every printable ASCII character; spaces map to [unused1].
    """

    def __init__(self, name="bert-base-cased"):
        self.bert = BertTokenizerFast.from_pretrained(name)
        vocab = self.bert.get_vocab()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.cls_id = self.bert.cls_token_id  # decoder start token
        self.sep_id = self.bert.sep_token_id  # end of sequence
        self.pad_id = self.bert.pad_token_id
        self.unk_id = self.bert.unk_token_id
        self.space_id = vocab.get("[unused1]", self.unk_id)

        self.id_to_char = {i: t for t, i in vocab.items() if len(t) == 1}
        self.id_to_char[self.space_id] = " "

    def encode(self, text):
        """Target ids: one per character, then [SEP]. The model prepends [CLS] itself."""
        ids = [self.space_id if c == " " else self.vocab.get(c, self.unk_id) for c in text]
        return ids + [self.sep_id]

    def decode(self, ids):
        chars = []
        for i in ids:
            i = int(i)
            if i == self.sep_id:
                break
            if i in (self.cls_id, self.pad_id):
                continue
            chars.append(self.id_to_char.get(i, "?"))
        return "".join(chars)

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
