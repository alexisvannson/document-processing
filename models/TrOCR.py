from transformers import VisionEncoderDecoderModel

from models.BERT import build_bert_decoder
from models.VIT import build_vit_encoder


def build_trocr(tokenizer, encoder_name="google/vit-base-patch16-384", decoder_name="bert-base-cased",
                max_length=40):
    """
    TrOCR-style recognizer: ViT encoder -> BERT decoder with cross-attention.
    Passing `labels` to forward() gives the teacher-forced cross-entropy loss
    (labels are shifted right behind [CLS] to build the decoder input).
    """
    model = VisionEncoderDecoderModel(
        encoder=build_vit_encoder(encoder_name),
        decoder=build_bert_decoder(decoder_name),
    )
    for config in (model.config, model.generation_config):
        config.decoder_start_token_id = tokenizer.cls_id
        config.pad_token_id = tokenizer.pad_id
        config.eos_token_id = tokenizer.sep_id
    model.generation_config.max_length = max_length
    model.generation_config.num_beams = 1
    return model
