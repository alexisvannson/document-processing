from transformers import ViTModel


def build_vit_encoder(name="google/vit-base-patch16-384"):
    """
    ImageNet-pretrained ViT. It cuts the text crop into 16x16 patches and returns one
    token per patch; self-attention models the 2D layout of the characters.
    """
    return ViTModel.from_pretrained(name, add_pooling_layer=False)
