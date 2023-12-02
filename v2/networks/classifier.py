from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification

class ViTClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        
    def forward(self, img):
        out = self.vit(img.to(self.vit.device),
                       interpolate_pos_encoding=True)
        
        return out['logits'].detach()