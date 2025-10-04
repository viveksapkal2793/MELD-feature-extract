import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class BlipImageEvalProcessor:
    def __init__(self, image_size=448, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size), 
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, image):
        return self.transform(image)