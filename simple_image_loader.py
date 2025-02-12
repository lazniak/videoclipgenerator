# simple_image_loader.py
import os
import torch
import numpy as np
from PIL import Image

class SimpleImageLoader:
    def __init__(self):
        self.type = "IMAGE_LOADER"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "multiline": False,
                    "default": "C:/path/to/your/image.png"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image_path):
        try:
            # Sprawdzenie czy plik istnieje
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")
            
            # Wczytanie obrazu
            i = Image.open(image_path)
            i = i.convert('RGB')
            
            # Konwersja do formatu wymaganego przez ComfyUI
            image = np.array(i).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            return (image,)
            
        except Exception as e:
            print(f"Błąd podczas ładowania obrazu: {str(e)}")
            raise Exception(f"Nie udało się załadować obrazu: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "SimpleImageLoader": SimpleImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleImageLoader": "Simple Image Loader"
}