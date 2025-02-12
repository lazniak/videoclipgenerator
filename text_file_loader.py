# text_file_loader.py
import os
import folder_paths

class TextFileLoader:
    def __init__(self):
        self.type = "TEXT_FILE_LOADER"
        self.output_node = True
        self.input_folder_dir = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "examples"
                }),
                "file_name": ("STRING", {
                    "multiline": False,
                    "default": "example.txt"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_text_file"
    CATEGORY = "text"

    def load_text_file(self, folder_path, file_name):
        try:
            # Zabezpieczenie przed wyjściem poza dozwolony katalog
            full_path = os.path.join(folder_path, file_name)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Plik nie istnieje: {full_path}")
                
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return (content,)
            
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku: {str(e)}")
            return ("",)

NODE_CLASS_MAPPINGS = {
    "TextFileLoader": TextFileLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFileLoader": "Text File Loader"
}