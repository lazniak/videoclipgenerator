import os
import glob
from pathlib import Path

class FilePathProcessor:
    """ComfyUI custom node for processing file paths and returning full system paths"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "process_paths"
    CATEGORY = "utils"
    
    def process_paths(self, input_path):
        try:
            # Konwertuj ścieżkę na absolutną
            abs_path = os.path.abspath(input_path)
            
            # Sprawdź czy ścieżka zawiera wzorzec (wildcard)
            if any(char in input_path for char in ['*', '?']):
                # Użyj glob do znalezienia wszystkich pasujących plików
                matching_files = glob.glob(input_path, recursive=True)
                # Konwertuj wszystkie ścieżki na absolutne
                full_paths = [os.path.abspath(f) for f in matching_files]
            else:
                # Jeśli to pojedyncza ścieżka, sprawdź czy istnieje
                if os.path.exists(abs_path):
                    if os.path.isdir(abs_path):
                        # Jeśli to katalog, zwróć wszystkie pliki w nim
                        full_paths = []
                        for root, _, files in os.walk(abs_path):
                            for file in files:
                                full_paths.append(os.path.abspath(os.path.join(root, file)))
                    else:
                        # Jeśli to plik, zwróć jego ścieżkę
                        full_paths = [abs_path]
                else:
                    full_paths = []
            
            # Połącz wszystkie ścieżki w jeden string z nową linią jako separator
            result = "\n".join(full_paths)
            return (result,)
            
        except Exception as e:
            print(f"Error processing paths: {str(e)}")
            return ("",)  # Zwróć pusty string w przypadku błędu

NODE_CLASS_MAPPINGS = {
    "FilePathProcessor": FilePathProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePathProcessor": "File Path Processor"
}