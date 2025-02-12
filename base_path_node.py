# base_path_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import folder_paths

class ComfyUIBasePath:
    """Node returning the base ComfyUI installation path"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}  # Node nie wymaga żadnych inputów
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_path",)
    FUNCTION = "get_base_path"
    CATEGORY = "utils"
    
    def get_base_path(self):
        try:
            # Pobierz ścieżkę bazową ComfyUI
            base_path = os.path.dirname(folder_paths.__file__)
            # Konwertuj separatory na forward slash
            base_path = base_path.replace('\\', '/')
            return (base_path,)
        except Exception as e:
            print(f"Error getting base path: {str(e)}")
            return ("",)

NODE_CLASS_MAPPINGS = {
    "ComfyUIBasePath": ComfyUIBasePath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIBasePath": "ComfyUI Base Path"
}