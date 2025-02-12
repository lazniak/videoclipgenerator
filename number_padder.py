# number_padder.py
class NumberPadder:
    def __init__(self):
        self.type = "NUMBER_PADDER"
        self.output_node = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("INT", {
                    "default": 42,
                    "min": 0,  # Minimalna wartość
                    "max": 999999999  # Maksymalna wartość - można dostosować
                }),
                "padding_length": ("INT", {
                    "default": 4,
                    "min": 1,  # Minimalna długość paddingu
                    "max": 20  # Maksymalna długość paddingu
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pad_number"
    CATEGORY = "formatting"

    def pad_number(self, number, padding_length):
        try:
            # Konwertuje liczbę na string i dodaje padding
            padded_number = str(number).zfill(padding_length)
            return (padded_number,)
            
        except Exception as e:
            print(f"Błąd podczas formatowania liczby: {str(e)}")
            return (str(number),)  # W razie błędu zwraca oryginalną liczbę jako string

NODE_CLASS_MAPPINGS = {
    "NumberPadder": NumberPadder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NumberPadder": "Number Padder"
}