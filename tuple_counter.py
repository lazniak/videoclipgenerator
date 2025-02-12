# tuple_counter.py
class ComfyTupleCounter:
    """Counts number of elements in ComfyUI tuple-formatted lists."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("STRING", {
                    "forceInput": True,
                    "tooltip": """Accepts ComfyUI tuple/list output format.
Examples of valid inputs:
- Single item: ('item',)
- Multiple items: ('item1', 'item2', 'item3')
- Nested tuples: (('item1',), ('item2',))
Note: This node is specifically designed to count elements in ComfyUI's native tuple format output."""
                })
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("elements_count",)
    FUNCTION = "count_elements"
    CATEGORY = "utils/list operations"
    INPUT_IS_LIST = True

    def count_elements(self, input_list):
        """
        Counts elements in ComfyUI tuple format.
        Returns 0 if input is invalid or empty.
        """
        try:
            # Handle tuple input
            if isinstance(input_list, tuple):
                count = len(input_list)
            # Handle list input
            elif isinstance(input_list, list):
                count = len(input_list)
            # Handle single item
            else:
                count = 1
                
            return (count,)
            
        except Exception as e:
            print(f"Error in ComfyTupleCounter: {str(e)}")
            return (0,)

# Register node
NODE_CLASS_MAPPINGS = {
    "ComfyTupleCounter": ComfyTupleCounter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyTupleCounter": "ComfyUI Tuple Counter"
}