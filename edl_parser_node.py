import os
import json
import webbrowser
from typing import Tuple, Optional

COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

class EDLSceneParser:
    """
    Custom node for ComfyUI that parses EDL files and extracts frame information
    """
    
    # Kolory dla logów
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    COUNTER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_counter.json')
    
    @staticmethod
    def _load_counter_static():
        """Static method to load counter value"""
        try:
            counter_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_counter.json')
            if os.path.exists(counter_file):
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            print(f"Error loading counter: {e}")
        return 0
    
    _current_usage_count = _load_counter_static()
    
    def __init__(self):
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count

    def _load_counter(self):
        """Load usage counter with error handling"""
        try:
            if os.path.exists(self.COUNTER_FILE):
                with open(self.COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            print(f"Error loading counter: {e}")
        return 0

    def _save_counter(self):
        """Save usage counter with error handling"""
        try:
            with open(self.COUNTER_FILE, 'w') as f:
                json.dump({'usage_count': self.usage_count}, f)
        except Exception as e:
            print(f"Error saving counter: {e}")

    def _increment_counter(self):
        """Increment and save usage counter"""
        self.usage_count += 1
        self._save_counter()
        self.__class__._current_usage_count = self.usage_count
        print(f"{self.BLUE}EDLSceneParser: Usage count updated: {self.usage_count}{self.ENDC}")

    def _open_coffee_link(self):
        """Open the Buy Me a Coffee link"""
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            print(f"Error opening coffee link: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edl_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the EDL file to parse"
                }),
                "scene": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "step": 1,
                    "tooltip": "Scene number to process"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.1,
                    "step": 0.1,
                    "tooltip": "Frames per second rate"
                }),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support the development! ☕\nBy enabling this option, you'll be directed to a page where you can show your appreciation for this tool."
                }),
                "usage_counter": ("STRING", {
                    "default": f"Usage count: {cls._current_usage_count}",
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks how many times this node has been used."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "BOOLEAN", "INT")
    RETURN_NAMES = ("edl_path", "actual_frame", "next_frame", "scene_duration", "has_next", "total_scenes")
    FUNCTION = "process_edl"
    CATEGORY = "utils"

    def parse_edl_file(self, edl_path: str) -> list:
        """Parse EDL file and return list of video scenes"""
        if not os.path.exists(edl_path):
            raise FileNotFoundError(f"EDL file not found: {edl_path}")
            
        video_scenes = []
        
        with open(edl_path, 'r') as f:
            lines = f.readlines()
            
        for i in range(len(lines)):
            line = lines[i].strip()
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 4 and parts[2] == "V":
                    # Get the clip name from the next line
                    clip_name = ""
                    if i + 1 < len(lines) and "FROM CLIP NAME:" in lines[i + 1]:
                        clip_name = lines[i + 1].split("FROM CLIP NAME:")[1].strip()
                    
                    video_scenes.append({
                        'index': len(video_scenes) + 1,
                        'clip_name': clip_name,
                        'timecode': parts[6:8]  # Record OUT point timecode
                    })
                    
        return video_scenes

    def calculate_frame_duration(self, timecode_start: str, timecode_end: str, fps: float) -> int:
        """Calculate frame duration from timecode"""
        def timecode_to_frames(tc: str, fps: float) -> int:
            hh, mm, ss, ff = map(int, tc.split(':'))
            total_seconds = hh * 3600 + mm * 60 + ss
            return int(total_seconds * fps) + ff
            
        start_frames = timecode_to_frames(timecode_start, fps)
        end_frames = timecode_to_frames(timecode_end, fps)
        return end_frames - start_frames

    def process_edl(self, edl_path: str, scene: int, fps: float, 
                   buy_me_a_coffee: bool = False, usage_counter: str = "") -> Tuple[str, str, str, int, bool, int]:
        """Main processing function for the node"""
        try:
            self._increment_counter()
            if buy_me_a_coffee:
                self._open_coffee_link()

            print(f"{self.BLUE}EDLSceneParser: Processing EDL file: {edl_path}{self.ENDC}")
            video_scenes = self.parse_edl_file(edl_path)
            
            total_scenes = len(video_scenes)
            print(f"{self.BLUE}EDLSceneParser: Total video scenes found: {total_scenes}{self.ENDC}")
            
            if not video_scenes:
                print(f"{self.YELLOW}EDLSceneParser: No scenes found in EDL file{self.ENDC}")
                return (edl_path, "", "", 0, False, 0)
                
            # Adjust scene index (1-based) to list index (0-based)
            scene_idx = scene - 1
            
            # Jeśli żądana scena przekracza dostępne sceny, użyj ostatniej sceny
            if scene_idx >= len(video_scenes):
                print(f"{self.YELLOW}EDLSceneParser: Scene {scene} exceeds total scenes ({total_scenes}). Using last scene.{self.ENDC}")
                last_scene = video_scenes[-1]
                duration = self.calculate_frame_duration(
                    last_scene['timecode'][0],
                    last_scene['timecode'][1],
                    fps
                )
                # Zwracamy ostatnią scenę zarówno jako actual_frame jak i next_frame
                return (edl_path, last_scene['clip_name'], last_scene['clip_name'], duration, False, total_scenes)
                
            current_scene = video_scenes[scene_idx]
            actual_frame = current_scene['clip_name']
            
            # Calculate scene duration
            duration = self.calculate_frame_duration(
                current_scene['timecode'][0],
                current_scene['timecode'][1],
                fps
            )
            
            # Check if there's a next scene
            has_next = scene_idx + 1 < len(video_scenes)
            # Jeśli nie ma następnej sceny, użyj ostatniej jako next_frame
            next_frame = video_scenes[scene_idx + 1]['clip_name'] if has_next else video_scenes[-1]['clip_name']
            
            print(f"{self.GREEN}EDLSceneParser: Successfully processed scene {scene} of {total_scenes}{self.ENDC}")
            return (edl_path, actual_frame, next_frame, duration, has_next, total_scenes)
            
        except Exception as e:
            print(f"{self.RED}EDLSceneParser: Error processing EDL file: {str(e)}{self.ENDC}")
            return (edl_path, "", "", 0, False, 0)

# Node registration
NODE_CLASS_MAPPINGS = {
    "EDLSceneParser": EDLSceneParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EDLSceneParser": "EDL Scene Parser"
}