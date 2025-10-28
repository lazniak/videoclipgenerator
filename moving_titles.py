import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pysrt
import requests
import json
import webbrowser
import logging
from time import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Optional, Tuple, Dict, Union, List
from enum import Enum
import colorsys
from functools import lru_cache

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MovingTitles')

COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

class BlendMode(str, Enum):
    NORMAL = "normal"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    COLOR_BURN = "color_burn"
    COLOR_DODGE = "color_dodge"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"

class TextEffect(str, Enum):
    NONE = "none"
    SHADOW = "shadow"
    OUTLINE = "outline"
    GLOW = "glow"
    GRADIENT = "gradient"
    NEON = "neon"
    METALLIC = "metallic"

class TextAnimation(str, Enum):
    NONE = "none"
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    TYPEWRITER = "typewriter"
    BOUNCE = "bounce"
    WAVE = "wave"
    ROTATE = "rotate"
    SCALE = "scale"

class TextAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class FontManager:
    """Manager dla fontów z auto-pobieraniem i obsługą custom fontów"""
    FONT_URLS = {
        'Roboto': 'https://fonts.gstatic.com/s/roboto/v29/KFOmCnqEu92Fr1Me5Q.ttf',
        'OpenSans': 'https://fonts.gstatic.com/s/opensans/v34/mem8YaGs126MiZpBA-U1Ug.ttf',
        'Lato': 'https://fonts.gstatic.com/s/lato/v23/S6uyw4BMUTPHvxk.ttf',
        'Montserrat': 'https://fonts.gstatic.com/s/montserrat/v25/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCs16Hw5aX8.ttf',
        'PlayfairDisplay': 'https://fonts.gstatic.com/s/playfairdisplay/v30/nuFvD-vYSZviVYUb_rj3ij__anPXJzDwcbmjWBN2PKdFvXDXbtY.ttf',
        'SourceCodePro': 'https://fonts.gstatic.com/s/sourcecodepro/v22/HI_SiYsKILxRpg3hIP6sJ7fM7PqlPevT.ttf',
        'Oswald': 'https://fonts.gstatic.com/s/oswald/v49/TK3_WkUHHAIjg75cFRf3bXL8LICs169vsUZiYA.ttf',
        'RobotoMono': 'https://fonts.gstatic.com/s/robotomono/v22/L0xuDF4xlVMF-BfR8bXMIhJHg45mwgGEFl0_3vq_ROW4.ttf'
    }

    def __init__(self):
        self.fonts_cache_dir = os.path.join(os.path.dirname(__file__), "fonts_cache")
        self.fonts = {}
        self.ensure_cache_dir()

    def ensure_cache_dir(self):
        """Upewnia się, że katalog cache istnieje"""
        if not os.path.exists(self.fonts_cache_dir):
            os.makedirs(self.fonts_cache_dir)

    def get_font(self, font_name: str, size: int = 32, custom_font_path: str = None) -> ImageFont.FreeTypeFont:
        """Pobiera font z cache, pobiera z internetu lub używa custom font"""
        # If custom font path is provided and exists, use it
        if custom_font_path and os.path.exists(custom_font_path):
            try:
                return ImageFont.truetype(custom_font_path, size)
            except Exception as e:
                logger.warning(f"Failed to load custom font from {custom_font_path}: {e}")
                logger.info("Falling back to default font")
        
        # Use built-in font
        if font_name not in self.FONT_URLS:
            raise ValueError(f"Unknown font: {font_name}")

        cache_path = os.path.join(self.fonts_cache_dir, f"{font_name}.ttf")
        
        # Pobierz font jeśli nie jest w cache
        if not os.path.exists(cache_path):
            try:
                response = requests.get(self.FONT_URLS[font_name])
                response.raise_for_status()
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                logger.error(f"Error downloading font {font_name}: {e}")
                raise

        return ImageFont.truetype(cache_path, size)

class MovingTitlesNode:
    """Advanced text animation node for ComfyUI"""
    
    # Colors for logs
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    # Inicjalizacja licznika
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
            logger.error(f"Error loading counter: {e}")
        return 0
    
    _current_usage_count = _load_counter_static()

    def __init__(self):
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count
        self.fonts_manager = FontManager()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "srt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to SRT subtitle file"
                }),
                "font_name": (list(FontManager.FONT_URLS.keys()),),
                "font_size": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Font size in pixels"
                }),
                "max_lines": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of lines to display at once. Lines will appear sequentially."
                }),
                "line_spacing": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Spacing between lines (multiplier of font size)"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
                "text_effect": (list(TextEffect.__members__.keys()),),
                "animation_type": (list(TextAnimation.__members__.keys()),),
                "blend_mode": (list(BlendMode.__members__.keys()),),
                "text_alignment": (list(TextAlignment.__members__.keys()),),
                "position_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Horizontal position (0-1)"
                }),
                "position_y": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Vertical position (0-1)"
                }),
            },
            "optional": {
                "custom_font_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to custom .ttf or .otf font file. Leave empty to use built-in fonts."
                }),
                "audio_signal": ("AUDIO",),
                "audio_sensitivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "text_color_hex": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Text color in HEX format"
                }),
                "shadow_color_hex": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Shadow color in HEX format"
                }),
                "gradient_start_hex": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Gradient start color in HEX format"
                }),
                "gradient_end_hex": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Gradient end color in HEX format"
                }),
                "outline_width": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Width of text outline"
                }),
                "glow_radius": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Radius of glow effect"
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Intensity of glow effect"
                }),
                "animation_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Speed of animation effects"
                }),
                "wave_amplitude": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Amplitude of wave animation"
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Frequency of wave animation"
                }),
                "mask_image": ("IMAGE",),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support development with a coffee! ☕"
                }),
                "usage_counter": ("STRING", {
                    "default": f"Usage count: {cls._current_usage_count}",
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks how many times this node has been used"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_titles"
    CATEGORY = "video/text"

    def _load_counter(self) -> int:
        """Load usage counter"""
        try:
            if os.path.exists(self.COUNTER_FILE):
                with open(self.COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            logger.error(f"Error loading counter: {e}")
        return 0

    def _save_counter(self):
        """Save usage counter"""
        try:
            with open(self.COUNTER_FILE, 'w') as f:
                json.dump({'usage_count': self.usage_count}, f)
        except Exception as e:
            logger.error(f"Error saving counter: {e}")

    def _increment_counter(self):
        """Increment and save counter"""
        self.usage_count += 1
        self._save_counter()
        self.__class__._current_usage_count = self.usage_count
        logger.info(f"{self.BLUE}Usage count updated: {self.usage_count}{self.ENDC}")

    def _open_coffee_link(self):
        """Open buy me a coffee link"""
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            logger.error(f"Error opening coffee link: {e}")

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert HEX color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_gradient_color(self, start_hex: str, end_hex: str, progress: float) -> Tuple[int, int, int]:
        """Create gradient color based on progress"""
        start_rgb = self.hex_to_rgb(start_hex)
        end_rgb = self.hex_to_rgb(end_hex)
        
        return tuple(int(start + (end - start) * progress) 
                    for start, end in zip(start_rgb, end_rgb))

    def apply_text_effect(self, image: Image.Image, text: str, position: Tuple[int, int],
                         font: ImageFont.FreeTypeFont, effect: TextEffect, params: dict) -> Image.Image:
        """Apply text effect to the image"""
        draw = ImageDraw.Draw(image)
        
        if effect == TextEffect.NONE:
            draw.text(position, text, font=font, fill=self.hex_to_rgb(params['text_color_hex']))
            
        elif effect == TextEffect.SHADOW:
            shadow_offset = params.get('shadow_offset', 3)
            # Draw shadow
            shadow_pos = (position[0] + shadow_offset, position[1] + shadow_offset)
            draw.text(shadow_pos, text, font=font, fill=self.hex_to_rgb(params['shadow_color_hex']))
            # Draw text
            draw.text(position, text, font=font, fill=self.hex_to_rgb(params['text_color_hex']))
            
        elif effect == TextEffect.OUTLINE:
            outline_width = params.get('outline_width', 2)
            # Draw outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((position[0] + dx, position[1] + dy), text, font=font,
                            fill=self.hex_to_rgb(params['shadow_color_hex']))
            # Draw main text
            draw.text(position, text, font=font, fill=self.hex_to_rgb(params['text_color_hex']))
            
        elif effect == TextEffect.GLOW:
            glow_radius = params.get('glow_radius', 10)
            glow_intensity = params.get('glow_intensity', 0.5)
            
            # Create glow layer
            glow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            glow_draw.text(position, text, font=font, fill=self.hex_to_rgb(params['shadow_color_hex']))
            
            # Apply blur to glow
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
            
            # Blend glow with original
            image = Image.blend(image, glow_layer, glow_intensity)
            
            # Draw main text
            draw.text(position, text, font=font, fill=self.hex_to_rgb(params['text_color_hex']))
            
        elif effect == TextEffect.GRADIENT:
            # Create gradient mask
            gradient_mask = Image.new('L', (font.getsize(text)[0], font.getsize(text)[1]))
            gradient_draw = ImageDraw.Draw(gradient_mask)
            
            for i in range(gradient_mask.height):
                progress = i / gradient_mask.height
                color = self.create_gradient_color(
                    params['gradient_start_hex'],
                    params['gradient_end_hex'],
                    progress
                )
                gradient_draw.line([(0, i), (gradient_mask.width, i)], fill=color)
            
            # Draw text with gradient
            image.paste(gradient_mask, position, gradient_mask)
            
        return image

    def apply_animation(self, position: Tuple[float, float], animation: TextAnimation,
                       progress: float, params: dict) -> Tuple[float, float]:
        """Apply animation to text position"""
        x, y = position
        canvas_width = params.get('canvas_width', 1920)
        canvas_height = params.get('canvas_height', 1080)
        
        if animation == TextAnimation.NONE:
            return x, y
            
        elif animation == TextAnimation.SLIDE_LEFT:
            x = (1 - progress) * canvas_width + x * progress
            
        elif animation == TextAnimation.SLIDE_RIGHT:
            x = -progress * canvas_width + x * (1 - progress)
            
        elif animation == TextAnimation.SLIDE_UP:
            y = (1 - progress) * canvas_height + y * progress
            
        elif animation == TextAnimation.SLIDE_DOWN:
            y = -progress * canvas_height + y * (1 - progress)
            
        elif animation == TextAnimation.BOUNCE:
            bounce_height = params.get('wave_amplitude', 10)
            y += bounce_height * abs(np.sin(progress * np.pi * 2))
            
        elif animation == TextAnimation.WAVE:
            wave_amplitude = params.get('wave_amplitude', 10)
            wave_frequency = params.get('wave_frequency', 1.0)
            y += wave_amplitude * np.sin(progress * wave_frequency * np.pi * 2)
            
        elif animation == TextAnimation.ROTATE:
            center_x, center_y = x + params.get('text_width', 0) / 2, y + params.get('text_height', 0) / 2
            angle = progress * 360
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            x = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
            y = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a
            
        return x, y

    def process_frame(self, frame_data: dict) -> torch.Tensor:
        """Process single frame with subtitles"""
        frame = frame_data['frame']
        params = frame_data['params']
        current_time = frame_data['current_time']
        
        # Convert frame to PIL Image
        frame_pil = Image.fromarray((frame.numpy() * 255).astype(np.uint8))
        
        # Create text layer
        text_layer = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        
        # Get active subtitles
        subs = pysrt.open(params['srt_path'])
        active_subs = [sub for sub in subs 
                      if sub.start.ordinal/1000.0 <= current_time <= sub.end.ordinal/1000.0]
        
        # Apply max_lines limit - take only the first max_lines subtitles
        max_lines = params.get('max_lines', 1)
        active_subs = active_subs[:max_lines]
        
        for idx, sub in enumerate(active_subs):
            # Calculate progress for animation
            progress = (current_time - sub.start.ordinal/1000.0) / \
                      (sub.end.ordinal/1000.0 - sub.start.ordinal/1000.0)
            
            # Get font
            font = self.fonts_manager.get_font(
                params['font_name'],
                size=params.get('font_size', 32),
                custom_font_path=params.get('custom_font_path', None)
            )
            
            # Calculate line spacing
            line_spacing = params.get('line_spacing', 1.2)
            font_size = params.get('font_size', 32)
            line_offset = idx * int(font_size * line_spacing)
            
            # Calculate position with multi-line offset
            base_x = int(params['position_x'] * frame_pil.width)
            base_y = int(params['position_y'] * frame_pil.height) + line_offset
            
            # Apply animation
            x, y = self.apply_animation(
                (base_x, base_y),
                TextAnimation[params['animation_type']],
                progress,
                {**params, 'text_width': font.getsize(sub.text)[0],
                 'text_height': font.getsize(sub.text)[1]}
            )
            
            # Apply text effect
            text_layer = self.apply_text_effect(
                text_layer,
                sub.text,
                (int(x), int(y)),
                font,
                TextEffect[params['text_effect']],
                params
            )
        
        # Apply blend mode
        result = Image.alpha_composite(frame_pil.convert('RGBA'), text_layer)
        
        # Convert back to tensor
        result_np = np.array(result) / 255.0
        return torch.from_numpy(result_np).float()

    def apply_titles(self, image: torch.Tensor, **params) -> Tuple[torch.Tensor]:
        """Main function for applying titles to video"""
        try:
            # Increment counter and handle coffee link
            self._increment_counter()
            if params.get('buy_me_a_coffee', False):
                self._open_coffee_link()
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Starting title animation{self.ENDC}")
            logger.info(f"{self.YELLOW}{'=' * 50}{self.ENDC}")
            
            start_time = time()
            
            # Prepare frames for processing
            batch_size = image.shape[0]
            frame_time = 1.0 / params['fps']
            
            frame_data_list = [
                {
                    'frame': image[i],
                    'params': params,
                    'current_time': i * frame_time,
                    'frame_number': i
                }
                for i in range(batch_size)
            ]
            
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_frames = list(executor.map(self.process_frame, frame_data_list))
            
            # Stack results
            result = torch.stack(processed_frames)
            
            total_time = time() - start_time
            fps = batch_size / total_time
            
            logger.info(f"\n{self.GREEN}Processing complete!")
            logger.info(f"Processed {batch_size} frames in {total_time:.2f}s ({fps:.1f} fps){self.ENDC}")
            
            return (result,)
            
        except Exception as e:
            logger.error(f"{self.RED}Error in apply_titles: {str(e)}{self.ENDC}")
            raise
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "MovingTitles": MovingTitlesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MovingTitles": "Moving Titles"
}