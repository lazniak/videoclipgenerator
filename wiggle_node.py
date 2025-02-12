import torch
import numpy as np
from PIL import Image, ImageOps
import math
import logging
from time import time as get_time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import psutil
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import json
import webbrowser
from functools import lru_cache

COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WiggleEffect')

class WiggleEffectNode:
    """Enhanced After Effects-style wiggle effect node for ComfyUI"""
    
    # Kolory dla logów
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    # Inicjalizacja zmiennych klasowych
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
    
    # Inicjalizacja licznika przy starcie klasy używając metody statycznej
    _current_usage_count = _load_counter_static()
    
    # Cache dla często używanych operacji
    _noise_cache = {}
    _transform_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Basic Settings
                "image": ("IMAGE",),
                "preserve_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, output will maintain the same size as input image. When disabled, custom width and height settings will be used. Keeping original size (enabled) is recommended for best quality and performance"
                }),
                "frame_width": ("INT", {
                    "default": 1280,
                    "min": 8,
                    "max": 8192,
                    "tooltip": "The width of the output frames in pixels. Higher values mean larger image width but require more processing power. Common values: 1280 (HD), 1920 (Full HD), 3840 (4K)"
                }),
                "frame_height": ("INT", {
                    "default": 720,
                    "min": 8,
                    "max": 8192,
                    "tooltip": "The height of the output frames in pixels. Higher values mean larger image height but require more processing power. Common values: 720 (HD), 1080 (Full HD), 2160 (4K)"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.25,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Frames Per Second - determines how many frames are generated per second of animation. Common values: 24 (film), 25 (PAL), 30 (NTSC), 60 (smooth motion). Higher FPS means smoother animation but more frames to process"
                }),

                # Time Settings
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Controls how fast the wiggle animation plays. Think of it like a speed multiplier: 1.0 is normal speed, 2.0 makes everything move twice as fast, 0.5 makes everything move at half speed. This affects all aspects of motion - position, rotation, and scaling. Useful for fine-tuning the feel of your animation without changing other parameters"
                }),
                "time_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Shifts the starting point of the animation in time (measured in seconds). This is like choosing where in the wiggle pattern you want to begin. Useful for: 1) Creating seamless loops by finding the right offset, 2) Synchronizing multiple wiggle effects by offsetting their start times, 3) Finding the perfect part of the motion pattern for your animation"
                }),

                # Position Settings
                "enable_position": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turns position animation on or off. When enabled, the image will move around its center point based on the position settings below. When disabled, the image stays centered with only scale and rotation effects (if enabled)"
                }),
                "position_mode": (["independent", "linked"], {
                    "default": "independent",
                    "tooltip": "Controls how the horizontal (X) and vertical (Y) movements relate to each other. In 'independent' mode, X and Y move separately, creating more random-looking motion (like a leaf in the wind). In 'linked' mode, X and Y are connected to create circular or elliptical paths (like a planet's orbit). Independent is great for organic motion, while linked is better for mechanical or planned movements"
                }),
                "position_frequency": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How rapidly the position changes occur. Think of it like adjusting the 'nervousness' of the movement. Low values (0.1-0.5) create slow, smooth drifting motions perfect for subtle effects like floating or swaying. Higher values (1.0+) create rapid, jittery movements good for bouncing or vibrating effects. The magic usually happens between 0.1 and 1.0 for natural-looking motion"
                }),
                "position_magnitude": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Controls how far the image moves from its center point, measured in pixels. Think of it as the 'intensity' of movement. At 15 pixels, you get subtle movement good for gentle floating effects. At 50-100 pixels, you get more dramatic swaying or bouncing. Higher values create wild, sweeping motions. The actual movement will be up to this amount in any direction. Remember: this interacts with motion blur - bigger movements create more blur!"
                }),
                "position_temporal_phase": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Shifts the timing of position changes, measured in degrees (0-360). This is like choosing which point in the wiggle cycle to start from. Very useful when you have multiple elements - giving each a different phase (like 0, 120, and 240) creates a pleasing cascade effect. Also great for fine-tuning when exactly movement peaks occur in your animation. Think of it as the 'timing offset' of the movement pattern"
                }),
                "position_spatial_phase": ("FLOAT", {
                    "default": 90.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Controls the spatial pattern of movement, measured in degrees (0-360). While temporal phase shifts timing, spatial phase shifts the actual movement pattern. A 90-degree difference between X and Y creates circular motions (when linked). Different values create different elliptical or figure-eight patterns. Great for creating complex, natural-looking movements or precise mechanical patterns"
                }),

                # Scale Settings
                "enable_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turns scaling animation on or off. When enabled, the image will grow and shrink based on the scale settings below. This can create breathing, pulsing, or throbbing effects. Can be combined with position and rotation for more complex animations. When disabled, the image maintains its original size while other effects continue to work"
                }),
                "scale_mode": (["linked", "independent", "none"], {
                    "default": "linked",
                    "tooltip": "Determines how horizontal and vertical scaling relate to each other. 'Linked' mode scales both dimensions together, preserving aspect ratio - great for pulsing or breathing effects. 'Independent' mode allows separate X/Y scaling for squash-and-stretch effects. 'None' is the same as disabling scale completely. Use 'linked' for natural growth/shrink, 'independent' for more dynamic or cartoony effects"
                }),
                "scale_frequency": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How rapidly scaling changes occur. Lower values (0.1-0.3) create slow, breathing-like effects. Higher values create rapid pulsing or vibrating. Very high values might look jittery unless combined with motion blur. The default 0.15 is good for subtle, organic scaling. Consider your animation's mood - slow for calm, faster for energy or tension"
                }),
                "scale_magnitude": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "The maximum amount of scaling, in percentage points. A value of 2.0 means the image will scale between 98% and 102% of its original size. Small values (1-5) create subtle breathing effects. Larger values (10-20) create more dramatic pulsing. Very large values can create explosive growth and shrinking. Remember: this is a percentage, so 100 means scaling from 0% to 200% - use with caution!"
                }),
                "scale_temporal_phase": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Controls when scaling changes occur in the animation cycle (0-360 degrees). Like position_temporal_phase, but for scaling. Use different values for position and scale phase to create complex animations where movement and scaling aren't synchronized. For example, setting this 90 degrees apart from position creates a flow where maximum size occurs a quarter-cycle after maximum movement"
                }),
                "scale_spatial_phase": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Affects the relationship between horizontal and vertical scaling patterns when in 'independent' mode. Different values create different patterns of stretching and squashing. 90 degrees creates alternating stretch/squash. 180 degrees makes them oppose each other. Experiment with this when creating squash-and-stretch effects for character animation or bouncing effects"
                }),

                # Rotation Settings
                "enable_rotation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Turns rotational animation on or off. When enabled, the image will rock back and forth or spin based on rotation settings below. Rotation happens around the image center. Combine with position and scale for effects like floating debris, drifting leaves, or wobbling characters. When disabled, the image maintains its original orientation"
                }),
                "rotation_frequency": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Controls how rapidly rotation changes occur. Low values (0.1-0.5) create gentle rocking or swaying. Higher values create faster spinning or shaking. The default 0.1 is good for subtle movement like paper floating or leaves swaying. Higher values (1.0+) work better for mechanical movement or intense effects. Consider using motion blur with faster rotations to smooth the movement"
                }),
                "rotation_magnitude": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "The maximum rotation angle in degrees. At 2.0, the image rocks ±2 degrees - subtle but noticeable. Values around 5-15 degrees create obvious swaying. 30-90 degrees create dramatic rocking or partial spins. 360 allows complete rotation. Small values (1-5) work well for organic movement, larger values for more dramatic effects. Remember that faster rotation at larger angles needs more motion blur to look smooth"
                }),
                "rotation_temporal_phase": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Sets when rotation peaks occur in the animation cycle (0-360 degrees). Like other temporal phases, this helps coordinate rotation with other movements. Try setting this 120 degrees apart from position and scale for complex, organic movement where each aspect peaks at different times. Great for creating flowing, natural-looking animation or controlling precise mechanical timing"
                }),
                "rotation_spatial_phase": ("FLOAT", {
                    "default": 60.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Influences the rotational movement pattern. While less intuitive than other spatial phases, this can create interesting variations in how rotation combines with other movements. Different values create different relationships between rotation and position/scale changes. Experiment with this when fine-tuning complex animations"
                }),
                # Motion Options
                "time_remap_curve": (["linear", "ease", "ease-in", "ease-out", "smooth"], {
                    "default": "smooth",
                    "tooltip": "Controls how movement transitions happen. 'Linear' moves at constant speed - mechanical but predictable. 'Ease' starts slow, speeds up in the middle, then slows down - good for natural movement. 'Ease-in' starts slow and speeds up - like falling. 'Ease-out' starts fast and slows down - like throwing upward. 'Smooth' is extra gentle with strongest smoothing - best for flowing, peaceful movements. Each option drastically changes the feel of your animation"
                }),
                "motion_blur": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adds motion blur based on movement speed and direction - just like a camera capturing fast motion. At 0, movements are crisp but can look choppy. Values around 0.3-0.5 give natural-looking blur. Higher values create more dramatic streaking effects. The blur automatically adapts to movement speed - faster movement creates stronger blur. Great for smoothing fast movements or creating artistic streaking effects"
                }),
                "motion_tile": (["disabled", "mirror", "repeat"], {
                    "default": "mirror",
                    "tooltip": "Controls what happens at the edges of your frame during movement. 'Disabled' shows empty space/transparency when movement reveals edges. 'Mirror' creates reflected copies at edges - great for seamless motion, prevents empty gaps. 'Repeat' tiles the image at edges like a pattern. 'Mirror' is usually best for natural footage, while 'repeat' can work well for patterns or textures. Important when your movement range is large enough to show edges"
                }),
                "motion_tile_auto_margin": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, automatically calculates how much extra space is needed around edges based on your movement settings. This ensures no empty spaces appear during motion. When disabled, you can manually set the margin size. Auto mode is usually best unless you need precise control over edge handling. The auto calculation considers position, scale, and rotation ranges to determine optimal margins"
                }),
                "motion_tile_margin": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Sets the size of edge margins in pixels when auto margin is disabled. This is how much extra space to add around your image for movement. Larger values prevent edge gaps but use more memory. As a rule of thumb, should be at least as large as your maximum movement range (position_magnitude). Too small margins may show edges, too large wastes processing power. Only used when motion_tile_auto_margin is false"
                }),

                # Noise Options
                "noise_type": (["perlin", "simplex", "value", "cellular"], {
                    "default": "perlin",
                    "tooltip": "Selects the algorithm that generates the wiggle pattern. 'Perlin' is smooth and natural, great for organic movement like leaves or floating. 'Simplex' is similar but more efficient, good for complex animations. 'Value' creates sharper, more defined changes - good for mechanical or glitch effects. 'Cellular' produces organic, tissue-like patterns great for biological or abstract effects. Perlin is the safest choice for most animations, experiment with others for unique effects"
                }),
                "noise_octaves": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Controls how many layers of detail are in the wiggle pattern. Each octave adds a finer layer of movement. 1 octave = simple, smooth movement. 4-6 octaves = natural, complex movement with several scales of detail. 8 octaves = very detailed movement but slower to calculate. More octaves create more natural, fractal-like motion but require more processing power. Think of it like adding harmonics to a musical note"
                }),
                "noise_persistence": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Determines how much each finer detail layer contributes to the movement. Low values (0.2-0.4) create movement dominated by broad, sweeping motions with subtle detail. Higher values (0.6-0.8) make finer details more noticeable, creating more nervous or jittery motion. At 1.0, all detail layers have equal strength. This is like adjusting the mix between broad and fine movement patterns"
                }),
                "noise_lacunarity": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.01,
                    "tooltip": "Controls how much finer each detail layer becomes. At 1.0, each layer has similar scale. Higher values (2.0+) make each layer much finer than the last, creating more distinction between broad and fine movements. Lower values keep the detail scales more similar. Typically, values between 1.5 and 2.0 give the most natural results. Think of it as adjusting the contrast between different scales of movement"
                }),

                # Output Options
                "interpolation": (["bicubic", "bilinear", "nearest"], {
                    "default": "bicubic",
                    "tooltip": "Determines how pixels are calculated when the image transforms. 'Bicubic' gives the smoothest, highest quality results but is slowest. 'Bilinear' is a good balance of quality and speed. 'Nearest' gives sharp, pixelated edges - usually only used for pixel art or special effects. For most footage, stick with bicubic. Use bilinear if you need more speed, or nearest for deliberately pixelated looks"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Sets the random seed for generating the wiggle pattern. The same seed always creates the same movement pattern (when other settings are unchanged). This lets you precisely recreate animations or synchronize multiple elements. Change the seed to get completely different movement patterns while keeping all other settings the same. Useful for trying different movements or creating variety among multiple animated elements"
                }),

                # Additional Options
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "A little checkbox that means a lot! Enable this to support the development of this node and future improvements. Every coffee helps fuel new features and optimizations. Think of it as buying the developer a coffee to say thanks! ☕"
                }),
                
                "usage_counter": ("STRING", {
                "default": f"Usage count: {cls._current_usage_count}",  # Używamy zmiennej klasowej
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks how many times this node has been used across all your projects"
                })
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to control where the wiggle effect appears. White areas in the mask get full effect, black areas stay still, gray areas get partial effect. Perfect for making only parts of your image wiggle while keeping other parts stable. Great for combining with animated masks for complex effects"
                }),
                "reference_frame": ("IMAGE", {
                    "tooltip": "Optional reference frame for advanced motion calculations. This is a placeholder for future features that will enable more complex motion analysis and matching. Currently not actively used but included for future compatibility"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "apply_wiggle"
    CATEGORY = "animation/transform"

    def __init__(self):
        """Initialize with optimized resource management"""
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._noise_cache = {}
        self._transform_cache = {}
        
        # Inicjalizacja licznika
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count  # Aktualizujemy wartość klasową
        
        # Sprawdzenie dostępnych zasobów
        self.memory_info = self._check_system_resources()
        
    def _check_system_resources(self) -> Dict[str, int]:
        """Check and log available system resources"""
        memory_info = {"system": 0, "gpu": 0}
        
        # System memory
        memory = psutil.virtual_memory()
        memory_info["system"] = memory.available
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            memory_info["gpu"] = gpu_memory
            logger.info(f"Available GPU memory: {gpu_memory / 1024**2:.0f}MB")
            
        logger.info(f"Available system memory: {memory_info['system'] / 1024**2:.0f}MB")
        return memory_info

    def _load_counter(self) -> int:
        """Load usage counter with error handling"""
        try:
            if os.path.exists(self.COUNTER_FILE):
                with open(self.COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            logger.error(f"Error loading counter: {e}")
        return 0

    def _save_counter(self):
        """Save usage counter with error handling"""
        try:
            with open(self.COUNTER_FILE, 'w') as f:
                json.dump({'usage_count': self.usage_count}, f)
        except Exception as e:
            logger.error(f"Error saving counter: {e}")

    def _increment_counter(self):
        """Increment and save usage counter"""
        self.usage_count += 1
        self._save_counter()
        self.__class__._current_usage_count = self.usage_count  # Aktualizujemy wartość klasową
        logger.info(f"{self.BLUE}Usage count updated: {self.usage_count}{self.ENDC}")

    def _open_coffee_link(self):
        """Open Buy Me a Coffee link with error handling"""
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            logger.error(f"Error opening coffee link: {e}")

    @lru_cache(maxsize=128)
    def convert_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Optimized tensor to PIL conversion"""
        # Ensure tensor is on CPU and in correct format
        if tensor.is_cuda:
            tensor = tensor.cpu()
            
        # Convert to numpy with proper scaling
        np_image = (tensor.numpy() * 255.0).astype(np.uint8)
        
        # Determine mode based on channels
        if np_image.shape[-1] == 1:
            return Image.fromarray(np_image.squeeze(-1), mode='L')
        elif np_image.shape[-1] == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[-1] == 4:
            return Image.fromarray(np_image, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {np_image.shape[-1]}")
        
    def convert_pil_to_tensor(self, image: Image.Image, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Optimized PIL to tensor conversion"""
        # Convert to numpy array with proper normalization
        np_image = np.array(image).astype(np.float32) / 255.0
        
        # Handle single channel images
        if len(np_image.shape) == 2:
            np_image = np_image[..., None]
        
        # Convert to tensor
        tensor = torch.from_numpy(np_image)
        
        # Match channels if needed
        if tensor.shape[-1] != original_shape[-1]:
            if tensor.shape[-1] == 1 and original_shape[-1] == 3:
                tensor = tensor.repeat(1, 1, 3)
        
        return tensor

    def _lerp(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """High precision linear interpolation"""
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        t = t.astype(np.float64)
        
        result = a + (b - a) * t
        epsilon = np.finfo(np.float64).eps
        result = np.clip(result + epsilon, -1.0, 1.0)
        
        return result

    def _smooth_step(self, x: np.ndarray) -> np.ndarray:
        """Enhanced smooth step function"""
        x = np.clip(x, 0.0, 1.0)
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    def _ease_in(self, x: np.ndarray) -> np.ndarray:
        """Cubic ease-in function"""
        return x * x * x

    def _ease_out(self, x: np.ndarray) -> np.ndarray:
        """Cubic ease-out function"""
        return 1.0 - np.power(1.0 - x, 3)

    def _ease_both(self, x: np.ndarray) -> np.ndarray:
        """Combined ease-in-out function"""
        return np.where(x < 0.5,
            4 * x * x * x,
            1 - np.power(-2 * x + 2, 3) / 2
        )

    def _perlin_noise(self, t: np.ndarray, seed: int) -> np.ndarray:
        """Generate high precision Perlin noise"""
        seed = seed & 0xFFFFFFFF
        np.random.seed(seed)
        
        PERM_SIZE = 256
        perm = np.random.permutation(PERM_SIZE)
        grad = np.random.uniform(-1, 1, PERM_SIZE)
        
        ti = np.floor(t).astype(int) & (PERM_SIZE - 1)
        tf = t - np.floor(t)
        
        u = self._smooth_step(tf)
        
        g0 = grad[perm[ti]]
        g1 = grad[perm[(ti + 1) & (PERM_SIZE - 1)]]
        
        return self._lerp(g0, g1, u)

    def _simplex_noise(self, t: np.ndarray, seed: int) -> np.ndarray:
        """Generate high precision Simplex noise"""
        seed = seed & 0xFFFFFFFF
        np.random.seed(seed)
        
        PERM_SIZE = 512
        perm = np.random.permutation(PERM_SIZE)
        grad = np.random.uniform(-1, 1, PERM_SIZE)
        
        F2 = np.float64(0.5 * (np.sqrt(3.0) - 1.0))
        G2 = np.float64((3.0 - np.sqrt(3.0)) / 6.0)
        
        s = t * F2
        i = np.floor(s)
        t0 = t - (i - i * G2)
        
        i0 = i.astype(int) & (PERM_SIZE - 1)
        i1 = (i0 + 1) & (PERM_SIZE - 1)
        
        g0 = grad[perm[i0]] * t0
        g1 = grad[perm[i1]] * (t0 - 1.0 + G2)
        
        return np.float64(70.0) * (g0 + g1)

    def _value_noise(self, t: np.ndarray, seed: int) -> np.ndarray:
        """Generate high precision Value noise"""
        seed = seed & 0xFFFFFFFF
        np.random.seed(seed)
        
        PERM_SIZE = 256
        values = np.random.uniform(-1, 1, PERM_SIZE)
        perm = np.random.permutation(PERM_SIZE)
        
        ti = np.floor(t).astype(int) & (PERM_SIZE - 1)
        tf = t - np.floor(t)
        
        u = self._smooth_step(tf)
        
        v0 = values[perm[ti]]
        v1 = values[perm[(ti + 1) & (PERM_SIZE - 1)]]
        
        return self._lerp(v0, v1, u)

    def _cellular_noise(self, t: np.ndarray, seed: int) -> np.ndarray:
        """Generate high precision Cellular/Worley noise"""
        seed = seed & 0xFFFFFFFF
        np.random.seed(seed)
        
        NUM_POINTS = 32
        points = np.sort(np.random.uniform(0, 1, NUM_POINTS))
        
        distances = np.abs(t.reshape(-1, 1) - points.reshape(1, -1))
        result = np.min(distances, axis=1)
        
        return 1.0 - 2.0 * result

    def _generate_noise(self, params: Dict[str, Any], time_values: np.ndarray) -> np.ndarray:
        """Optimized noise generation with caching"""
        cache_key = (
            params['noise_type'],
            params.get('frequency', 1.0),
            params.get('temporal_phase', 0.0),
            params.get('spatial_phase', 0.0),
            params.get('seed', 0),
            tuple(time_values)
        )
        
        if cache_key in self._noise_cache:
            return self._noise_cache[cache_key]
            
        t = time_values * 2 * np.pi * params.get('frequency', 1.0) + \
            np.radians(params.get('temporal_phase', 0.0))
        
        if params['noise_type'] == 'perlin':
            noise = self._perlin_noise(t, params['seed'])
        elif params['noise_type'] == 'simplex':
            noise = self._simplex_noise(t, params['seed'])
        elif params['noise_type'] == 'value':
            noise = self._value_noise(t, params['seed'])
        else:  # cellular
            noise = self._cellular_noise(t, params['seed'])
            
        if params.get('time_remap_curve') == 'smooth':
            noise = self._smooth_step(noise * 0.5 + 0.5) * 2.0 - 1.0
            
        # Cache result
        self._noise_cache[cache_key] = noise
        
        # Limit cache size
        if len(self._noise_cache) > 1000:
            # Remove oldest entries
            for k in list(self._noise_cache.keys())[:100]:
                del self._noise_cache[k]
                
        return noise

    def _generate_centered_noise(self, frequency: float, magnitude: float, 
                               time: float, temporal_phase: float,
                               spatial_phase: float, noise_type: str,
                               octaves: int, persistence: float,
                               lacunarity: float, time_remap: str,
                               seed: int) -> float:
        """Generate ultra-smooth, continuous noise with improved interpolation"""
        
        # Zmniejszamy bazową częstotliwość dla płynniejszego ruchu
        base_frequency = frequency * 0.2
        result = 0.0
        amplitude_sum = 0.0
        
        # Dodajemy bardzo wolny ruch bazowy
        base_time = time * 0.1
        base_params = {
            'frequency': base_frequency * 0.05,
            'temporal_phase': temporal_phase,
            'spatial_phase': spatial_phase,
            'noise_type': noise_type,
            'octaves': 1,
            'persistence': 1.0,
            'lacunarity': 1.0,
            'time_remap_curve': 'smooth',
            'seed': seed + 12345
        }
        
        base_motion = self._generate_noise(base_params, np.array([base_time]))[0]
        
        # Dodajemy warstwy oktaw z płynną interpolacją
        for i in range(octaves):
            # Każda oktawa ma własne przesunięcie fazowe i czasowe
            phase_offset = i * np.pi / octaves
            time_offset = np.sin(time * 0.05 + i * 2.39) * 0.1
            
            current_frequency = base_frequency * (lacunarity ** (i * 0.7))
            current_persistence = persistence ** (i * 0.8)
            
            octave_params = {
                'frequency': current_frequency,
                'temporal_phase': temporal_phase + phase_offset * 180.0,
                'spatial_phase': spatial_phase + i * 60.0,
                'noise_type': noise_type,
                'octaves': 1,
                'persistence': 1.0,
                'lacunarity': 1.0,
                'time_remap_curve': 'smooth',
                'seed': seed + i * 1000
            }
            
            t = time + time_offset
            octave_noise = self._generate_noise(octave_params, np.array([t]))[0]
            
            # Płynniejsze łączenie oktaw
            current_amplitude = current_persistence * (1.0 - i / octaves * 0.3)
            result += octave_noise * current_amplitude
            amplitude_sum += current_amplitude
        
        # Łączymy ruchy z bazowym
        if amplitude_sum > 0:
            result = result / amplitude_sum
        result = result * 0.7 + base_motion * 0.3
        
        # Aplikujemy dodatkowe wygładzanie
        if time_remap == 'smooth':
            result = self._smooth_step(result * 0.5 + 0.5) * 2.0 - 1.0
        elif time_remap == 'ease':
            result = self._ease_both(result * 0.5 + 0.5) * 2.0 - 1.0
        
        return np.clip(result * magnitude / 2.0, -magnitude/2, magnitude/2)
        
    def _calculate_transform_matrix(self, position: Tuple[float, float],
                                  scale: Tuple[float, float],
                                  rotation: float,
                                  frame_width: int,
                                  frame_height: int) -> Tuple[float, ...]:
        """Calculate affine transform matrix with high precision handling"""
        # Jeśli brak transformacji, zwróć macierz jednostkową
        if (position == (0.0, 0.0) and 
            scale == (1.0, 1.0) and 
            rotation == 0.0):
            return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        # Convert to high precision
        tx, ty = np.float64(position[0]), np.float64(position[1])
        sx, sy = np.float64(scale[0]), np.float64(scale[1])
        angle_rad = np.float64(math.radians(rotation))
        
        # Calculate dimensions
        W = np.float64(frame_width)
        H = np.float64(frame_height)
        
        # Image center point
        ox = W / 2
        oy = H / 2
        
        # Obliczenia wykorzystujące wysoką precyzję
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Macierz skalowania i rotacji
        a = sx * cos_theta
        b = -sx * sin_theta
        d = sy * sin_theta
        e = sy * cos_theta
        
        # Korekta pozycji względem środka
        tx_adjusted = ox + tx - (a * ox + b * oy)
        ty_adjusted = oy + ty - (d * ox + e * oy)
        
        # Final matrix components with high precision
        c = float(np.format_float_positional(tx_adjusted, precision=16, unique=True))
        f = float(np.format_float_positional(ty_adjusted, precision=16, unique=True))
        
        return tuple(float(np.format_float_positional(x, precision=16, unique=True))
                    for x in (a, b, c, d, e, f))

    def _prepare_motion_tile_canvas(self, image: Image.Image, params: Dict[str, Any]) -> Tuple[Image.Image, int]:
        """Prepare extended canvas with proper mirror edges"""
        if params['motion_tile'] == "disabled":
            return image, 0

        W, H = image.size
        
        # Obliczanie paddingu z walidacją
        if params['motion_tile_auto_margin']:
            base_padding = min(W, H) * 0.2  # Zmniejszone z 0.5 dla lepszej wydajności
        else:
            margin_percent = params['motion_tile_margin'] / 100.0
            base_padding = min(W, H) * margin_percent
        
        # Limit paddingu do 25% oryginalnego rozmiaru
        padding = int(min(base_padding, min(W, H) * 0.25))
        
        if params['motion_tile'] == "mirror":
            # Tworzenie canvasu o dokładnym rozmiarze
            canvas_width = W + 2 * padding
            canvas_height = H + 2 * padding
            canvas = Image.new(image.mode, (canvas_width, canvas_height))
            
            # 1. Umieszczenie oryginalnego obrazu w centrum
            canvas.paste(image, (padding, padding))
            
            # 2. Odbicia krawędzi
            # Górna krawędź
            top_strip = image.crop((0, 0, W, padding))
            top_strip = ImageOps.flip(top_strip)
            canvas.paste(top_strip, (padding, 0))
            
            # Dolna krawędź
            bottom_strip = image.crop((0, H-padding, W, H))
            bottom_strip = ImageOps.flip(bottom_strip)
            canvas.paste(bottom_strip, (padding, H+padding))
            
            # Lewa krawędź
            left_strip = image.crop((0, 0, padding, H))
            left_strip = ImageOps.mirror(left_strip)
            canvas.paste(left_strip, (0, padding))
            
            # Prawa krawędź
            right_strip = image.crop((W-padding, 0, W, H))
            right_strip = ImageOps.mirror(right_strip)
            canvas.paste(right_strip, (W+padding, padding))
            
            # 3. Narożniki z podwójnym odbiciem
            # Lewy górny
            corner_tl = image.crop((0, 0, padding, padding))
            corner_tl = ImageOps.mirror(ImageOps.flip(corner_tl))
            canvas.paste(corner_tl, (0, 0))
            
            # Prawy górny
            corner_tr = image.crop((W-padding, 0, W, padding))
            corner_tr = ImageOps.mirror(ImageOps.flip(corner_tr))
            canvas.paste(corner_tr, (W+padding, 0))
            
            # Lewy dolny
            corner_bl = image.crop((0, H-padding, padding, H))
            corner_bl = ImageOps.mirror(ImageOps.flip(corner_bl))
            canvas.paste(corner_bl, (0, H+padding))
            
            # Prawy dolny
            corner_br = image.crop((W-padding, H-padding, W, H))
            corner_br = ImageOps.mirror(ImageOps.flip(corner_br))
            canvas.paste(corner_br, (W+padding, H+padding))
            
        elif params['motion_tile'] == "repeat":
            canvas_width = W + 2 * padding
            canvas_height = H + 2 * padding
            canvas = Image.new(image.mode, (canvas_width, canvas_height))
            
            for y in range(-1, 2):
                for x in range(-1, 2):
                    canvas.paste(image, (
                        padding + x * W,
                        padding + y * H
                    ))

        return canvas, padding

    def _apply_motion_blur(self, image: Image.Image, transforms: Dict[str, Any], params: Dict[str, Any]) -> Image.Image:
        """Apply motion blur based on movement velocity and direction"""
        if params['motion_blur'] <= 0:
            return image

        # Obliczanie wektora prędkości z transformacji
        pos_x, pos_y = transforms['position']
        scale_x, scale_y = transforms['scale']
        rotation = transforms['rotation']
        
        # Konwersja do wektorów numpy dla lepszych obliczeń
        velocity_vector = np.array([pos_x, pos_y], dtype=np.float64)
        scale_vector = np.array([scale_x - 1.0, scale_y - 1.0], dtype=np.float64)
        rotation_rad = math.radians(rotation)
        
        # Obliczanie całkowitej prędkości ruchu
        velocity_magnitude = np.linalg.norm(velocity_vector)
        
        # Uwzględnienie skali w prędkości
        scale_contribution = np.linalg.norm(scale_vector) * min(image.width, image.height) * 0.1
        
        # Uwzględnienie rotacji w prędkości
        # Prędkość rotacji jest proporcjonalna do odległości od środka
        rotation_radius = min(image.width, image.height) * 0.5
        rotation_velocity = abs(rotation_rad) * rotation_radius
        
        # Obliczanie kierunku rozmycia
        if velocity_magnitude > 0:
            # Jeśli jest ruch translacyjny, używamy jego kierunku
            blur_angle = math.atan2(velocity_vector[1], velocity_vector[0])
            direction_weight = 0.8  # Waga dla kierunku ruchu translacyjnego
        else:
            # Jeśli nie ma ruchu translacyjnego, używamy kierunku rotacji
            blur_angle = math.copysign(math.pi/2, rotation_rad)
            direction_weight = 0.2  # Mniejsza waga dla samej rotacji
            
        # Kombinacja wszystkich prędkości z odpowiednimi wagami
        total_velocity = (velocity_magnitude * 1.0 +  # Waga dla ruchu translacyjnego
                         scale_contribution * 0.5 +   # Waga dla zmiany skali
                         rotation_velocity * 0.3)     # Waga dla rotacji
        
        # Obliczanie siły rozmycia bazując na prędkości i parametrze motion_blur
        blur_strength = total_velocity * params['motion_blur']
        
        # Limit maksymalnej siły rozmycia
        max_blur = min(image.width, image.height) * 0.15
        blur_strength = min(blur_strength, max_blur)
        
        if blur_strength < 0.5:  # Próg minimalny dla widocznego efektu
            return image
        
        # Obliczanie składowych wektora rozmycia
        blur_x = math.cos(blur_angle) * blur_strength
        blur_y = math.sin(blur_angle) * blur_strength
        
        # Adaptacyjna liczba kroków bazująca na sile rozmycia
        num_steps = max(3, min(12, int(blur_strength / 1.5)))
        
        # Inicjalizacja obrazu wynikowego
        blurred = Image.new(image.mode, image.size, (0,) * len(image.mode))
        accumulated_weight = 0.0
        
        # Zaawansowana implementacja motion blur
        for i in range(num_steps):
            # Używamy rozkładu gaussowskiego dla próbek
            t = (i / (num_steps - 1) - 0.5) * 2.0
            
            # Pozycja próbki z rozkładem gaussowskim
            gaussian_weight = math.exp(-4 * t * t)  # Szerokość rozkładu Gaussa
            
            # Obliczanie offsetu dla danej próbki
            offset_x = blur_x * t
            offset_y = blur_y * t
            
            # Tworzenie macierzy transformacji
            matrix = (1, 0, offset_x, 0, 1, offset_y)
            
            # Stosujemy transformację z wysoką jakością interpolacji
            offset_image = image.transform(
                image.size,
                Image.AFFINE,
                matrix,
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0,) * len(image.mode)
            )
            
            # Aktualizacja wag
            weight = gaussian_weight
            accumulated_weight += weight
            
            # Łączenie z wynikiem
            if i == 0:
                blurred = Image.blend(blurred, offset_image, weight)
            else:
                blurred = Image.blend(blurred, offset_image, weight / accumulated_weight)
            
        # Finalne mieszanie z oryginalnym obrazem dla zachowania detali
        final_blend_factor = min(1.0, blur_strength / 50.0)  # Adaptacyjny współczynnik mieszania
        return Image.blend(image, blurred, final_blend_factor)
        
    def _prepare_transforms(self, num_frames: int, params: dict) -> List[Dict[str, Any]]:
        """Prepare transformation parameters centered around image center"""
        logger.info(f"{self.BOLD}Preparing transforms for {num_frames} frames{self.ENDC}")
        logger.info(f"\n{self.HEADER}Generating transformation parameters:{self.ENDC}")
        logger.info(f"{self.YELLOW}{'-' * 30}{self.ENDC}")
        
        start_time = get_time()
        
        # Zmiana interpretacji parametru speed - teraz większa wartość = szybszy ruch
        adjusted_speed = params['speed']  # Teraz używamy wartości bezpośrednio
        duration = num_frames / params['fps']  # Base duration
        
        # Obliczamy czas dla każdej klatki z nową interpretacją prędkości
        base_time_values = np.linspace(
            params['time_offset'],
            params['time_offset'] + (duration * adjusted_speed),  # Mnożymy przez speed zamiast dzielić
            num_frames,
            dtype=np.float64
        )
        
        transforms_list = []
        
        # Logowanie informacji o prędkości
        logger.info(f"{self.BLUE}Animation speed: {adjusted_speed}x{self.ENDC}")
        
        # Optymalizacja - wyliczenie podstawowych parametrów
        enable_position = params.get('enable_position', True)
        enable_scale = params.get('enable_scale', True)
        enable_rotation = params.get('enable_rotation', True)
        
        # Przygotowanie podstawowych parametrów szumu
        noise_type = params['noise_type']
        octaves = params['noise_octaves']
        persistence = params['noise_persistence']
        lacunarity = params['noise_lacunarity']
        time_remap = params['time_remap_curve']
        
        for t in base_time_values:
            if len(transforms_list) % 10 == 0:
                progress = (len(transforms_list) / num_frames) * 100
                progress_bar = "=" * int(progress/5) + "-" * (20 - int(progress/5))
                logger.info(f"{self.GREEN}[{progress_bar}] {progress:.1f}% - Transform {len(transforms_list)}/{num_frames}{self.ENDC}")
            
            # Inicjalizacja słownika transformacji dla bieżącej klatki
            transforms = {}

            # Position
            if enable_position:
                if params['position_mode'] == 'independent':
                    # Niezależne generowanie dla X i Y
                    pos_x = self._generate_centered_noise(
                        frequency=params['position_frequency'],
                        magnitude=params['position_magnitude'],
                        time=t,
                        temporal_phase=params['position_temporal_phase'],
                        spatial_phase=params['position_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed']
                    )
                    
                    pos_y = self._generate_centered_noise(
                        frequency=params['position_frequency'],
                        magnitude=params['position_magnitude'],
                        time=t,
                        temporal_phase=params['position_temporal_phase'] + 90.0,
                        spatial_phase=params['position_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed'] + 12345
                    )
                else:  # linked
                    # Generowanie ruchu po okręgu/elipsie
                    angle = self._generate_centered_noise(
                        frequency=params['position_frequency'],
                        magnitude=2 * np.pi,
                        time=t,
                        temporal_phase=params['position_temporal_phase'],
                        spatial_phase=params['position_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed']
                    )
                    
                    radius = self._generate_centered_noise(
                        frequency=params['position_frequency'] * 1.1,
                        magnitude=params['position_magnitude'],
                        time=t,
                        temporal_phase=params['position_temporal_phase'],
                        spatial_phase=params['position_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed'] + 67890
                    )
                    
                    pos_x = np.cos(angle) * radius
                    pos_y = np.sin(angle) * radius
                
                transforms['position'] = (float(pos_x), float(pos_y))
            else:
                transforms['position'] = (0.0, 0.0)
            
            # Scale
            if enable_scale and params['scale_mode'] != 'none':
                if params['scale_mode'] == 'linked':
                    scale_noise = self._generate_centered_noise(
                        frequency=params['scale_frequency'],
                        magnitude=params['scale_magnitude'] / 100.0,
                        time=t,
                        temporal_phase=params['scale_temporal_phase'],
                        spatial_phase=params['scale_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed'] + 2
                    )
                    scale = 1.0 + scale_noise
                    transforms['scale'] = (float(scale), float(scale))
                else:  # independent
                    scale_x = 1.0 + self._generate_centered_noise(
                        frequency=params['scale_frequency'],
                        magnitude=params['scale_magnitude'] / 100.0,
                        time=t,
                        temporal_phase=params['scale_temporal_phase'],
                        spatial_phase=params['scale_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed'] + 2
                    )
                    scale_y = 1.0 + self._generate_centered_noise(
                        frequency=params['scale_frequency'],
                        magnitude=params['scale_magnitude'] / 100.0,
                        time=t,
                        temporal_phase=params['scale_temporal_phase'] + 90.0,
                        spatial_phase=params['scale_spatial_phase'],
                        noise_type=noise_type,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        time_remap=time_remap,
                        seed=params['seed'] + 3
                    )
                    transforms['scale'] = (float(scale_x), float(scale_y))
            else:
                transforms['scale'] = (1.0, 1.0)
            
            # Rotation
            if enable_rotation:
                rotation = self._generate_centered_noise(
                    frequency=params['rotation_frequency'],
                    magnitude=params['rotation_magnitude'],
                    time=t,
                    temporal_phase=params['rotation_temporal_phase'],
                    spatial_phase=params['rotation_spatial_phase'],
                    noise_type=noise_type,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    time_remap=time_remap,
                    seed=params['seed'] + 4
                )
                transforms['rotation'] = float(rotation)
            else:
                transforms['rotation'] = 0.0
            
            transforms_list.append(transforms)
        
        elapsed = get_time() - start_time
        logger.info(f"{self.BOLD}Transforms prepared in {elapsed:.2f}s{self.ENDC}")
        return transforms_list
        
    def _process_single_frame(self, frame_data: Dict[str, Any]) -> torch.Tensor:
        """Process single frame with optimized memory usage"""
        try:
            frame_number = frame_data.get('frame_number', 0)
            if frame_number % 10 == 0:
                logger.info(f"{self.BLUE}Processing frame {frame_number}{self.ENDC}")
            
            # Resize
            pil_image = frame_data['image']
            transforms = frame_data['transforms']
            params = frame_data['params']
            target_size = frame_data['target_size']
            
            resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Motion tile
            extended_image, padding = self._prepare_motion_tile_canvas(resized_image, params)
            
            # Calculate and apply transform
            matrix = self._calculate_transform_matrix(
                transforms['position'],
                transforms['scale'],
                transforms['rotation'],
                params['frame_width'],
                params['frame_height']
            )
            
            transformed = extended_image.transform(
                extended_image.size,
                Image.AFFINE,
                matrix,
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0,) * len(extended_image.mode)
            )
            
            # Motion blur if enabled
            if params['motion_blur'] > 0:
                transformed = self._apply_motion_blur(transformed, transforms, params)
            
            # Crop to final size
            final_image = transformed.crop((
                padding,
                padding,
                padding + target_size[0],
                padding + target_size[1]
            ))
            
            return self.convert_pil_to_tensor(
                final_image,
                (target_size[1], target_size[0], len(final_image.getbands()))
            )
            
        except Exception as e:
            logger.error(f"{self.RED}Error processing frame {frame_number}: {e}{self.ENDC}")
            raise

    def apply_wiggle(self, image: torch.Tensor, **params) -> tuple:
        try:
            self._increment_counter()
            
            if params.get('buy_me_a_coffee', False):
                self._open_coffee_link()
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Starting wiggle effect processing{self.ENDC}")
            logger.info(f"{self.YELLOW}{'=' * 50}{self.ENDC}")
            
            start_time = get_time()
            
            # Initialize and check memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Limit memory usage to 80%
                available_memory = int(gpu_memory * 0.8)
                logger.info(f"{self.BLUE}GPU Memory Limit: {available_memory/1024**2:.0f}MB{self.ENDC}")
            else:
                # Fallback to system memory if GPU not available
                system_memory = psutil.virtual_memory().available
                available_memory = int(system_memory * 0.8)
                logger.info(f"{self.BLUE}System Memory Limit: {available_memory/1024**2:.0f}MB{self.ENDC}")
            
            # Get input image dimensions
            B, H, W, C = image.shape
            
            # Determine target dimensions based on preserve_size
            if params['preserve_size']:
                target_W = W
                target_H = H
                logger.info(f"{self.BLUE}Preserving input dimensions: {W}x{H}{self.ENDC}")
            else:
                target_W = params['frame_width']
                target_H = params['frame_height']
                logger.info(f"{self.BLUE}Using custom dimensions: {target_W}x{target_H}{self.ENDC}")
                
            target_size = (target_W, target_H)
            
            logger.info(f"{self.BLUE}Processing {B} frames at {target_W}x{target_H}{self.ENDC}")
                
            # Calculate optimal batch size
            single_frame_size = target_W * target_H * C * 4  # 4 bytes per float32
            if torch.cuda.is_available():
                max_frames_in_memory = available_memory // (single_frame_size * 3)  # Use available_memory instead of memory_limit
                batch_size = min(16, max(1, max_frames_in_memory // 2))
            else:
                batch_size = 4
            
            logger.info(f"{self.BLUE}Batch size: {batch_size}{self.ENDC}")
            
            # Generate transforms
            transforms_list = self._prepare_transforms(B, params)
            
            logger.info(f"\n{self.HEADER}Processing frames in batches:{self.ENDC}")
            logger.info(f"{self.YELLOW}{'-' * 30}{self.ENDC}")
            
            # Process frames in batches
            processed_frames = []
            
            for batch_start in range(0, B, batch_size):
                batch_end = min(batch_start + batch_size, B)
                progress = (batch_start / B) * 100
                progress_bar = "=" * int(progress/5) + "-" * (20 - int(progress/5))
                logger.info(f"{self.GREEN}[{progress_bar}] {progress:.1f}% - Batch {batch_start}-{batch_end}/{B}{self.ENDC}")
                
                current_batch = image[batch_start:batch_end].contiguous()
                
                try:
                    # Prepare frame data for processing
                    frame_data_list = []
                    for i in range(len(current_batch)):
                        frame_index = batch_start + i
                        frame_data = {
                            'image': self.convert_tensor_to_pil(current_batch[i]),
                            'transforms': transforms_list[frame_index],
                            'params': params,
                            'target_size': target_size,
                            'frame_number': frame_index
                        }
                        frame_data_list.append(frame_data)
                    
                    # Process frames in parallel
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        batch_results = list(executor.map(self._process_single_frame, frame_data_list))
                    
                    # Stack batch results
                    batch_tensor = torch.stack(batch_results)
                    processed_frames.append(batch_tensor)
                    
                finally:
                    # Clean up after batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Optional garbage collection
                if batch_end % (batch_size * 4) == 0:
                    import gc
                    gc.collect()
            
            # Combine results
            result = torch.cat(processed_frames, dim=0)
            
            total_time = get_time() - start_time
            fps = B / total_time
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Processing complete!{self.ENDC}")
            logger.info(f"{self.GREEN}Total frames processed: {B}{self.ENDC}")
            logger.info(f"{self.GREEN}Processing time: {total_time:.2f}s ({fps:.1f} fps){self.ENDC}")
            logger.info(f"{self.YELLOW}{'=' * 50}{self.ENDC}")
            
            return (result, None)
            
        except Exception as e:
            logger.error(f"{self.RED}Error in apply_wiggle: {str(e)}{self.ENDC}")
            raise
            
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            self._noise_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "WiggleEffect": WiggleEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WiggleEffect": "After Effects Wiggle"
}