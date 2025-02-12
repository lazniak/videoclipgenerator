# File: optical_compensation_node.py
import torch
import numpy as np
from PIL import Image
import math
import logging
from time import time as get_time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import psutil
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import webbrowser
from functools import lru_cache
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OpticalCompensation')

class OpticalCompensationNode:
    """After Effects-style optical compensation node for ComfyUI with chromatic aberration and edge blur"""
    
    # Colors for logs
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
    
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "field_of_view": ("FLOAT", {
                    "default": 90.0,
                    "min": 1.0,
                    "max": 360.0,
                    "step": 0.1,
                    "tooltip": "Field of view angle in degrees"
                }),
                "convergence": ("FLOAT", {
                    "default": -45.0,
                    "min": -200.0,
                    "max": 200.0,
                    "step": 0.1,
                    "tooltip": "Controls the strength and direction of the compensation effect (+: concave, -: convex)"
                }),
                "chromatic_aberration": ("FLOAT", {
                    "default": -20.0,
                    "min": -200.0,
                    "max": 200.0,
                    "step": 0.1,
                    "tooltip": "Adds RGB channel separation"
                }),
                "blend_mode": (["screen", "additive"], {
                    "default": "screen",
                    "label": "Blend Mode"
                }),
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Amount of blur at the edges (0-1)"
                }),
                "edge_roundness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Controls the roundness of edge blur gradient (0-1)"
                }),
                "feather_width": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Width of the feathered edge gradient (0-1)"
                }),
                "interpolation": (["bicubic", "bilinear", "nearest"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation Method"
                }),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support development with a coffee! ☕"
                }),
                "usage_counter": ("STRING", {
                "default": f"Usage count: {cls._current_usage_count}",  # Używamy zmiennej klasowej
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks how many times this node has been used across all your projects"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_compensation"
    CATEGORY = "image/transform"

    def __init__(self):
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count

    @lru_cache(maxsize=128)
    def _create_compensation_map(self, width: int, height: int, fov: float, convergence: float, 
                               channel_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create lookup maps for optical compensation with channel offset for chromatic aberration"""
        # Normalize coordinates to [-1, 1]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)

        # Calculate radius from center
        R = np.sqrt(X**2 + Y**2)
        
        # Calculate angles (in radians) - wzmocniony efekt FOV
        fov_rad = np.radians(fov * 2.0)
        max_theta = fov_rad / 2
        
        # Calculate normalized radius (0 to 1)
        R_norm = R / np.max(R)
        
        # Apply convergence with channel offset
        convergence_with_offset = convergence + channel_offset
        convergence_factor = -(convergence_with_offset * 2.0) / 100.0
        
        # Calculate base distortion angle
        theta = R_norm * max_theta

        # Calculate FOV scaling adaptively
        fov_scale = 1.0 / (np.tan(fov_rad / 4) + 0.5)

        # Adaptive scaling factor to prevent black borders
        max_scale = 1.5
        if convergence_factor >= 0:
            factor = np.sin(theta) * (1 + convergence_factor * 1.5)
            scale_factor = 1.0 + abs(convergence_factor) * 0.3
        else:
            factor = np.tan(theta) * (1 - abs(convergence_factor) * 1.5)
            scale_factor = 1.0 + abs(convergence_factor) * 0.2
            
        # Limit scale factor
        scale_factor = min(scale_factor, max_scale)
            
        # Calculate the direction vectors from the center
        dir_x = X
        dir_y = Y
        
        # Normalize direction vectors where R > 0
        norm = np.where(R > 0, R, 1.0)
        dir_x = dir_x / norm
        dir_y = dir_y / norm
        
        # Apply distortion along these directions
        scale = np.where(R > 0, factor / (R + 1e-6), 0) * 1.5
        
        # Calculate the displacement vectors
        displacement_x = dir_x * R * scale * scale_factor
        displacement_y = dir_y * R * scale * scale_factor
        
        # Apply the displacement from the original coordinates with FOV scaling
        map_x = X + displacement_x * fov_scale
        map_y = Y + displacement_y * fov_scale
        
        # Find the current bounds of the mapping
        x_min, x_max = map_x.min(), map_x.max()
        y_min, y_max = map_y.min(), map_y.max()
        
        # Calculate scaling to fit within [-1, 1] while preserving aspect ratio
        x_scale = 2.0 / (x_max - x_min)
        y_scale = 2.0 / (y_max - y_min)
        scale = min(x_scale, y_scale)
        
        # Apply the scaling
        map_x = map_x * scale
        map_y = map_y * scale
        
        # Normalize to 0-1 range
        map_x = (map_x + 1) / 2
        map_y = (map_y + 1) / 2
        
        # Ensure the maps stay within valid range with soft clamping
        def soft_clamp(x, min_val=0, max_val=1, margin=0.1):
            return np.clip(x, min_val - margin, max_val + margin)
        
        map_x = soft_clamp(map_x)
        map_y = soft_clamp(map_y)

        return map_x.astype(np.float32), map_y.astype(np.float32)

    def _create_edge_mask(self, width: int, height: int, roundness: float, feather_width: float) -> np.ndarray:
        """Create a gradient mask for edge blurring"""
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate normalized radius
        R = np.sqrt(X**2 + Y**2)
        
        # Apply roundness transformation
        roundness = 1.0 - roundness  # Invert for more intuitive control
        R = np.power(R, 2.0 - roundness)
        
        # Create gradient
        inner_radius = 1.0 - feather_width
        mask = np.clip((1.0 - R) / feather_width, 0, 1)
        mask = np.where(R < inner_radius, 1.0, mask)
        
        return mask

    def _apply_edge_blur(self, image: np.ndarray, mask: np.ndarray, blur_strength: float) -> np.ndarray:
        """Apply edge blur with gradient mask"""
        if blur_strength <= 0:
            return image
            
        # Calculate adaptive kernel size based on image size and blur strength
        min_dim = min(image.shape[:2])
        max_kernel = int(min_dim * 0.1)  # Maximum 10% of smaller dimension
        kernel_size = int(max_kernel * blur_strength)
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        
        # Create blurred version
        blurred = ndimage.gaussian_filter(image, sigma=[kernel_size/3.0, kernel_size/3.0, 0])
        
        # Expand mask dimensions to match image
        mask = np.expand_dims(mask, axis=2) if mask.ndim == 2 else mask
        
        # Blend original and blurred based on mask
        result = image * mask + blurred * (1 - mask)
        
        return result

    def _apply_compensation_to_frame(self, frame_data: Dict[str, Any]) -> torch.Tensor:
        try:
            frame = frame_data['frame']
            params = frame_data['params']
            frame_number = frame_data.get('frame_number', 0)
            chromatic_aberration = params.get('chromatic_aberration', 0.0)
            blend_mode = params.get('blend_mode', 'screen')
            edge_blur = params.get('edge_blur', 0.0)
            edge_roundness = params.get('edge_roundness', 0.5)
            feather_width = params.get('feather_width', 0.3)

            if frame_number % 10 == 0:
                logger.info(f"{self.BLUE}Processing frame {frame_number}{self.ENDC}")

            # Convert to numpy
            np_frame = frame.numpy()
            height, width = np_frame.shape[:2]

            # Initialize result array
            result = np.zeros_like(np_frame)

            # Create grid for interpolation
            grid_x = np.linspace(0, width - 1, width)
            grid_y = np.linspace(0, height - 1, height)

            if abs(chromatic_aberration) > 0.001:  # If chromatic aberration is enabled
                # Process each channel separately with different convergence values
                channel_offsets = [
                    -chromatic_aberration,  # Red channel
                    0,                      # Green channel
                    chromatic_aberration    # Blue channel
                ]

                for channel, offset in enumerate(channel_offsets):
                    # Create maps for this channel
                    map_x, map_y = self._create_compensation_map(
                        width, height,
                        params['field_of_view'],
                        params['convergence'],
                        offset
                    )

                    # Create interpolator for this channel
                    interpolator = RegularGridInterpolator(
                        (grid_y, grid_x),
                        np_frame[..., channel],
                        method='linear',
                        bounds_error=False,
                        fill_value=0
                    )
                    
                    # Prepare points for interpolation
                    pts = np.stack([
                        map_y.flatten() * (height - 1),
                        map_x.flatten() * (width - 1)
                    ], axis=-1)
                    
                    # Apply interpolation
                    channel_result = interpolator(pts).reshape(height, width)
                    
                    # Apply blending mode
                    if blend_mode == 'screen':
                        result[..., channel] = channel_result
                    else:  # additive
                        result[..., channel] = channel_result

                # Normalize result based on blend mode
                if blend_mode == 'additive':
                    result = np.clip(result, 0, 1)

            else:  # No chromatic aberration
                # Standard processing for all channels
                map_x, map_y = self._create_compensation_map(
                    width, height,
                    params['field_of_view'],
                    params['convergence']
                )

                for c in range(np_frame.shape[2]):
                    interpolator = RegularGridInterpolator(
                        (grid_y, grid_x),
                        np_frame[..., c],
                        method='linear',
                        bounds_error=False,
                        fill_value=0
                    )
                    
                    pts = np.stack([
                        map_y.flatten() * (height - 1),
                        map_x.flatten() * (width - 1)
                    ], axis=-1)
                    
                    result[..., c] = interpolator(pts).reshape(height, width)

            # Apply edge blur if enabled
            if edge_blur > 0:
                # Create gradient mask
                edge_mask = self._create_edge_mask(width, height, edge_roundness, feather_width)
                
                # Apply edge blur
                result = self._apply_edge_blur(result, edge_mask, edge_blur)

            return torch.from_numpy(result)

        except Exception as e:
            logger.error(f"{self.RED}Error processing frame {frame_number}: {e}{self.ENDC}")
            raise

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
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            logger.error(f"Error opening coffee link: {e}")

    def apply_compensation(self, image: torch.Tensor, **params) -> tuple:
        try:
            self._increment_counter()
            
            if params.get('buy_me_a_coffee', False):
                self._open_coffee_link()
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Starting optical compensation{self.ENDC}")
            logger.info(f"{self.YELLOW}{'=' * 50}{self.ENDC}")
            
            start_time = get_time()
            
            # Get dimensions
            B, H, W, C = image.shape
            logger.info(f"{self.BLUE}Processing {B} frames at {W}x{H}{self.ENDC}")
            
            # Process frames in parallel
            frame_data_list = [
                {
                    'frame': image[i],
                    'params': params,
                    'frame_number': i
                }
                for i in range(B)
            ]

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_frames = list(executor.map(self._apply_compensation_to_frame, frame_data_list))

            # Stack results
            result = torch.stack(processed_frames)

            total_time = get_time() - start_time
            fps = B / total_time
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Processing complete!{self.ENDC}")
            logger.info(f"{self.GREEN}Total frames processed: {B}{self.ENDC}")
            logger.info(f"{self.GREEN}Processing time: {total_time:.2f}s ({fps:.1f} fps){self.ENDC}")
            logger.info(f"{self.YELLOW}{'=' * 50}{self.ENDC}")

            return (result,)

        except Exception as e:
            logger.error(f"{self.RED}Error in apply_compensation: {str(e)}{self.ENDC}")
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
    "OpticalCompensation": OpticalCompensationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpticalCompensation": "Optical Compensation"
}