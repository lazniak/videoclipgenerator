#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import uuid
from pathlib import Path
import ffmpeg
import numpy as np
import subprocess
import tempfile
import wave
import folder_paths
import logging
import torch
import torchaudio
import json
import webbrowser
from datetime import datetime

COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

# Konfiguracja logowania
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'video_merger_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VideoMerger')

class VideoMerger:
    """Node for merging multiple video files with customizable codecs"""
    
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
            logger.error(f"Error loading counter: {e}")
        return 0
    
    # Inicjalizacja licznika przy starcie klasy używając metody statycznej
    _current_usage_count = _load_counter_static()
    
    @classmethod
    def CODECS(cls):
        return {
            "h264": {
                "name": "H.264/AVC",
                "ffmpeg_name": "libx264",
                "formats": ["mp4", "mkv", "mov"]
            },
            "h265": {
                "name": "H.265/HEVC",
                "ffmpeg_name": "libx265",
                "formats": ["mp4", "mkv", "mov"]
            },
            "vp9": {
                "name": "VP9",
                "ffmpeg_name": "libvpx-vp9",
                "formats": ["webm", "mkv"]
            },
            "mpeg4": {
                "name": "MPEG-4",
                "ffmpeg_name": "mpeg4",
                "formats": ["avi", "mkv"]
            },
            "copy": {
                "name": "Stream Copy (No re-encode)",
                "ffmpeg_name": "copy",
                "formats": ["mp4", "mkv", "mov", "avi", "webm"]
            }
        }
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = os.path.join(self.output_dir, "temp_merger")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count
        
        self.output_formats = {
            "mp4": ".mp4",
            "mkv": ".mkv",
            "mov": ".mov",
            "avi": ".avi",
            "webm": ".webm"
        }
        
        self.video_codecs = {k: v["name"] for k, v in self.CODECS().items()}
        
        self.audio_codecs = {
            "aac": "AAC",
            "mp3": "MP3",
            "ac3": "AC3",
            "pcm_s16le": "PCM 16-bit WAV",
            "pcm_s24le": "PCM 24-bit WAV",
            "pcm_f32le": "PCM 32-bit Float WAV"
        }
        
        self.video_quality_presets = {
            "medium": "Medium (Default)",
            "veryslow": "Very Slow (Best Quality)",
            "slow": "Slow (Better Quality)",
            "fast": "Fast (Good Quality)",
            "veryfast": "Very Fast (Lower Quality)",
            "ultrafast": "Ultra Fast (Lowest Quality)"
        }

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
        self.__class__._current_usage_count = self.usage_count
        logger.info(f"{self.BLUE}Usage count updated: {self.usage_count}{self.ENDC}")

    def _open_coffee_link(self):
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            logger.error(f"Error opening coffee link: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "output_filename": ("STRING", {
                    "multiline": False,
                    "default": "FinalVideo"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "output_format": (list(cls().output_formats.keys()),),
                "video_codec": (list(cls().video_codecs.keys()),),
                "video_quality": (list(cls().video_quality_presets.keys()),),
                "video_bitrate": ("STRING", {
                    "multiline": False,
                    "default": "8M"
                }),
                "save_mode": (["overwrite", "iterate"],),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support development with a coffee! ☕"
                }),
                "usage_counter": ("STRING", {
                    "default": f"Usage count: {cls._current_usage_count}",
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks how many times this node has been used across all your projects"
                })
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_codec": (list(cls().audio_codecs.keys()),),
                "audio_bitrate": ("STRING", {
                    "multiline": False,
                    "default": "192k"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "merge_videos"
    CATEGORY = "video"

    def get_suitable_format(self, video_codec):
        """Wybiera odpowiedni format wyjściowy dla danego kodeka"""
        codec_info = self.CODECS().get(video_codec)
        if not codec_info:
            return "mp4"
        return codec_info['formats'][0]

    def get_output_path(self, base_filename, save_mode, video_codec, requested_format=None):
        """Generuje ścieżkę wyjściową w zależności od trybu zapisu i kodeka"""
        base_name = os.path.splitext(base_filename)[0]
        
        output_format = requested_format
        if output_format not in self.CODECS()[video_codec]['formats']:
            output_format = self.get_suitable_format(video_codec)
            logger.warning(f"Format {requested_format} is not compatible with codec {video_codec}. Using {output_format} instead.")
        
        extension = self.output_formats[output_format]
        
        if save_mode == "iterate":
            counter = 1
            while True:
                new_path = os.path.join(self.output_dir, f"{base_name}_{counter:03d}{extension}")
                if not os.path.exists(new_path):
                    return new_path, output_format
                counter += 1
        else:  # "overwrite"
            return os.path.join(self.output_dir, f"{base_name}{extension}"), output_format

    def verify_audio_file(self, audio_path):
        """Weryfikuje właściwości pliku audio"""
        try:
            info = torchaudio.info(audio_path)
            logger.info("Audio file verification:")
            logger.info(f"- Sample rate: {info.sample_rate} Hz")
            logger.info(f"- Num channels: {info.num_channels}")
            logger.info(f"- Num frames: {info.num_frames}")
            logger.info(f"- Duration: {info.num_frames / info.sample_rate:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Audio file verification failed: {str(e)}")
            return False

    def calculate_video_duration(self, video_files, fps):
        """Oblicza całkowitą długość wideo w sekundach"""
        try:
            total_frames = 0
            for video_file in video_files:
                probe = ffmpeg.probe(video_file)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                duration = float(video_info['duration'])
                frames = int(duration * fps)
                total_frames += frames
            
            total_duration = total_frames / fps
            logger.info(f"Total video duration: {total_duration:.2f} seconds ({total_frames} frames at {fps} fps)")
            return total_duration, total_frames
        except Exception as e:
            logger.error(f"Error calculating video duration: {str(e)}")
            return None, None

    def process_audio_data(self, audio_data, video_duration):
        """Przetwarza i analizuje dane audio, zawsze przycinając do długości wideo"""
        try:
            logger.info("Processing audio data")
            
            # Rozpakuj dane audio
            if isinstance(audio_data, dict):
                waveform = audio_data['waveform']
                sample_rate = int(audio_data['sample_rate'])
            else:
                waveform, sample_rate = audio_data
                sample_rate = int(sample_rate)

            # Konwertuj do numpy
            if isinstance(waveform, torch.Tensor):
                if len(waveform.shape) == 3:
                    waveform = waveform.squeeze(0)
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                waveform = waveform.cpu().numpy()

            # Oblicz długość audio i wideo w próbkach
            audio_samples = waveform.shape[-1]
            video_samples = int(video_duration * sample_rate)
            audio_duration = audio_samples / sample_rate

            logger.info(f"Original audio duration: {audio_duration:.2f} seconds ({audio_samples} samples)")
            logger.info(f"Video duration: {video_duration:.2f} seconds ({video_samples} samples)")

            # Zawsze przycinaj audio do długości wideo
            if audio_samples > video_samples:
                logger.info("Trimming audio to match video duration")
                waveform = waveform[..., :video_samples]
                logger.info(f"Trimmed audio duration: {video_duration:.2f} seconds ({video_samples} samples)")
            elif audio_samples < video_samples:
                logger.warning(f"Audio is shorter than video by {video_duration - audio_duration:.2f} seconds")

            # Normalizacja
            if waveform.dtype == np.float32 or waveform.dtype == np.float64:
                waveform = np.clip(waveform, -1.0, 1.0)
                waveform = (waveform * np.iinfo(np.int16).max).astype(np.int16)
            elif waveform.dtype != np.int16:
                waveform = waveform.astype(np.int16)

            # Format kanałów
            if len(waveform.shape) == 1:
                waveform = waveform.reshape(1, -1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]

            final_duration = waveform.shape[-1] / sample_rate
            logger.info(f"Final audio processing results:")
            logger.info(f"- Duration: {final_duration:.2f} seconds")
            logger.info(f"- Channels: {waveform.shape[0]}")
            logger.info(f"- Sample rate: {sample_rate} Hz")
            logger.info(f"- Samples: {waveform.shape[-1]}")

            return waveform, sample_rate
            
        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}")
            raise

    def save_audio_to_wav(self, audio_data, video_duration):
        """Zapisuje przetworzone audio do pliku WAV"""
        try:
            logger.info("Starting audio conversion to WAV")
            temp_audio_path = os.path.join(self.temp_dir, f'temp_audio_{uuid.uuid4()}.wav')
            
            waveform, sample_rate = self.process_audio_data(audio_data, video_duration)
            
            # Użyj torchaudio do zapisu pliku WAV
            waveform_tensor = torch.from_numpy(waveform)
            torchaudio.save(
                temp_audio_path,
                waveform_tensor,
                sample_rate,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            logger.info(f"Audio saved successfully to: {temp_audio_path}")
            
            # Weryfikacja zapisanego pliku
            info = torchaudio.info(temp_audio_path)
            logger.info(f"Saved file properties:")
            logger.info(f"- Sample rate: {info.sample_rate} Hz")
            logger.info(f"- Channels: {info.num_channels}")
            logger.info(f"- Duration: {info.num_frames / info.sample_rate:.2f} seconds")
            
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise

    def merge_videos(self, input_path, output_filename, fps, output_format, video_codec, 
                    video_quality, video_bitrate, save_mode="overwrite", 
                    audio=None, audio_codec="aac", audio_bitrate="192k",
                    buy_me_a_coffee=False, usage_counter=""):
        try:
            # Increment usage counter and handle coffee link
            self._increment_counter()
            if buy_me_a_coffee:
                self._open_coffee_link()

            # Sprawdź kompatybilność kodeka z formatem
            if output_format not in self.CODECS()[video_codec]['formats']:
                original_format = output_format
                output_format = self.get_suitable_format(video_codec)
                logger.warning(f"Format {original_format} is not compatible with codec {video_codec}. Using {output_format} instead.")
            
            logger.info(f"\n{self.HEADER}{self.BOLD}Starting video merge process with settings:{self.ENDC}")
            logger.info(f"{self.BLUE}- Input path: {input_path}")
            logger.info(f"- Output filename: {output_filename}")
            logger.info(f"- Output format: {output_format}")
            logger.info(f"- Video codec: {video_codec} ({video_bitrate})")
            logger.info(f"- Video quality: {video_quality}")
            logger.info(f"- Audio codec: {audio_codec} ({audio_bitrate})")
            logger.info(f"- FPS: {fps}{self.ENDC}")
            
            # Sprawdź ścieżki
            if not os.path.exists(input_path):
                error_msg = "Input path does not exist"
                logger.error(error_msg)
                return (error_msg,)
            
            # Znajdź pliki wideo
            video_files = []
            if os.path.isdir(input_path):
                for ext in self.output_formats.values():
                    found_files = glob.glob(os.path.join(input_path, f'*{ext}'))
                    video_files.extend(found_files)
                    logger.info(f"Found {len(found_files)} files with extension {ext}")
            else:
                error_msg = "Input path must be a directory"
                logger.error(error_msg)
                return (error_msg,)
            
            if not video_files:
                error_msg = "No video files found in directory"
                logger.error(error_msg)
                return (error_msg,)
            
            video_files.sort()
            
            # Oblicz całkowitą długość wideo
            video_duration, total_frames = self.calculate_video_duration(video_files, fps)
            if video_duration is None:
                return ("Error calculating video duration",)
            
            # Przygotuj ścieżkę wyjściową
            output_path, actual_format = self.get_output_path(output_filename, save_mode, video_codec, output_format)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Przygotuj plik z listą
            temp_list_path = os.path.join(self.temp_dir, f'temp_list_{uuid.uuid4()}.txt')
            with open(temp_list_path, 'w', encoding='utf-8') as f:
                for video_file in video_files:
                    f.write(f"file '{video_file}'\n")
            
            try:
                # Konfiguracja FFmpeg
                input_params = {
                    'f': 'concat',
                    'safe': '0'
                }
                
                output_params = {}
                
                # Parametry wideo
                if video_codec != "copy":
                    codec_info = self.CODECS()[video_codec]
                    output_params.update({
                        'vcodec': codec_info['ffmpeg_name'],
                        'b:v': video_bitrate,
                        'preset': video_quality,
                        'r': fps
                    })
                else:
                    output_params['vcodec'] = 'copy'
                
                # Przygotuj audio jeśli istnieje
                temp_audio_path = None
                if audio is not None:
                    try:
                        temp_audio_path = self.save_audio_to_wav(audio, video_duration)
                        if self.verify_audio_file(temp_audio_path):
                            audio_input = ffmpeg.input(temp_audio_path)
                            
                            # Parametry audio
                            output_params.update({
                                'acodec': audio_codec,
                                'strict': '-2'
                            })
                            
                            if not audio_codec.startswith('pcm_'):
                                output_params['b:a'] = audio_bitrate
                            
                            # Przygotuj strumień z audio
                            stream = ffmpeg.output(
                                ffmpeg.input(temp_list_path, **input_params),
                                audio_input,
                                output_path,
                                **output_params
                            )
                        else:
                            logger.error("Audio verification failed, continuing without audio")
                            stream = ffmpeg.output(
                                ffmpeg.input(temp_list_path, **input_params),
                                output_path,
                                **output_params
                            )
                    except Exception as e:
                        logger.error(f"Error setting up audio: {str(e)}")
                        stream = ffmpeg.output(
                            ffmpeg.input(temp_list_path, **input_params),
                            output_path,
                            **output_params
                        )
                else:
                    stream = ffmpeg.output(
                        ffmpeg.input(temp_list_path, **input_params),
                        output_path,
                        **output_params
                    )
                
                # Wykonaj FFmpeg
                logger.info(f"FFmpeg command: {' '.join(ffmpeg.compile(stream))}")
                ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
                logger.info(f"{self.GREEN}FFmpeg process completed successfully{self.ENDC}")
                
                # Zwróć względną ścieżkę
                relative_path = os.path.relpath(output_path, self.output_dir)
                logger.info(f"{self.GREEN}Successfully merged videos to: {relative_path}{self.ENDC}")
                return (relative_path,)
                
            finally:
                # Cleanup
                for temp_file in [temp_list_path, temp_audio_path]:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            logger.info(f"Removed temporary file: {temp_file}")
                        except Exception as e:
                            logger.error(f"Error removing temporary file {temp_file}: {str(e)}")
                            
        except Exception as e:
            error_msg = f"Error during merge: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "VideoMerger": VideoMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMerger": "Video Merger"
}