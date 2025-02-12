import json
import webbrowser
import whisper
import os
import uuid
import math
import torch
import torchaudio
import folder_paths
import numba
import numpy as np
from pathlib import Path
import difflib
from datetime import datetime

# Na początku pliku, po importach:
COFFEE_LINK = "https://buymeacoffee.com/eyb8tkx3to"

class VibeMusicEngine:
    # Dodaj kolory dla logów oraz inicjalizację licznika
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
        self.temp_manager = None
        self.input_audio = None
        self.fps = None
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_counter(self) -> int:
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
        print(f"{self.BLUE}Usage count updated: {self.usage_count}{self.ENDC}")

    def _open_coffee_link(self):
        try:
            coffee_link = f"{COFFEE_LINK}/?coffees={self.usage_count}"
            webbrowser.open(coffee_link)
        except Exception as e:
            print(f"Error opening coffee link: {e}")
            
    @classmethod
    def INPUT_TYPES(cls):
        # Lista wspieranych języków przez Whisper
        WHISPER_LANGUAGES = [
            "auto", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", 
            "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", 
            "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", 
            "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", 
            "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", 
            "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", 
            "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh"
        ]

        return {
            "required": { 
                "audio": ("AUDIO", {
                    "tooltip": "Main audio input for speech-to-text processing. Should contain clear vocal content for optimal transcription."
                }),
                "choose_model": (
                    ["base", "tiny", "small", "medium", "large", "large-v2", "large-v3", "medium.en", "small.en", "base.en", "tiny.en"],
                    {
                        "default": "base",
                        "tooltip": "Select Whisper model size:\ntiny: Fastest, lowest accuracy\nbase: Good balance of speed/accuracy\nsmall: Better accuracy, slower\nmedium: High accuracy, slower\nlarge/large-v2: Best accuracy, slowest\n*.en models: Optimized for English"
                    }
                ),
                "source_language": (WHISPER_LANGUAGES, {
                    "default": "auto",
                    "tooltip": "Select the source language of the audio:\nauto: Automatically detect language\nen: English\npl: Polish\nde: German\netc."
                }),
                "target_language": (["source"] + WHISPER_LANGUAGES[1:], {
                    "default": "source",
                    "tooltip": "Select the target language for transcription:\nsource: Keep detected/specified source language\nen: English\npl: Polish\nde: German\netc."
                }),
                "continuation_text": ("STRING", {
                    "default": "<LOGICAL CONTINUATION OF PREVIOUS VIDEO SHOOT>",
                    "multiline": False,
                    "tooltip": "Text to be used for logical continuation of scenes. This text will be inserted when a scene is split into multiple parts."
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second for video output. Common values:\n24fps: Film\n25fps: PAL\n29.97fps: NTSC\n30fps: Digital\n60fps: High frame rate"
                }),
                "words_per_line": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of words to group per subtitle line. Lower values create shorter, more readable lines but more frequent cuts."
                }),
                "prev_context_words": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of previous words to add as context before the main text line. Helps with context understanding and prompt generation."
                }),
                "next_context_words": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of next words to add as context after the main text line. Helps with context understanding and prompt generation."
                }),
                "context_brackets": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, adds [PREV: ...] and [NEXT: ...] brackets around context words. If disabled, adds context words without brackets."
                }),
                "max_shot_length_frames": ("INT", {
                    "default": 125,
                    "min": 1,
                    "step": 1,
                    "tooltip": "Maximum length of a single shot in frames. Longer shots will be split. At 25fps, 125 frames = 5 seconds."
                }),
                "min_shot_length_frames": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Minimum length of a single shot in frames. Prevents too rapid cuts. At 25fps, 25 frames = 1 second."
                }),
                "filename_prefix": ("STRING", {
                    "default": "sc",
                    "tooltip": "Prefix for output files. Will be used for both EDL and SRT files. Example: 'sc' will create 'sc_00001.mp4', 'sc.edl', etc."
                }),
                "files_in_edl": ("STRING", {
                    "default": "mp4",
                    "tooltip": "File extension for video files referenced in EDL. Common formats: mp4, mov, avi, mxf. Must match your video workflow."
                }),
                "beat_detection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable automatic beat detection for music-synchronized cuts. Uses advanced audio analysis to find rhythmic patterns."
                }),
                "beat_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sensitivity of beat detection:\n0.1: Only strongest beats\n0.5: Balanced detection\n1.0: Detect subtle beats\nHigher values may create false positives"
                }),
                "beat_brutality": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strictly to align cuts to beats:\n0.0: Subtle alignment\n0.5: Balanced\n1.0: Exact beat alignment\nHigher values may create jarring cuts"
                }),
                "save_edl": (["overwrite", "no", "iterate"], {
                    "default": "overwrite",
                    "tooltip": "EDL save mode:\noverwrite: Replace existing file\nno: Don't save EDL\niterate: Create numbered versions (sc_001.edl, sc_002.edl, etc.)"
                }),
                "save_srt": (["no", "overwrite", "iterate"], {
                    "default": "overwrite",
                    "tooltip": "SRT subtitle save mode:\nno: Don't save SRT\noverwrite: Replace existing file\niterate: Create numbered versions"
                }),
                "srt_mode": (["raw", "clean", "word", "original", "all"], {
                    "default": "raw",
                    "tooltip": "Subtitle format mode:\nraw: Include all system messages\nclean: Only text content\nword: Word-by-word timing (karaoke style)\noriginal: Raw Whisper output\nall: Save all formats (raw, clean, word, and original)"
                }),
                "frame_gap": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Gap between words in frames when using word mode. Affects readability and timing precision."
                }),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support the development! ☕\nBy enabling this option, you'll be directed to a page where you can show your appreciation for this tool.\nYour support helps maintain and improve these nodes!"
                }),
                "usage_counter": ("STRING", {
                    "default": f"Usage count: {cls._current_usage_count}",
                    "multiline": False,
                    "readonly": True,
                    "tooltip": "Tracks the total number of times this node has been used.\nThis helps us understand how the tool is being utilized and guides future development.\nThe counter updates automatically with each use."
                })
            },
            "optional": {
                "beats_audio": ("AUDIO", {
                    "tooltip": "Optional separate audio input for beat detection. Useful when vocal track and music track are separate.\nIf not provided, main audio will be used for beat detection."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Lyrics", "frame_based_text", "Frame_long", "frame_numbers", "text_lines", "start_frame_numbers", "end_frame_numbers", "num_of_elements", "durations", "Start_ms", "Duration_ms")
    FUNCTION = "process_audio"
    CATEGORY = "VibeMusicEngine"
    OUTPUT_IS_LIST = (False, False, False, True, True, True, True, False, True, True, True)
    
    def create_original_whisper_srt(self, result, fps):
        """Creates SRT content directly from Whisper output"""
        srt_content = []
        counter = 1
        
        for segment in result["segments"]:
            start_time = self.frames_to_srt_time(int(segment["start"] * fps), fps)
            end_time = self.frames_to_srt_time(int(segment["end"] * fps), fps)
            
            srt_content.extend([
                str(counter),
                f"{start_time} --> {end_time}",
                segment["text"].strip(),
                ""
            ])
            counter += 1
        
        return "\n".join(srt_content)

    def save_srt_file(self, content, filename_prefix, save_mode, srt_type=""):
        """Saves SRT file with different modes and types"""
        if save_mode == "no":
            return None
            
        output_folder = self.ensure_output_folder(filename_prefix)
        base_name = os.path.basename(filename_prefix) if filename_prefix else "sc"
        
        # Add suffix for different SRT types
        suffix = f"_{srt_type}" if srt_type else ""
        
        if save_mode == "iterate":
            counter = 1
            while True:
                srt_path = os.path.join(output_folder, f"{base_name}_{counter:03d}{suffix}.srt")
                if not os.path.exists(srt_path):
                    break
                counter += 1
        else:  # "overwrite" or other modes
            srt_path = os.path.join(output_folder, f"{base_name}{suffix}.srt")
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\033[94mVibeMusicEngine:\033[0m Saved SRT file ({srt_type or 'default'}): {srt_path}")
        return srt_path

    class TempFileManager:
        """Manages temporary files for audio processing"""
        def __init__(self, base_dir, prefix="word_verify"):
            self.base_dir = base_dir
            self.prefix = prefix
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temp_files = []
            
            # Create session directory
            self.session_dir = os.path.join(base_dir, f"{prefix}_{self.session_id}")
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"\033[94mVibeMusicEngine:\033[0m Created temp directory: {self.session_dir}")

        def create_temp_path(self, word_index, stage):
            """Create a path for temporary file with meaningful name"""
            filename = f"{self.prefix}_{word_index:04d}_{stage}_{uuid.uuid4().hex[:8]}.wav"
            path = os.path.join(self.session_dir, filename)
            self.temp_files.append(path)
            return path

        def cleanup(self):
            """Clean up all temporary files and directory"""
            for file_path in self.temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"\033[93mVibeMusicEngine:\033[0m Failed to remove temp file {file_path}: {str(e)}")

            try:
                os.rmdir(self.session_dir)
                print(f"\033[94mVibeMusicEngine:\033[0m Removed temp directory: {self.session_dir}")
            except Exception as e:
                print(f"\033[93mVibeMusicEngine:\033[0m Failed to remove temp directory: {str(e)}")

    def process_audio(self, audio, choose_model, fps, source_language, target_language, 
                     words_per_line, prev_context_words, next_context_words, context_brackets,
                     max_shot_length_frames, min_shot_length_frames, filename_prefix, 
                     files_in_edl, beat_detection, beat_sensitivity, beat_brutality, 
                     save_edl, save_srt, srt_mode, continuation_text, frame_gap=1.0, beats_audio=None, 
                     buy_me_a_coffee=False, usage_counter=""):
        # Inicjalizacja zmiennej temp_path na początku
        temp_path = None
        
        try:
            self._increment_counter()
            if buy_me_a_coffee:
                self._open_coffee_link()
                
            print("\033[94mVibeMusicEngine:\033[0m Starting audio processing...")
            
            # Log CUDA status
            print(f"\033[94mVibeMusicEngine:\033[0m Using device: {self.device}")
            if self.device == "cuda":
                print(f"\033[94mVibeMusicEngine:\033[0m CUDA Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            # Store input audio and fps for later use
            self.input_audio = audio
            self.fps = fps
            
            # Validate parameters
            if max_shot_length_frames < min_shot_length_frames:
                print("\033[91mVibeMusicEngine:\033[0m ERROR: max_shot_length_frames must be greater than min_shot_length_frames")
                max_shot_length_frames = min_shot_length_frames * 2
                print(f"\033[94mVibeMusicEngine:\033[0m Automatically corrected max_shot_length_frames to {max_shot_length_frames}")

            files_in_edl = files_in_edl.lower().lstrip('.')
            if not files_in_edl:
                files_in_edl = "mp4"

            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            
            # Handle beat detection audio
            beats_waveform = None
            if beat_detection and beats_audio is not None:
                beats_waveform = beats_audio['waveform']
                beats_sample_rate = beats_audio['sample_rate']
                print("\033[94mVibeMusicEngine:\033[0m Detected separate audio for beat analysis")
            elif beat_detection:
                beats_waveform = waveform
                beats_sample_rate = sample_rate
                print("\033[94mVibeMusicEngine:\033[0m Using main audio for beat analysis")

            print(f"\033[94mVibeMusicEngine:\033[0m Waveform data before processing:")
            print(f"\033[94mVibeMusicEngine:\033[0m - Shape: {waveform.shape}")
            print(f"\033[94mVibeMusicEngine:\033[0m - Sample rate: {sample_rate}")

            # Process waveform shape
            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            print(f"\033[94mVibeMusicEngine:\033[0m Waveform data after processing:")
            print(f"\033[94mVibeMusicEngine:\033[0m - Shape: {waveform.shape}")

            # Calculate basic parameters
            audio_samples = waveform.shape[1]
            audio_duration = audio_samples / sample_rate
            frame_long = math.ceil(audio_duration * fps)
            
            print(f"\033[94mVibeMusicEngine:\033[0m Audio parameters:")
            print(f"\033[94mVibeMusicEngine:\033[0m - Number of samples: {audio_samples}")
            print(f"\033[94mVibeMusicEngine:\033[0m - Duration in seconds: {audio_duration:.3f}")
            print(f"\033[94mVibeMusicEngine:\033[0m - Number of frames: {frame_long}")
            print(f"\033[94mVibeMusicEngine:\033[0m - FPS: {fps}")

            # Save temporary audio file for Whisper
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"whisper_input_{uuid.uuid4()}.wav")
            torchaudio.save(temp_path, waveform, sample_rate)

            # Prepare Whisper options based on language settings
            whisper_options = {
                "word_timestamps": True,
                "task": "transcribe"  # domyślnie transkrypcja
            }
            
            # Handle source language
            if source_language != "auto":
                whisper_options["language"] = source_language
                print(f"\033[94mVibeMusicEngine:\033[0m Using specified source language: {source_language}")
            else:
                print("\033[94mVibeMusicEngine:\033[0m Using automatic language detection")

            # Handle target language and translation
            if target_language != "source":
                whisper_options["task"] = "translate"  # zmień task na translate jeśli tłumaczymy
                print(f"\033[94mVibeMusicEngine:\033[0m Translating to English first")
                
                # Load model on the appropriate device
                model = whisper.load_model(choose_model).to(self.device)
                initial_result = model.transcribe(temp_path, **whisper_options)
                
                # Translation to target language if not English
                if target_language != "en":
                    try:
                        from transformers import pipeline
                        translator = pipeline("translation", 
                                           model=f"Helsinki-NLP/opus-mt-en-{target_language}",
                                           device=0 if self.device == "cuda" else -1)
                        
                        # Translate each segment while preserving timestamps
                        for segment in initial_result["segments"]:
                            translated = translator(segment["text"])[0]["translation_text"]
                            segment["text"] = translated
                            
                            if "words" in segment:
                                # Update individual words if available
                                translated_words = translated.split()
                                for i, word in enumerate(segment["words"]):
                                    if i < len(translated_words):
                                        word["word"] = translated_words[i]
                        
                        print(f"\033[94mVibeMusicEngine:\033[0m Translated to: {target_language}")
                        result = initial_result
                    except Exception as e:
                        print(f"\033[93mVibeMusicEngine:\033[0m Translation to {target_language} failed: {str(e)}")
                        print("\033[93mVibeMusicEngine:\033[0m Falling back to English translation")
                        result = initial_result
                else:
                    result = initial_result
            else:
                # Standard transcription
                model = whisper.load_model(choose_model).to(self.device)
                result = model.transcribe(temp_path, **whisper_options)
                print("\033[94mVibeMusicEngine:\033[0m Keeping source language")

            # Log detected language
            if source_language == "auto" and "language" in result:
                detected_language = result["language"]
                print(f"\033[94mVibeMusicEngine:\033[0m Detected source language: {detected_language}")
                if target_language == "source":
                    print(f"\033[94mVibeMusicEngine:\033[0m Using detected language for output")

            segments_alignment = []
            words_alignment = []
            text_lines = []
            frame_starts = []

            # Zbieramy wszystkie słowa ze wszystkich segmentów
            all_words = []
            for segment in result['segments']:
                if "words" in segment:
                    for word in segment["words"]:
                        if word["word"].strip():
                            all_words.append(word)

            # Process words and build lines with context
            current_pos = 0
            while current_pos < len(all_words):
                # Zbierz główne słowa dla linii
                main_words = all_words[current_pos:current_pos + words_per_line]
                if not main_words:
                    break

                # Dodaj kontekst przed
                start_idx = max(0, current_pos - prev_context_words)
                prev_words = all_words[start_idx:current_pos]
                
                # Dodaj kontekst po
                end_idx = min(len(all_words), current_pos + words_per_line + next_context_words)
                next_words = all_words[current_pos + words_per_line:end_idx]

                # Połącz wszystkie części
                line_parts = []
                if prev_words:
                    prev_text = " ".join(w["word"].strip() for w in prev_words)
                    if context_brackets:
                        prev_text = f"[PREV: {prev_text}]"
                    line_parts.append(prev_text)
                
                line_parts.append(" ".join(w["word"].strip() for w in main_words))
                
                if next_words:
                    next_text = " ".join(w["word"].strip() for w in next_words)
                    if context_brackets:
                        next_text = f"[NEXT: {next_text}]"
                    line_parts.append(next_text)

                # Dodaj do wyników
                text_lines.append(" ".join(line_parts))
                frame_start = max(0, int(main_words[0]["start"] * fps))
                frame_starts.append(frame_start)

                # Zaktualizuj indeks
                current_pos += words_per_line

            # Add initial marker if needed
            if not frame_starts or frame_starts[0] > 0:
                frame_starts.insert(0, 0)
                text_lines.insert(0, "<music starts>")

            # Initialize scene arrays
            start_frames = []
            end_frames = []
            scene_texts = []
            current_frame = 0

            # Process segments
            print(f"\033[94mVibeMusicEngine:\033[0m Processing {len(frame_starts)} text segments")
            
            # Process initial gap if exists
            for i in range(len(frame_starts)):
                if current_frame < frame_starts[i]:
                    gap_length = frame_starts[i] - current_frame
                    while gap_length > 0:
                        scene_length = min(gap_length, max_shot_length_frames)
                        scene_length = max(scene_length, min_shot_length_frames)
                        
                        start_frames.append(current_frame)
                        end_frames.append(current_frame + scene_length)
                        scene_texts.append("<SILENCE>")
                        
                        current_frame += scene_length
                        gap_length -= scene_length

                # Process text segment
                if i < len(frame_starts) - 1:
                    scene_total_length = frame_starts[i + 1] - current_frame
                else:
                    scene_total_length = frame_long - current_frame

                remaining_length = scene_total_length
                while remaining_length > 0:
                    scene_length = min(remaining_length, max_shot_length_frames)
                    scene_length = max(scene_length, min_shot_length_frames)
                    
                    if current_frame + scene_length > frame_long:
                        scene_length = frame_long - current_frame
                    
                    if scene_length >= min_shot_length_frames:
                        start_frames.append(current_frame)
                        end_frames.append(current_frame + scene_length)
                        if len(start_frames) == 1 or current_frame == frame_starts[i]:
                            scene_texts.append(text_lines[i])
                        else:
                            scene_texts.append(continuation_text)
                    
                    current_frame += scene_length
                    remaining_length -= scene_length

            # Process final gap if exists
            while current_frame < frame_long:
                scene_length = min(frame_long - current_frame, max_shot_length_frames)
                scene_length = max(scene_length, min_shot_length_frames)
                
                if scene_length >= min_shot_length_frames:
                    start_frames.append(current_frame)
                    end_frames.append(current_frame + scene_length)
                    scene_texts.append("<SILENCE>")
                
                current_frame += scene_length

            # Process beats if enabled
            if beat_detection and beats_waveform is not None:
                print("\033[94mVibeMusicEngine:\033[0m Starting beat analysis...")
                beat_times = self.detect_beats(beats_waveform, beats_sample_rate, 
                                             beat_sensitivity, beat_brutality)
                
                if len(beat_times) > 0:
                    original_frames = start_frames.copy()
                    try:
                        start_frames = self.adjust_cuts_to_beats(
                            start_frames, 
                            beat_times, 
                            fps, 
                            min_shot_length_frames,
                            beat_brutality
                        )
                        
                        end_frames = start_frames[1:] + [frame_long]
                        
                        print(f"\033[94mVibeMusicEngine:\033[0m Beat analysis complete:")
                        print(f"\033[94mVibeMusicEngine:\033[0m - Original cuts: {len(original_frames)}")
                        print(f"\033[94mVibeMusicEngine:\033[0m - Adjusted cuts: {len(start_frames)}")
                    except Exception as e:
                        print(f"\033[91mVibeMusicEngine:\033[0m Beat adjustment failed: {str(e)}")
                        print("\033[91mVibeMusicEngine:\033[0m Using original cuts")
                        start_frames = original_frames
                        end_frames = start_frames[1:] + [frame_long]
                        
            # Calculate final timings
            durations = [end_frames[i] - start_frames[i] for i in range(len(start_frames))]
            start_ms = [str(int(frame / fps * 1000)) for frame in start_frames]
            duration_ms = [str(int(duration / fps * 1000)) for duration in durations]

            # Convert frame numbers to strings
            frame_numbers_str = tuple(str(f) for f in start_frames)
            start_frame_numbers_str = tuple(str(f) for f in start_frames)
            end_frame_numbers_str = tuple(str(f) for f in end_frames)
            durations_str = tuple(str(d) for d in durations)
            
            # Generate frame-based text
            frame_based_lines = [f'"{start_frames[i]}" : "{scene_texts[i]}"' for i in range(len(start_frames))]
            frame_based_text = ",\n".join(frame_based_lines)

            # Create and save EDL file
            edl_content = self.create_edl_content(start_frames, end_frames, fps, filename_prefix, files_in_edl)
            edl_path = self.save_edl_file(edl_content, filename_prefix, save_edl)
            
            # Handle different SRT saving modes
            if save_srt != "no":
                if srt_mode == "all":
                    # Save all versions
                    srt_content_raw = self.create_srt_content(start_frames, end_frames, scene_texts, fps, srt_mode="raw", result=result)
                    srt_content_clean = self.create_srt_content(start_frames, end_frames, scene_texts, fps, srt_mode="clean", result=result)
                    srt_content_word = self.create_srt_content(start_frames, end_frames, scene_texts, fps, srt_mode="word", result=result)
                    srt_content_original = self.create_srt_content(start_frames, end_frames, scene_texts, fps, srt_mode="original", result=result)
                    
                    # Save all versions with the specified save mode
                    self.save_srt_file(srt_content_raw, filename_prefix, save_srt, "raw")
                    self.save_srt_file(srt_content_clean, filename_prefix, save_srt, "clean")
                    self.save_srt_file(srt_content_word, filename_prefix, save_srt, "word")
                    self.save_srt_file(srt_content_original, filename_prefix, save_srt, "original")
                else:
                    # Standard save with selected mode
                    srt_content = self.create_srt_content(start_frames, end_frames, scene_texts, fps, srt_mode=srt_mode, result=result)
                    srt_path = self.save_srt_file(srt_content, filename_prefix, save_srt, srt_mode)

            # Print file saving status
            if save_edl == "no" and save_srt == "no":
                print("\033[93mVibeMusicEngine:\033[0m Both EDL and SRT saving are disabled")
            else:
                saved_files = []
                if edl_path:
                    saved_files.append(f"EDL: {edl_path}")
                if save_srt != "no":
                    saved_files.append(f"SRT files saved with mode: {save_srt}")
                if saved_files:
                    print("\033[94mVibeMusicEngine:\033[0m Files saved:")
                    for file in saved_files:
                        print(f"\033[94mVibeMusicEngine:\033[0m - {file}")

            # Display signature
            print("\n" + "="*50)
            print("\033[95mVibeMusicEngine - by PabloGFX\033[0m")
            print("\033[96mYouTube: https://youtube.com/@lazniak\033[0m")
            print("\033[92mFor support: https://buymeacoffee.com/eyb8tkx3to\033[0m")
            print("="*50 + "\n")

            if self.device == "cuda":
                print(f"\033[94mVibeMusicEngine:\033[0m Final CUDA Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

            return (
                result["text"].strip(),      # Lyrics
                frame_based_text,
                frame_long,
                frame_numbers_str,
                tuple(scene_texts),
                start_frame_numbers_str,
                end_frame_numbers_str,
                len(start_frames),
                durations_str,
                tuple(start_ms),
                tuple(duration_ms)
            )

        except Exception as e:
            print(f"\033[91mVibeMusicEngine:\033[0m Error during audio processing: {str(e)}")
            raise

        finally:
            # Cleanup temporary whisper file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print("\033[94mVibeMusicEngine:\033[0m Temporary whisper file removed")
                except Exception as e:
                    print(f"\033[93mVibeMusicEngine:\033[0m Failed to remove temporary whisper file: {e}")

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def create_srt_content(self, start_frames, end_frames, text_lines, fps, srt_mode="raw", result=None):
        """Creates SRT file content with multiple modes"""
        if srt_mode == "word" and result is not None:
            self.fps = fps  # Zapisz FPS do wykorzystania w innych metodach
            return self.create_word_based_srt(result, fps, self.input_audio)
        elif srt_mode == "original" and result is not None:
            return self.create_original_whisper_srt(result, fps)
            
        srt_content = []
        counter = 1
        
        for i in range(len(start_frames)):
            start_time = self.frames_to_srt_time(start_frames[i], fps)
            end_time = self.frames_to_srt_time(end_frames[i], fps)
            
            text = text_lines[i]
            
            # Skip system messages in clean mode
            if srt_mode == "clean":
                if text in ["<SILENCE>", "<music starts>", "<LOGICAL CONTINUATION OF PREVIOUS VIDEO SHOOT>"]:
                    continue
            
            srt_content.extend([
                str(counter),
                f"{start_time} --> {end_time}",
                text,
                ""
            ])
            counter += 1

        return "\n".join(srt_content)
    
    def create_edl_content(self, start_frames, end_frames, fps, filename_prefix, files_in_edl):
        """Creates EDL file content"""
        edl_content = [
            f"TITLE: {os.path.basename(filename_prefix) if filename_prefix else 'sc'}",
            "FCM: NON-DROP FRAME\n"
        ]

        for i in range(len(start_frames)):
            clip_number = f"{i+1:06d}"
            source_start = "00:00:00:00"
            source_end = self.frames_to_timecode(end_frames[i] - start_frames[i], fps)
            
            record_start = self.frames_to_timecode(start_frames[i], fps)
            record_end = self.frames_to_timecode(end_frames[i], fps)

            edl_content.extend([
                f"{clip_number}  AX       V     C        {source_start} {source_end} {record_start} {record_end} ",
                f"* FROM CLIP NAME: {self.get_clip_name(filename_prefix, i+1, files_in_edl)}\n"
            ])

        return "\n".join(edl_content)

    def get_clip_name(self, filename_prefix, index, files_in_edl):
        """Generates clip name according to schema with specified extension"""
        base_name = os.path.basename(filename_prefix) if filename_prefix else "sc"
        clean_extension = files_in_edl.lstrip('.')
        return f"{base_name}_{index:05d}_.{clean_extension}"

    def frames_to_timecode(self, frames, fps):
        """Converts frame count to timecode format HH:MM:SS:FF"""
        total_seconds = frames / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        remaining_frames = int(frames % fps)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remaining_frames:02d}"

    def frames_to_srt_time(self, frames, fps):
        """Converts frame count to SRT time format (HH:MM:SS,mmm)"""
        total_seconds = frames / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds * 1000) % 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def ensure_output_folder(self, filename_prefix):
        """Creates output folder in ComfyUI main output directory"""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        comfy_dir = current_dir
        while comfy_dir and not os.path.exists(os.path.join(comfy_dir, 'main.py')):
            parent = os.path.dirname(comfy_dir)
            if parent == comfy_dir:
                break
            comfy_dir = parent

        output_dir = os.path.join(comfy_dir, 'output')
        
        if filename_prefix:
            prefix_path = os.path.dirname(filename_prefix)
            if prefix_path:
                output_dir = os.path.join(output_dir, prefix_path)

        os.makedirs(output_dir, exist_ok=True)
        print(f"\033[94mVibeMusicEngine:\033[0m Created output directory: {output_dir}")
        
        return output_dir

    def save_edl_file(self, content, filename_prefix, save_mode):
        """Saves EDL file with different modes"""
        if save_mode == "no":
            return None
            
        output_folder = self.ensure_output_folder(filename_prefix)
        base_name = os.path.basename(filename_prefix) if filename_prefix else "sc"
        
        if save_mode == "iterate":
            counter = 1
            while True:
                edl_path = os.path.join(output_folder, f"{base_name}_{counter:03d}.edl")
                if not os.path.exists(edl_path):
                    break
                counter += 1
        else:  # "overwrite"
            edl_path = os.path.join(output_folder, f"{base_name}.edl")
        
        with open(edl_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\033[94mVibeMusicEngine:\033[0m Saved EDL file: {edl_path}")
        return edl_path

    def analyze_audio_segment(self, segment, threshold=0.002):
        """Analyze audio segment for voice activity"""
        rms = torch.sqrt(torch.mean(segment**2))
        return rms.item() > threshold

    def find_precise_boundary(self, waveform, sample_rate, time_point, direction='forward', window_size=0.005, threshold=0.002):
        """Find precise boundary of speech in audio
        direction: 'forward' to find end, 'backward' to find start"""
        step = window_size if direction == 'forward' else -window_size
        max_steps = int(0.1 / abs(window_size))  # Maximum 100ms search
        
        for i in range(max_steps):
            test_time = time_point + (i * step)
            if test_time < 0 or test_time >= waveform.shape[1] / sample_rate:
                break
                
            start_sample = int(test_time * sample_rate)
            end_sample = int((test_time + window_size) * sample_rate)
            
            if end_sample > waveform.shape[1]:
                break
                
            segment = waveform[:, start_sample:end_sample]
            is_speech = self.analyze_audio_segment(segment, threshold)
            
            if (direction == 'forward' and not is_speech) or (direction == 'backward' and is_speech):
                return test_time
        
        return time_point
        
    @staticmethod
    @numba.jit(nopython=True)
    def calculate_rms(signal, frame_length, hop_length):
        """Calculate RMS energy with Numba acceleration"""
        num_frames = ((len(signal) - frame_length) // hop_length) + 1
        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            frame = signal[start:start + frame_length]
            rms[i] = np.sqrt(np.mean(frame**2))
        return rms

    @staticmethod
    @numba.jit(nopython=True)
    def find_peaks(signal, threshold, min_distance):
        """Find peaks in signal with Numba acceleration"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if len(peaks) == 0 or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return np.array(peaks)

    def detect_beats(self, waveform, sample_rate, sensitivity, brutality):
        """Detects beats using Numba-accelerated peak detection"""
        try:
            # Convert PyTorch tensor to numpy array and ensure mono
            if len(waveform.shape) > 1:
                if waveform.shape[0] == 2:  # Stereo
                    waveform = waveform.mean(dim=0)  # Average channels
                elif waveform.shape[1] == 2:  # Alternative stereo format
                    waveform = waveform.mean(dim=1)
                else:
                    waveform = waveform.squeeze()
            
            # Convert to numpy
            audio_np = waveform.numpy().astype(np.float32)
            
            # Ensure audio is 1D
            if len(audio_np.shape) > 1:
                if audio_np.shape[0] == 2:
                    audio_np = np.mean(audio_np, axis=0)
                elif audio_np.shape[1] == 2:
                    audio_np = np.mean(audio_np, axis=1)
                else:
                    audio_np = audio_np.flatten()

            # Parameters for beat detection
            frame_length = min(2048, len(audio_np) // 4)
            hop_length = min(512, frame_length // 4)
            min_bpm = 60
            max_bpm = 200
            
            if len(audio_np) < frame_length:
                raise ValueError(f"Audio too short: {len(audio_np)} samples < {frame_length} frame length")
            
            # Calculate RMS energy using Numba
            try:
                rms = self.calculate_rms(audio_np, frame_length, hop_length)
            except Exception as e:
                print(f"\033[93mVibeMusicEngine:\033[0m RMS calculation failed: {str(e)}")
                raise
            
            if len(rms) == 0:
                raise ValueError("No RMS values calculated")
            
            threshold_offset = (1.0 - sensitivity) * np.std(rms)
            peak_threshold = np.mean(rms) + threshold_offset
            
            min_samples = max(1, int(60.0 * sample_rate / (max_bpm * hop_length)))
            
            try:
                peak_indices = self.find_peaks(rms, peak_threshold, min_samples)
            except Exception as e:
                print(f"\033[93mVibeMusicEngine:\033[0m Peak finding failed: {str(e)}")
                raise
            
            if len(peak_indices) == 0:
                raise ValueError("No peaks detected")
            
            beat_times = peak_indices * hop_length / sample_rate
            
            min_beat_distance = 60.0 / max_bpm
            filtered_beats = [beat_times[0]]
            for beat in beat_times[1:]:
                if beat - filtered_beats[-1] >= min_beat_distance:
                    filtered_beats.append(beat)
            
            beat_times = np.array(filtered_beats)
            
            if brutality < 1.0 and len(beat_times) > 0:
                jitter = np.random.uniform(-0.1, 0.1, size=len(beat_times)) * (1.0 - brutality)
                beat_times = beat_times + jitter
                beat_times = np.sort(beat_times)
            
            return beat_times
            
        except Exception as e:
            print(f"\033[91mVibeMusicEngine:\033[0m Using time-based beat division: {str(e)}")
            duration = len(audio_np) / sample_rate
            avg_beat_duration = 60.0 / 120.0  # Default 120 BPM
            num_beats = max(int(duration / avg_beat_duration), 1)
            return np.linspace(0, duration, num_beats)

    def adjust_cuts_to_beats(self, original_cuts, beat_times, fps, min_shot_length_frames, brutality):
        """Adjusts cut points to nearest beats"""
        if len(beat_times) == 0:
            return original_cuts

        adjusted_cuts = []
        for cut in original_cuts:
            cut_time = cut / fps
            
            try:
                time_diffs = np.abs(beat_times - cut_time)
                closest_beat_idx = np.argmin(time_diffs)
                closest_beat_time = beat_times[closest_beat_idx]
                
                time_diff = abs(closest_beat_time - cut_time)
                max_adjustment = 0.5 * brutality
                
                if time_diff < max_adjustment:
                    adjusted_frame = int(round(closest_beat_time * fps))
                else:
                    adjusted_frame = cut
                    
                if adjusted_cuts and (adjusted_frame - adjusted_cuts[-1]) < min_shot_length_frames:
                    continue
                    
                adjusted_cuts.append(adjusted_frame)
            except Exception as e:
                print(f"\033[93mVibeMusicEngine:\033[0m Warning: Beat adjustment failed for cut at {cut_time}s: {str(e)}")
                adjusted_cuts.append(cut)
                
        if not adjusted_cuts:
            return original_cuts
            
        return adjusted_cuts

    # Metody do przetwarzania słów
    def create_simple_word_srt(self, result, fps):
        """Simplified fallback method for word-based SRT creation"""
        srt_content = []
        counter = 1
        frame_gap = 1.0 / fps
        
        for segment_idx, segment in enumerate(result['segments']):
            if 'words' not in segment:
                continue
                
            for word_idx, word in enumerate(segment['words']):
                if word['word'].strip():
                    start_time = word['start']
                    end_time = word['end']
                    
                    next_start = None
                    if word_idx + 1 < len(segment['words']):
                        next_start = segment['words'][word_idx + 1]['start']
                    elif segment_idx + 1 < len(result['segments']):
                        if 'words' in result['segments'][segment_idx + 1]:
                            next_words = result['segments'][segment_idx + 1]['words']
                            if next_words:
                                next_start = next_words[0]['start']
                    
                    end_time_with_gap = end_time + frame_gap
                    if next_start is not None and end_time_with_gap > next_start:
                        end_time_with_gap = next_start - frame_gap/2
                    
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time_with_gap * fps)
                    
                    if end_frame <= start_frame + 1:
                        end_frame = start_frame + 2
                        
                    start_time_srt = self.frames_to_srt_time(start_frame, fps)
                    end_time_srt = self.frames_to_srt_time(end_frame, fps)
                    
                    srt_content.extend([
                        str(counter),
                        f"{start_time_srt} --> {end_time_srt}",
                        word['word'].strip(),
                        ""
                    ])
                    counter += 1
        
        return "\n".join(srt_content)

    def verify_segment_with_whisper(self, segment, sample_rate, target_word, model, temp_manager, word_index, stage, original_timing=None, total_words=0):
        """Process audio segment with Whisper preserving original audio levels"""
        temp_path = temp_manager.create_temp_path(word_index, stage)
        try:
            progress_info = f"Word {word_index}" + (f" from {total_words}" if total_words else "")
            print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - {stage}: '{target_word}'")
            if original_timing:
                print(f"\033[94mVibeMusicEngine:\033[0m Timing window: {original_timing['start']:.3f}s - {original_timing['end']:.3f}s")
            
            # Wykrywanie ciszy bez normalizacji
            rms = torch.sqrt(torch.mean(segment**2))
            is_silent = rms < 0.001
            
            if is_silent:
                print(f"\033[93mVibeMusicEngine:\033[0m {progress_info} - Segment appears to be silent (RMS: {rms:.6f})")
                return {
                    'text': '',
                    'similarity': 0.0,
                    'segments': [],
                    'is_silent': True
                }
            
            # Zapisz surowe audio bez normalizacji
            torchaudio.save(temp_path, segment, sample_rate)
            
            # Przetwarzanie przez Whisper
            verification = model.transcribe(
                temp_path,
                word_timestamps=True,
                condition_on_previous_text=False
            )
            
            detected_text = verification['text'].strip().lower()
            similarity = difflib.SequenceMatcher(None, detected_text, target_word.lower()).ratio()
            
            print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Detected: '{detected_text}'")
            print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Similarity: {similarity:.2f}")
            print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Audio RMS: {rms:.6f}")
            
            return {
                'text': detected_text,
                'similarity': similarity,
                'segments': verification.get('segments', []),
                'is_silent': False
            }
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def create_word_based_srt(self, result, fps, audio):
        """Creates word-by-word SRT content with precise boundary detection"""
        try:
            self.temp_manager = self.TempFileManager(folder_paths.get_temp_directory())
            
            # Get waveform and sample rate
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            
            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            print(f"\033[94mVibeMusicEngine:\033[0m Starting word verification process")
            print(f"\033[94mVibeMusicEngine:\033[0m Audio shape: {waveform.shape}")
            print(f"\033[94mVibeMusicEngine:\033[0m Sample rate: {sample_rate}")
            
            # Load Whisper model
            model = whisper.load_model("tiny")
            
            # Count total words for progress tracking
            total_words = sum(1 for segment in result['segments'] 
                            if 'words' in segment 
                            for word in segment['words'] 
                            if word['word'].strip() and not all(c in '.,!?;:' for c in word['word'].strip()))
            
            verified_words = []
            word_index = 0
            
            # Process each segment
            for segment_idx, segment in enumerate(result['segments']):
                if 'words' not in segment:
                    continue
                
                # Process words in segment
                for word_idx, word in enumerate(segment['words']):
                    cleaned_word = word['word'].strip()
                    if not cleaned_word or all(c in '.,!?;:' for c in cleaned_word):
                        continue
                    
                    word_index += 1
                    print(f"\033[94mVibeMusicEngine:\033[0m Processing word {word_index}/{total_words}: '{cleaned_word}'")
                    
                    try:
                        # Find next word start time for gap management
                        next_word_start = None
                        if word_idx + 1 < len(segment['words']):
                            next_word = segment['words'][word_idx + 1]
                            if next_word['word'].strip() and not all(c in '.,!?;:' for c in next_word['word'].strip()):
                                next_word_start = next_word['start']
                        elif segment_idx + 1 < len(result['segments']):
                            next_segment = result['segments'][segment_idx + 1]
                            if 'words' in next_segment and next_segment['words']:
                                next_word = next_segment['words'][0]
                                if next_word['word'].strip() and not all(c in '.,!?;:' for c in next_word['word'].strip()):
                                    next_word_start = next_word['start']
                        
                        start, end, confidence = self.verify_word_timing(
                            waveform,
                            sample_rate,
                            {'word': cleaned_word, 'start': word['start'], 'end': word['end']},
                            model,
                            self.temp_manager,
                            word_index,
                            total_words,
                            next_word_start
                        )
                        
                        verified_words.append({
                            'word': cleaned_word,
                            'start': start,
                            'end': end,
                            'confidence': confidence
                        })
                        
                    except Exception as e:
                        print(f"\033[91mVibeMusicEngine:\033[0m Error processing word '{cleaned_word}': {str(e)}")
                        verified_words.append({
                            'word': cleaned_word,
                            'start': word['start'],
                            'end': word['end'],
                            'confidence': 0
                        })
            
            # Create SRT content
            srt_content = []
            counter = 1
            
            for word_data in verified_words:
                start_frame = int(round(word_data['start'] * fps))
                end_frame = int(round(word_data['end'] * fps))
                
                # Ensure minimum duration (2 frames)
                if end_frame <= start_frame + 1:
                    end_frame = start_frame + 2
                    
                start_time = self.frames_to_srt_time(start_frame, fps)
                end_time = self.frames_to_srt_time(end_frame, fps)
                
                srt_content.extend([
                    str(counter),
                    f"{start_time} --> {end_time}",
                    word_data['word'],
                    ""
                ])
                counter += 1
            
            return "\n".join(srt_content)
            
        except Exception as e:
            print(f"\033[91mVibeMusicEngine:\033[0m Error in word verification: {str(e)}")
            print("\033[93mVibeMusicEngine:\033[0m Falling back to simple processing...")
            return self.create_simple_word_srt(result, fps)
            
        finally:
            if self.temp_manager:
                self.temp_manager.cleanup()

    def verify_word_timing(self, waveform, sample_rate, word_data, model, temp_manager, word_index, total_words, next_word_start=None):
        """Verify and refine word timing with precise boundary detection"""
        progress_info = f"Word {word_index} from {total_words}"
        original_start = word_data['start']
        original_end = word_data['end']
        frame_gap = 1.0 / self.fps

        print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Processing '{word_data['word']}'")
        print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Original timing: {original_start:.3f}s - {original_end:.3f}s")
        
        segment_start = int(original_start * sample_rate)
        segment_end = int(original_end * sample_rate)
        original_segment = waveform[:, segment_start:segment_end]
        original_rms = torch.sqrt(torch.mean(original_segment**2))
        
        if original_rms < 0.001:
            print(f"\033[93mVibeMusicEngine:\033[0m {progress_info} - Segment is silent, using original timing")
            return original_start, original_end + frame_gap, 0.0
        
        refined_start = self.find_precise_boundary(
            waveform, 
            sample_rate, 
            original_start, 
            direction='backward'
        )
        
        refined_end = self.find_precise_boundary(
            waveform, 
            sample_rate, 
            original_end, 
            direction='forward'
        )
        
        final_end = refined_end + frame_gap
        
        if next_word_start is not None and final_end > next_word_start:
            final_end = next_word_start - frame_gap/2
        
        segment = waveform[:, int(refined_start * sample_rate):int(refined_end * sample_rate)]
        verification = self.verify_segment_with_whisper(
            segment,
            sample_rate,
            word_data['word'],
            model,
            temp_manager,
            word_index,
            "refined",
            {'start': refined_start, 'end': refined_end},
            total_words
        )
        
        print(f"\033[94mVibeMusicEngine:\033[0m {progress_info} - Refined timing: {refined_start:.3f}s - {final_end:.3f}s")
        return refined_start, final_end, verification['similarity']
    
# Node registration
NODE_CLASS_MAPPINGS = {
    "VibeMusicEngine": VibeMusicEngine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeMusicEngine": "Vibe Music Engine"
}