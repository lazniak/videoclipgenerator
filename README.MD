# Vibe Music Engine for ComfyUI

## Overview

Vibe Music Engine is a custom node plugin for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that processes audio input using OpenAI’s Whisper models. It provides transcription, translation, and precise word-level timing—perfect for generating subtitles and edit decision lists (EDL) for video editing workflows. In addition, it features beat detection to synchronize video cuts with the music rhythm.

## Features

- **Audio Transcription & Translation**: Uses Whisper to transcribe audio with support for automatic language detection. When needed, it can also translate the transcription.
- **Subtitle Generation**: Creates SRT files in multiple formats:
  - *Raw*: Directly from Whisper output.
  - *Clean*: Only text content, without system messages.
  - *Word-by-word*: Precise timing for each word.
  - *Original*: The full Whisper transcript.
- **EDL File Creation**: Automatically generates EDL files for seamless integration into video editing software.
- **Beat Detection**: Adjusts video cut points to the nearest beats using customizable sensitivity and brutality settings.
- **Flexible Configuration**: Extensive parameters allow you to control everything from frame rates to subtitle grouping, context words, and file naming.
- **Usage Tracking & Support**: Maintains a usage counter and includes an optional “Buy Me a Coffee” feature to support ongoing development.

## Installation

1. **Download the Node**: Place the `vibe_music_engine.py` file into your ComfyUI custom nodes directory.
2. **Install Dependencies**: Ensure that you have the following Python packages installed:
   - Python 3.7 or higher
   - [PyTorch](https://pytorch.org/)
   - [torchaudio](https://pytorch.org/audio/stable/index.html)
   - [Whisper](https://github.com/openai/whisper)
   - [Numba](http://numba.pydata.org/)
   - [NumPy](https://numpy.org/)
   - [Transformers](https://huggingface.co/docs/transformers)
3. **Restart ComfyUI**: After placing the file in the proper directory, restart ComfyUI to load the new node.

## Usage

Once installed, the Vibe Music Engine node appears under the "VibeMusicEngine" category in the ComfyUI interface. Simply connect an audio input and adjust the settings to fit your needs.

The node processes the audio and returns:
- **Lyrics**: The complete transcribed text.
- **Frame-based Text**: A mapping of text to specific frame numbers.
- **Frame Count**: Total number of frames calculated from the audio duration.
- **Frame Numbers**: Start and end frame numbers for each segment.
- **Subtitle Lines**: Processed subtitle text for each scene.
- **EDL & SRT Files**: Automatically generated files saved to the output directory according to your specified mode (overwrite, iterate, or disabled).

## Parameters

The node provides a rich set of configurable options:

- **Audio Input**: Primary audio for processing.
- **Model Selection**: Choose between various Whisper models (e.g., tiny, base, small, medium, large variants).
- **Language Options**: Specify source and target languages for both transcription and translation.
- **Frame Rate (FPS)**: Define the video frame rate (e.g., 25.0 fps) to calculate timecodes.
- **Subtitle Settings**: Options include words per line, context words before/after, and whether to use context brackets.
- **EDL Settings**: Configure file naming and save modes for EDL generation.
- **Beat Detection**: Enable beat detection with sensitivity and alignment (brutality) settings.
- **Miscellaneous**: Additional options like a donation prompt (Buy Me a Coffee) and a usage counter.

## Example

Below is an illustrative example of how the node might be used in a ComfyUI flow:

```plaintext
[Audio Input] --> [Vibe Music Engine Node]
                      |
                      +--> Transcribed Lyrics
                      +--> Frame-based Mapping
                      +--> Generated SRT/EDL Files
