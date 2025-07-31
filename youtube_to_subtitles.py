#!/usr/bin/env python3
"""
YouTube to Subtitles Converter
Downloads YouTube videos, converts them to MP3, and generates subtitles using AI.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Whisper model configuration
# Options: "base", "medium", "large"
# - base: ~1GB VRAM, faster, lower quality
# - medium: ~3GB VRAM, balanced speed/quality (default)
# - large: ~10GB VRAM, slower, highest quality
WHISPER_MODEL = "medium"

# Model configurations
MODEL_CONFIGS = {
    "base": {
        "model_name": "openai/whisper-base",
        "max_length": 448,
        "description": "Fast, lower quality (~1GB VRAM)"
    },
    "medium": {
        "model_name": "openai/whisper-medium", 
        "max_length": 448,
        "description": "Balanced speed/quality (~3GB VRAM)"
    },
    "large": {
        "model_name": "openai/whisper-large-v3",
        "max_length": 448,
        "description": "Highest quality, slower (~10GB VRAM)"
    }
}

import os
import sys
import zipfile
import urllib.request
import argparse
import subprocess
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import time

def download_file(url, filename):
    """Download a file from URL to filename."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully!")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} successfully!")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def check_and_download_ytdlp():
    """Check if yt-dlp.exe exists, download if not."""
    ytdlp_path = Path("tools/yt-dlp.exe")
    
    if ytdlp_path.exists():
        print("yt-dlp.exe found in tools folder.")
        return True
    
    print("yt-dlp.exe not found. Downloading...")
    
    # Create tools directory if it doesn't exist
    Path("tools").mkdir(exist_ok=True)
    
    # Download yt-dlp.exe from the latest release
    ytdlp_url = "https://github.com/yt-dlp/yt-dlp/releases/download/2025.07.21/yt-dlp.exe"
    
    if download_file(ytdlp_url, str(ytdlp_path)):
        return True
    else:
        print("Failed to download yt-dlp.exe")
        return False

def check_and_download_ffmpeg():
    """Check if ffmpeg exists, download and extract if not."""
    ffmpeg_dir = Path("tools/ffmpeg-master-latest-win64-gpl-shared")
    ffmpeg_bin = ffmpeg_dir / "bin" / "ffmpeg.exe"
    
    if ffmpeg_bin.exists():
        print("ffmpeg found in tools folder.")
        return True
    
    print("ffmpeg not found. Downloading...")
    
    # Create tools directory if it doesn't exist
    Path("tools").mkdir(exist_ok=True)
    
    # Download ffmpeg zip
    ffmpeg_url = "https://github.com/btbn/ffmpeg-builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
    zip_path = Path("tools/ffmpeg-master-latest-win64-gpl-shared.zip")
    
    if download_file(ffmpeg_url, str(zip_path)):
        # Extract the zip file
        if extract_zip(str(zip_path), "tools"):
            # Clean up the zip file
            zip_path.unlink()
            return True
    
    print("Failed to download or extract ffmpeg")
    return False

def clean_youtube_url(url):
    """Remove problematic query parameters from YouTube URL."""
    import urllib.parse
    
    # Parse the URL
    parsed = urllib.parse.urlparse(url)
    
    # Parse query parameters
    query_params = urllib.parse.parse_qs(parsed.query)
    
    # Remove problematic parameters
    params_to_remove = ['list', 'start_radio', 'index', 'feature']
    for param in params_to_remove:
        if param in query_params:
            del query_params[param]
    
    # Rebuild query string
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    
    # Reconstruct URL
    clean_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    
    return clean_url

def convert_youtube_to_mp3(youtube_url):
    """Convert YouTube video to MP3 using yt-dlp and ffmpeg."""
    # Clean the URL first
    clean_url = clean_youtube_url(youtube_url)
    print(f"Cleaned URL: {clean_url}")
    
    ytdlp_path = Path("tools/yt-dlp.exe")
    ffmpeg_path = Path("tools/ffmpeg-master-latest-win64-gpl-shared/bin")
    
    if not ytdlp_path.exists():
        print("Error: yt-dlp.exe not found. Please run the script to download it first.")
        return False
    
    if not ffmpeg_path.exists():
        print("Error: ffmpeg not found. Please run the script to download it first.")
        return False
    
    # Create downloaded directory if it doesn't exist
    downloaded_dir = Path("downloaded")
    downloaded_dir.mkdir(exist_ok=True)
    print(f"Output directory: {downloaded_dir.absolute()}")
    
    # First, get the video title to check if MP3 already exists
    print("Checking if file already exists...")
    try:
        # Get video info to extract title
        info_cmd = [
            str(ytdlp_path),
            "--get-title",
            clean_url
        ]
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        video_title = result.stdout.strip()
        
        # Check if MP3 file already exists
        mp3_filename = f"{video_title}.mp3"
        mp3_path = downloaded_dir / mp3_filename
        
        if mp3_path.exists():
            print(f"MP3 file already exists: {mp3_path}")
            print("Skipping download - file is already available.")
            return str(mp3_path)
            
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not check if file exists: {e}")
        print("Proceeding with download...")
    
    # Build the command
    cmd = [
        str(ytdlp_path),
        "--ffmpeg-location", str(ffmpeg_path),
        "-o", str(downloaded_dir / "%(title)s.%(ext)s"),  # Output template
        "-x",  # Extract audio
        "--audio-format", "mp3",  # Convert to MP3
        "--audio-quality", "0",  # Best quality
        clean_url
    ]
    
    print(f"Converting {youtube_url} to MP3...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Conversion completed successfully!")
        print(result.stdout)
        
        # Find the created MP3 file
        mp3_files = list(downloaded_dir.glob("*.mp3"))
        if mp3_files:
            # Get the most recently created MP3 file
            latest_mp3 = max(mp3_files, key=lambda x: x.stat().st_mtime)
            return str(latest_mp3)
        else:
            print("Warning: Could not find created MP3 file")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"Error output: {e.stderr}")
        return None

def transcribe_audio(audio_file_path, output_dir="downloaded"):
    """
    Transcribe an audio file using Whisper model with chunking for longer videos
    """
    # Check if CUDA is available and force GPU usage
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Set device explicitly
        torch.cuda.set_device(0)
    else:
        device = "cpu"
        print("⚠️  CUDA not available, using CPU (this will be much slower)")
    
    # Get model configuration
    if WHISPER_MODEL not in MODEL_CONFIGS:
        print(f"❌ Invalid model '{WHISPER_MODEL}'. Available options: {list(MODEL_CONFIGS.keys())}")
        return None
    
    model_config = MODEL_CONFIGS[WHISPER_MODEL]
    model_name = model_config["model_name"]
    max_length = model_config["max_length"]
    
    print(f"🤖 Using Whisper {WHISPER_MODEL} model: {model_config['description']}")
    print("Loading Whisper model and processor...")
    with tqdm(total=2, desc="Loading model", unit="step") as pbar:
        processor = WhisperProcessor.from_pretrained(model_name)
        pbar.update(1)
        
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else device
        )
        pbar.update(1)
    
    print(f"Transcribing: {audio_file_path}")
    
    # Load audio using librosa
    import librosa
    print("Loading audio file...")
    audio, sr = librosa.load(audio_file_path, sr=16000)
    
    # Calculate audio duration
    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Define chunk parameters
    chunk_duration = 30  # 30 seconds per chunk
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(2 * sr)  # 2 second overlap between chunks
    
    # Calculate number of chunks
    num_chunks = max(1, int(len(audio) / (chunk_samples - overlap_samples)))
    print(f"Processing {num_chunks} chunks of {chunk_duration} seconds each")
    
    # Process chunks
    transcriptions = []
    
    with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for i in range(num_chunks):
            # Calculate chunk boundaries
            start_sample = i * (chunk_samples - overlap_samples)
            end_sample = min(start_sample + chunk_samples, len(audio))
            
            # Extract chunk
            chunk = audio[start_sample:end_sample]
            
            # Skip very short chunks
            if len(chunk) < sr * 5:  # Skip chunks shorter than 5 seconds
                continue
            
            # Process chunk
            start_time = time.time()
            
            # Process audio chunk
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device, dtype=model.dtype)
            
            # Generate transcription for this chunk
            predicted_ids = model.generate(
                input_features,
                language="en",
                task="transcribe",
                max_length=max_length,
                num_beams=1,
                do_sample=False
            )
            
            # Decode the transcription
            chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Add chunk info for debugging
            chunk_start_time = start_sample / sr
            chunk_end_time = end_sample / sr
            print(f"\nChunk {i+1}/{num_chunks} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s): {chunk_transcription}")
            
            transcriptions.append(chunk_transcription)
            
            elapsed_time = time.time() - start_time
            print(f"⏱️  Chunk {i+1} completed in {elapsed_time:.1f} seconds")
            pbar.update(1)
    
    # Combine all transcriptions
    full_transcription = " ".join(transcriptions)
    
    # Create output filename (same name as video but .txt extension)
    audio_path = Path(audio_file_path)
    output_filename = audio_path.stem + ".txt"
    output_path = Path(output_dir) / output_filename
    
    # Save transcription to file
    print(f"\nSaving complete transcription...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription)
    
    print(f"Transcription saved to: {output_path}")
    print("\nComplete Transcription:")
    print("=" * 50)
    print(full_transcription)
    print("=" * 50)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos, convert to MP3, and generate subtitles")
    parser.add_argument("-url", "--url", required=True, help="YouTube URL to download and convert")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription step")
    
    args = parser.parse_args()
    
    print("YouTube to Subtitles Converter")
    print("=" * 40)
    
    # Check and download required tools
    print("\n1. Checking for yt-dlp...")
    if not check_and_download_ytdlp():
        print("Failed to set up yt-dlp. Exiting.")
        sys.exit(1)
    
    print("\n2. Checking for ffmpeg...")
    if not check_and_download_ffmpeg():
        print("Failed to set up ffmpeg. Exiting.")
        sys.exit(1)
    
    print("\n3. Converting YouTube video to MP3...")
    mp3_file = convert_youtube_to_mp3(args.url)
    if not mp3_file:
        print("Failed to convert video. Exiting.")
        sys.exit(1)
    
    print(f"✅ MP3 file created: {mp3_file}")
    
    # Transcribe the audio if not skipped
    subtitle_file = None
    if not args.no_transcribe:
        print("\n4. Generating subtitles...")
        try:
            subtitle_file = transcribe_audio(mp3_file)
            print(f"✅ Subtitle file created: {subtitle_file}")
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without transcription...")
    else:
        print("\n4. Skipping transcription as requested.")
    
    print("\n🎉 Process completed successfully!")
    print(f"📁 MP3 file: {mp3_file}")
    if not args.no_transcribe and subtitle_file:
        print(f"📄 Subtitle file: {subtitle_file}")

if __name__ == "__main__":
    main() 