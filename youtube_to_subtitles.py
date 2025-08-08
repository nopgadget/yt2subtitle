#!/usr/bin/env python3
"""
YouTube to Subtitles Converter
Downloads YouTube videos, converts them to MP3, and generates subtitles using AI.
Also supports direct transcription of local MP3 files.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Whisper model configuration
# Options: "base", "medium", "large"
# - base: ~1GB VRAM, faster, lower quality
# - medium: ~3GB VRAM, balanced speed/quality (default)
# - large: ~10GB VRAM, slower, highest quality
WHISPER_MODEL = "large"

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

def clean_youtube_url(url, playlist=False):
    """Remove problematic query parameters from YouTube URL."""
    import urllib.parse
    
    # Parse the URL
    parsed = urllib.parse.urlparse(url)
    
    # Parse query parameters
    query_params = urllib.parse.parse_qs(parsed.query)
    
    # Remove problematic parameters
    params_to_remove = ['start_radio', 'index', 'feature']
    
    # Only remove playlist-related parameters if not downloading a playlist
    if not playlist:
        params_to_remove.extend(['list'])
    
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

def convert_youtube_to_mp3(youtube_url, playlist=False):
    """Convert YouTube video to MP3 using yt-dlp and ffmpeg."""
    # Clean the URL first
    clean_url = clean_youtube_url(youtube_url, playlist=playlist)
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
    
    # For playlists, skip the individual file existence check since we want to download all videos
    if not playlist:
        # First, get the video title to check if MP3 already exists (only for single videos)
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
    
    # Add playlist-specific options if downloading a playlist
    if playlist:
        cmd.extend(["--yes-playlist"])  # Force playlist download
        print("🎵 Downloading playlist...")
    else:
        cmd.extend(["--no-playlist"])  # Don't download playlists
    
    print(f"Converting {youtube_url} to MP3...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Conversion completed successfully!")
        if result.stdout:
            print(result.stdout)
        
        # Find the created MP3 file(s)
        mp3_files = list(downloaded_dir.glob("*.mp3"))
        if mp3_files:
            if playlist:
                # For playlists, return a list of all created MP3 files
                # Sort by modification time to get them in download order
                mp3_files.sort(key=lambda x: x.stat().st_mtime)
                return [str(mp3_file) for mp3_file in mp3_files]
            else:
                # For single videos, return the most recently created MP3 file
                latest_mp3 = max(mp3_files, key=lambda x: x.stat().st_mtime)
                return str(latest_mp3)
        else:
            print("Warning: Could not find created MP3 file")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def transcribe_audio(audio_file_path, output_dir="downloaded", start_chunk=None, end_chunk=None, start_time=None, end_time=None, include_timestamps=False):
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
    
    # Determine chunk range based on arguments
    start_chunk_idx = 0
    end_chunk_idx = num_chunks
    
    if start_chunk is not None:
        start_chunk_idx = max(0, min(start_chunk, num_chunks - 1))
        print(f"Starting from chunk {start_chunk_idx + 1}")
    
    if end_chunk is not None:
        end_chunk_idx = min(end_chunk, num_chunks)
        print(f"Ending at chunk {end_chunk_idx}")
    
    # Convert time ranges to chunk ranges if specified
    if start_time is not None:
        start_sample = int(start_time * sr)
        start_chunk_idx = max(0, start_sample // (chunk_samples - overlap_samples))
        print(f"Start time {start_time:.1f}s corresponds to chunk {start_chunk_idx + 1}")
    
    if end_time is not None:
        end_sample = int(end_time * sr)
        end_chunk_idx = min(num_chunks, end_sample // (chunk_samples - overlap_samples) + 1)
        print(f"End time {end_time:.1f}s corresponds to chunk {end_chunk_idx}")
    
    # Validate chunk range
    if start_chunk_idx >= end_chunk_idx:
        print("❌ Invalid chunk range: start chunk must be less than end chunk")
        return None
    
    chunks_to_process = end_chunk_idx - start_chunk_idx
    print(f"Will process chunks {start_chunk_idx + 1} to {end_chunk_idx} ({chunks_to_process} chunks)")
    
    # Process chunks
    transcriptions = []
    
    with tqdm(total=chunks_to_process, desc="Processing chunks", unit="chunk") as pbar:
        for i in range(start_chunk_idx, end_chunk_idx):
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
            
            # Store transcription with timestamp info if requested
            if include_timestamps:
                # Format timestamp as MM:SS
                start_minutes = int(chunk_start_time // 60)
                start_seconds = int(chunk_start_time % 60)
                end_minutes = int(chunk_end_time // 60)
                end_seconds = int(chunk_end_time % 60)
                
                timestamped_transcription = f"[{start_minutes:02d}:{start_seconds:02d}-{end_minutes:02d}:{end_seconds:02d}] [Chunk {i+1}] {chunk_transcription}"
                transcriptions.append(timestamped_transcription)
            else:
                transcriptions.append(chunk_transcription)
            
            elapsed_time = time.time() - start_time
            print(f"⏱️  Chunk {i+1} completed in {elapsed_time:.1f} seconds")
            pbar.update(1)
    
    # Combine all transcriptions
    if include_timestamps:
        full_transcription = "\n".join(transcriptions)
    else:
        full_transcription = " ".join(transcriptions)
    
    # Create output filename (same name as video but .txt extension)
    audio_path = Path(audio_file_path)
    output_filename = audio_path.stem + ".txt"
    
    # Add range info to filename if processing a subset
    if start_chunk_idx > 0 or end_chunk_idx < num_chunks:
        if start_time is not None and end_time is not None:
            output_filename = f"{audio_path.stem}_t{start_time:.0f}-{end_time:.0f}s.txt"
        elif start_chunk is not None and end_chunk is not None:
            output_filename = f"{audio_path.stem}_c{start_chunk}-{end_chunk}.txt"
        else:
            output_filename = f"{audio_path.stem}_range.txt"
    
    # Add timestamp indicator to filename if timestamps are included
    if include_timestamps:
        output_filename = output_filename.replace(".txt", "_timestamps.txt")
    
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
    parser.add_argument("-url", "--url", help="YouTube URL to download and convert")
    parser.add_argument("-file", "--file", help="Local MP3 file to transcribe")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription step")
    parser.add_argument("--playlist", action="store_true", help="Download entire playlist (if URL is a playlist)")
    parser.add_argument("--start-chunk", type=int, help="Start processing from this chunk (0-based)")
    parser.add_argument("--end-chunk", type=int, help="End processing at this chunk (0-based)")
    parser.add_argument("--start-time", type=float, help="Start transcribing from this time (seconds)")
    parser.add_argument("--end-time", type=float, help="End transcribing at this time (seconds)")
    parser.add_argument("--include-timestamps", action="store_true", help="Include timestamps in the output subtitle file")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.url and not args.file:
        parser.error("Either --url or --file must be specified")
    if args.url and args.file:
        parser.error("Cannot specify both --url and --file")
    if args.playlist and args.file:
        parser.error("--playlist option is only valid with --url")
    
    print("YouTube to Subtitles Converter")
    print("=" * 40)
    
    mp3_files = []
    
    if args.url:
        # YouTube URL processing
        print("\n1. Checking for yt-dlp...")
        if not check_and_download_ytdlp():
            print("Failed to set up yt-dlp. Exiting.")
            sys.exit(1)
        
        print("\n2. Checking for ffmpeg...")
        if not check_and_download_ffmpeg():
            print("Failed to set up ffmpeg. Exiting.")
            sys.exit(1)
        
        print("\n3. Converting YouTube video to MP3...")
        if args.playlist:
            print("🎵 Playlist mode enabled - will download all videos in the playlist")
        
        mp3_files = convert_youtube_to_mp3(args.url, playlist=args.playlist)
        if not mp3_files:
            print("Failed to convert video. Exiting.")
            sys.exit(1)
        
        # Handle single file vs playlist
        if args.playlist and isinstance(mp3_files, list):
            print(f"✅ {len(mp3_files)} MP3 files created from playlist:")
            for i, mp3_file in enumerate(mp3_files, 1):
                print(f"   {i:2d}. {Path(mp3_file).name}")
        else:
            print(f"✅ MP3 file created: {Path(mp3_files).name}")
            mp3_files = [mp3_files]  # Convert to list for consistent processing
    
    else:
        # Local MP3 file processing
        mp3_path = Path(args.file)
        if not mp3_path.exists():
            print(f"Error: MP3 file not found: {mp3_path}")
            sys.exit(1)
        
        if not mp3_path.suffix.lower() == '.mp3':
            print(f"Warning: File {mp3_path} doesn't have .mp3 extension")
        
        print(f"\n3. Using local MP3 file: {mp3_path.name}")
        mp3_files = [str(mp3_path)]
        print(f"✅ MP3 file found: {mp3_path.name}")
    
    # Transcribe the audio if not skipped
    subtitle_files = []
    if not args.no_transcribe:
        print("\n4. Generating subtitles...")
        
        # Show range information if specified
        if args.start_chunk is not None or args.end_chunk is not None:
            print(f"📊 Chunk range: {args.start_chunk or 0} to {args.end_chunk or 'end'}")
        if args.start_time is not None or args.end_time is not None:
            print(f"⏰ Time range: {args.start_time or 0:.1f}s to {args.end_time or 'end':.1f}s")
        
        for i, mp3_file in enumerate(mp3_files, 1):
            try:
                print(f"\n📝 Transcribing file {i}/{len(mp3_files)}: {Path(mp3_file).name}")
                subtitle_file = transcribe_audio(mp3_file, start_chunk=args.start_chunk, end_chunk=args.end_chunk, start_time=args.start_time, end_time=args.end_time, include_timestamps=args.include_timestamps)
                if subtitle_file:
                    subtitle_files.append(subtitle_file)
                    print(f"✅ Subtitle file created: {Path(subtitle_file).name}")
            except Exception as e:
                print(f"❌ Error during transcription of {Path(mp3_file).name}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without transcription for this file...")
    else:
        print("\n4. Skipping transcription as requested.")
    
    print("\n🎉 Process completed successfully!")
    if args.url and args.playlist:
        print(f"📁 {len(mp3_files)} MP3 files downloaded from playlist")
        if not args.no_transcribe and subtitle_files:
            print(f"📄 {len(subtitle_files)} subtitle files created")
    else:
        print(f"📁 MP3 file: {Path(mp3_files[0]).name}")
        if not args.no_transcribe and subtitle_files:
            print(f"📄 Subtitle file: {Path(subtitle_files[0]).name}")

if __name__ == "__main__":
    main() 