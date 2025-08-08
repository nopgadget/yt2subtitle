# YouTube to Subtitles Converter

A powerful tool that downloads YouTube videos, converts them to MP3, and generates high-quality subtitles using AI-powered speech recognition with chunking support for long videos.

## 🚀 Features

- **YouTube Video Download**: Downloads videos and converts to MP3 format
- **Playlist Support**: Download entire playlists with the `--playlist` flag
- **GPU-Accelerated Transcription**: Uses NVIDIA GPU for fast AI transcription
- **Chunked Processing**: Handles videos of any length with 30-second chunks
- **Progress Tracking**: Real-time progress bars for all operations
- **Quality Subtitles**: Uses Whisper medium model for accurate transcriptions
- **Smart File Management**: Skips existing files to avoid re-downloading
- **Automatic Tool Setup**: Downloads required tools (yt-dlp, ffmpeg) automatically

## 📋 Requirements

### System Requirements
- **Windows 10/11** (tested on Windows 10)
- **NVIDIA GPU** with CUDA support (recommended for fast transcription)
- **Python 3.8+** (tested with Python 3.13)
- **8GB+ RAM** (for AI model loading)
- **Internet connection** for downloading videos and models

### GPU Requirements (Recommended)
- **NVIDIA GPU** with CUDA 12.9+ support
- **8GB+ VRAM** (for Whisper medium model)
- **CUDA drivers** installed

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd yt2subtitle

# Or download and extract the ZIP file
```

### 2. Set Up Python Environment

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n subtitles python=3.13
conda activate subtitles

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using pip
```bash
# Create a virtual environment
python -m venv subtitles_env
subtitles_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Setup Script (Optional)
```bash
# The setup script will install dependencies and verify your system
python setup.py
```

### 4. Verify GPU Setup
```bash
# Test CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🎯 Usage

### Basic Usage
```bash
# Download video and generate subtitles
python youtube_to_subtitles.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage
```bash
# Download video only (skip transcription)
python youtube_to_subtitles.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --no-transcribe

# Download entire playlist and generate subtitles for all videos
python youtube_to_subtitles.py --url "https://www.youtube.com/playlist?list=PLAYLIST_ID" --playlist

# Download playlist without transcription
python youtube_to_subtitles.py --url "https://www.youtube.com/playlist?list=PLAYLIST_ID" --playlist --no-transcribe

# Example with a real video
python youtube_to_subtitles.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## 📁 Output Files

The script creates files in the `downloaded/` directory:

- **`Video Title.mp3`** - Audio file extracted from YouTube video
- **`Video Title.txt`** - Subtitle file with transcribed text

### Example Output (Single Video)
```
downloaded/
├── Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3
└── Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).txt
```

### Example Output (Playlist)
```
downloaded/
├── Video 1 Title.mp3
├── Video 1 Title.txt
├── Video 2 Title.mp3
├── Video 2 Title.txt
├── Video 3 Title.mp3
└── Video 3 Title.txt
```

## 🔧 How It Works

### 1. Tool Setup
- **yt-dlp**: Downloads YouTube videos
- **ffmpeg**: Converts video to MP3 format
- **Whisper**: AI model for speech-to-text transcription

### 2. Processing Pipeline
1. **URL Cleaning**: Removes problematic query parameters
2. **Video Download**: Downloads and converts to MP3
3. **Audio Chunking**: Splits long audio into 30-second segments
4. **AI Transcription**: Uses Whisper model with GPU acceleration
5. **Chunk Combination**: Merges all transcribed chunks
6. **File Saving**: Creates subtitle file with matching name

### 3. Chunking System
- **Chunk Duration**: 30 seconds per chunk
- **Overlap**: 2-second overlap between chunks for continuity
- **Processing**: Each chunk processed individually
- **Combination**: All chunks merged into final transcription

### 4. GPU Acceleration
- Uses NVIDIA GPU for fast transcription
- Automatically detects and uses available GPU
- Falls back to CPU if GPU unavailable

## ⚙️ Configuration

### Model Settings
The script uses a configurable Whisper model. Edit `WHISPER_MODEL` in the script:

```python
# Options: "base", "medium", "large"
WHISPER_MODEL = "medium"  # Change this to your preferred model
```

**Available Models:**
- **base**: `openai/whisper-base` - Fast, lower quality (~1GB VRAM)
- **medium**: `openai/whisper-medium` - Balanced speed/quality (~3GB VRAM) - **Default**
- **large**: `openai/whisper-large-v3` - Highest quality, slower (~10GB VRAM)

**Model Parameters:**
- **Language**: English (forced)
- **Task**: Transcription
- **Max Length**: 448 tokens (configurable per model)
- **Decoding**: Greedy (fast)

### Chunking Settings
- **Chunk Duration**: 30 seconds
- **Overlap**: 2 seconds between chunks
- **Minimum Chunk**: 5 seconds (skips shorter chunks)

### GPU Settings
- **Device**: CUDA (if available)
- **Precision**: Float16 (for memory efficiency)
- **Memory**: Auto-mapping for optimal usage

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```bash
# Check if CUDA is properly installed
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Install CUDA drivers and PyTorch with CUDA support

#### 2. Missing Dependencies
```bash
# Install missing audio libraries
pip install librosa soundfile audioread
```

#### 3. Out of Memory Errors
```bash
# Reduce model size by changing WHISPER_MODEL in the script
WHISPER_MODEL = "base"  # Use base model instead of medium
# Or use CPU if GPU memory is insufficient
```

#### 4. Download Failures
```bash
# Check internet connection
# Verify YouTube URL is accessible
# Try different video (some may be restricted)
```

### Performance Optimization

#### For Faster Processing
- Use `whisper-base` instead of `whisper-medium`
- Ensure GPU has sufficient VRAM
- Close other GPU-intensive applications

#### For Better Quality
- Use `whisper-large` model (requires more VRAM)
- Increase `max_length` parameter
- Use beam search instead of greedy decoding

## 📊 Performance

### Typical Processing Times
- **Video Download**: 1-5 minutes (depends on video size)
- **Model Loading**: 30-60 seconds (first run only)
- **Transcription**: 5-10 seconds per 30-second chunk
- **GPU vs CPU**: 5-10x faster with GPU

### Memory Usage
- **Whisper Medium**: ~3GB VRAM
- **Whisper Base**: ~1GB VRAM
- **System RAM**: 4-8GB recommended

## 🔒 Legal Considerations

- **YouTube Terms of Service**: Respect YouTube's terms of service
- **Copyright**: Only download content you have permission to use
- **Fair Use**: Ensure your use complies with copyright laws
- **Personal Use**: This tool is for personal/educational use only

## 📄 License

This project is for educational and personal use only. Please respect all applicable laws and terms of service.

## 🙏 Acknowledgments

- **yt-dlp**: YouTube video downloading
- **ffmpeg**: Audio/video processing
- **OpenAI Whisper**: Speech recognition model
- **Hugging Face**: Model hosting and transformers library
- **PyTorch**: Deep learning framework
- **librosa**: Audio processing

---

**Happy transcribing! 🎵📝**