# VoiceCraft - Voice Cloning Application

VoiceCraft is a Streamlit application that allows you to process audio, transcribe it, and clone voices.

## Setup Instructions

@echo off
echo Setting up VoiceCraft environment...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
echo Python not found. Installing Python...
:: Download Python installer
curl -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
:: Install Python with pip and add to PATH
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
:: Delete installer
del python_installer.exe
) else (
echo Python is already installed.
)

:: Create a virtual environment
echo Creating virtual environment...
python -m venv voicecraft_env

:: Activate the virtual environment
echo Activating virtual environment...
call voicecraft_env\Scripts\activate.bat

:: Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install streamlit librosa soundfile noisereduce numpy matplotlib torch torchvision torchaudio whisper pyttsx3 psutil

:: Install F5-TTS
echo Installing F5-TTS...
pip install f5-tts

echo Setup complete! Run VoiceCraft with: run_voicecraft.bat
pause

@echo off
echo Starting VoiceCraft...
call voicecraft_env\Scripts\activate.bat
streamlit run app.py

1. Run `setup_voicecraft.bat` to install Python and all required dependencies
2. Run `create_sample_audio.bat` to create a sample audio file for testing
3. Run `run_voicecraft.bat` to start the application

## Features

- Audio processing with noise reduction
- Audio transcription using OpenAI's Whisper
- Voice cloning using F5-TTS
- Alternative voice synthesis options

## System Requirements

- Windows 10 or 11
- At least 8GB RAM
- 10GB free disk space
- Internet connection (for initial setup)

## Troubleshooting

If you encounter issues with F5-TTS voice cloning:

1. Try using the "Alternative: Simple Voice Synthesis" option
2. Try using the "Windows Native TTS" option
3. Check the debug output for more information

```

## Step 6: Create a Folder Structure

Create a ZIP file with the following structure:

```

VoiceCraft/
├── app.py
├── setup_voicecraft.bat
├── run_voicecraft.bat
├── create_sample_audio.bat
├── README.md
└── data/
└── (empty folder for project data)
