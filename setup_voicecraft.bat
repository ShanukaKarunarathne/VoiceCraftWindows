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
