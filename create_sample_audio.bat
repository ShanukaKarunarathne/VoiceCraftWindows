@echo off
echo Creating sample audio file for testing...
call voicecraft_env\Scripts\activate.bat

python -c "
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.save_to_file('This is a sample audio file for testing voice cloning with VoiceCraft. The quick brown fox jumps over the lazy dog.', 'sample_audio.wav')
engine.runAndWait()
print('Sample audio file created: sample_audio.wav')
"

echo Done! You can use sample_audio.wav for testing.
pause
