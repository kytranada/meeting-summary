# Python Script to Transcribe an Audio File using Whisper

# This script transcribes an audio file using OpenAI Whisper and saves the result to a text file.

import speech_recognition as sr
import os
from datetime import datetime
import tempfile
from pydub import AudioSegment

# Configuration
WHISPER_MODEL = "base.en"  # Default model
LANGUAGE = "english"       # Language for transcription

def save_text_to_file(text_to_save, original_audio_filename):
    """Saves transcription to a file named after the audio file."""
    try:
        base, ext = os.path.splitext(original_audio_filename)
        output_filename = f"{base}_transcription.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"# Transcription of: {os.path.basename(original_audio_filename)}\n")
            f.write(f"# Model used: {WHISPER_MODEL}\n")
            f.write(f"# Language: {LANGUAGE if LANGUAGE else 'auto-detected'}\n")
            f.write(f"# Transcribed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n\n")
            f.write(text_to_save)
        print(f"Transcription saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving transcription: {e}")

def convert_to_wav(audio_filepath):
    """Convert audio to WAV format if it's not already."""
    filename, file_extension = os.path.splitext(audio_filepath)
    
    # If already WAV format, return the original path
    if file_extension.lower() == '.wav':
        return audio_filepath
    
    print(f"Converting {file_extension} file to WAV format...")
    try:
        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # Convert using pydub
        audio = AudioSegment.from_file(audio_filepath)
        audio.export(temp_wav.name, format="wav")
        print("Conversion successful")
        return temp_wav.name
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def transcribe_audio_file(audio_filepath):
    """Transcribes an audio file using Whisper."""
    if not os.path.exists(audio_filepath):
        print(f"Error: File not found at '{audio_filepath}'")
        return

    # Convert to WAV if needed
    wav_filepath = convert_to_wav(audio_filepath)
    if not wav_filepath:
        print("Failed to prepare audio for transcription.")
        return
    
    temp_file_created = wav_filepath != audio_filepath
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_filepath) as source:
            print(f"Loading audio file: {os.path.basename(audio_filepath)}...")
            try:
                audio_data = recognizer.record(source)
                print("Audio loaded successfully.")
            except Exception as e:
                print(f"Error loading audio: {e}")
                return

        print(f"Transcribing with Whisper (Model: {WHISPER_MODEL})...")
        try:
            text = recognizer.recognize_whisper(
                audio_data,
                model=WHISPER_MODEL,
                language=LANGUAGE if WHISPER_MODEL.endswith(".en") or LANGUAGE else None
            )
            print("\n--- Transcription Complete ---")
            print(text[:1000] + "..." if len(text) > 1000 else text)
            save_text_to_file(text, audio_filepath)
        except sr.UnknownValueError:
            print("Whisper could not understand the audio.")
        except sr.RequestError as e:
            print(f"Whisper request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    finally:
        # Clean up temp file if one was created
        if temp_file_created and os.path.exists(wav_filepath):
            try:
                os.remove(wav_filepath)
                print("Temporary files cleaned up.")
            except:
                pass

if __name__ == "__main__":
    print("Audio File Transcription Tool")
    audio_file_to_transcribe = input("Enter the full path to the audio file: ").strip()
    if audio_file_to_transcribe.startswith(('"', "'")) and audio_file_to_transcribe.endswith(('"', "'")):
        audio_file_to_transcribe = audio_file_to_transcribe[1:-1]
    transcribe_audio_file(audio_file_to_transcribe)
