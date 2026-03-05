import os
import subprocess
import torch
import whisper
from TTS.api import TTS
from deep_translator import GoogleTranslator

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def process_dubbing(video_in, target_lang="hi"):
    # 1. Extraction & Transcription (Whisper Medium for Code-Switching)
    print("Transcribing and Translating Audio...")
    model = whisper.load_model("medium")
    result = model.transcribe(video_in, task="translate", fp16=False)
    english_text = result["text"]
    hindi_text = GoogleTranslator(source='en', target='hi').translate(english_text)
    
    # 2. Voice Cloning (Coqui XTTS v2)
    print("Generating Voice Clone...")
    # (Note: Requires the PyTorch 2.6 weights_only=False patch in production)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts.tts_to_file(text=hindi_text, speaker_wav="reference_audio.wav", 
                    language=target_lang, file_path="hindi_raw.wav")

    # 3. Spatial Isolation & Lip-Sync (The 'Resourceful' Trick)
    print("Isolating Speaker and Running Wav2Lip...")
    # Crop right half to ignore TV screen interference
    run_command("ffmpeg -y -i clip_15s.mp4 -filter:v 'crop=iw/2:ih:iw/2:0' right_half.mp4")
    
    # Run Wav2Lip on isolated speaker
    run_command("python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth --face right_half.mp4 --audio hindi_raw.wav --outfile right_sync.mp4")

    # 4. Face Restoration & Re-stitching
    print("Restoring Face and Compiling at 60 FPS...")
    run_command("python GFPGAN/inference_gfpgan.py -i right_sync.mp4 -o final_out -v 1.4 -s 2")
    
    # Final assembly with original left-half background
    run_command("ffmpeg -y -i clip_15s.mp4 -i final_out/restored_videos/right_sync.mp4 -filter_complex '[0:v][1:v]overlay=W/2:0' -framerate 60 final_submission.mp4")

if __name__ == "__main__":
    process_dubbing("clip_15s.mp4")
