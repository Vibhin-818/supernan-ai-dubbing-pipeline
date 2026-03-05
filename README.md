# Supernan AI: High-Fidelity Video Dubbing Pipeline (₹0 Budget)

This repository contains a modular Python pipeline designed to automate the dubbing of training videos from English/Kannada to Hindi with high visual and audio fidelity.

## 🚀 Key Features
* **Code-Switching Support:** Uses Whisper `task="translate"` to handle mixed Kannada-English speech accurately.
* **Natural Voice Cloning:** Zero-shot cloning with Coqui XTTS v2, featuring a smart-sync algorithm that avoids robotic time-stretching.
* **Spatial Isolation Sync:** An engineering workaround using FFmpeg to isolate the speaker and prevent the AI from lip-syncing background faces (like those on TV screens).
* **Face Restoration:** Integrated GFPGAN v1.4 to upscale and sharpen the mouth area, fixing common Wav2Lip blurriness.



## 🛠️ Architecture & Resourcefulness
Built for a strict ₹0 compute budget, this pipeline avoids expensive APIs (ElevenLabs/OpenAI).
1. **Whisper Medium:** Handles transcription and translation of regional dialects.
2. **Coqui TTS:** Multi-lingual voice cloning with a PyTorch 2.6 security patch.
3. **Wav2Lip + GFPGAN:** High-resolution visual synchronization.

## 💰 Scaling & Cost Analysis
### Estimated Cost per Minute
Using open-source models on a cloud GPU (e.g., NVIDIA T4 or RTX 4090):
* **Software Cost:** ₹0 (Open Source).
* **Compute Cost:** ~$0.02 - $0.05 per minute of processed video (based on ~$0.40/hr spot instances).

### The "500 Hours Overnight" Challenge
To process 500 hours of video in a single night, I would implement a **Distributed Worker Architecture**:
* **Chunking:** Use FFmpeg to split long videos into 3-minute segments with 5-second overlaps.
* **Message Broker:** Queue these segments into **RabbitMQ** or **Redis**.
* **Worker Nodes:** Deploy a Kubernetes cluster of GPU-enabled pods (Auto-scaling).
* **Parallel Processing:** Each pod handles one chunk (Transcription -> Dub -> Sync).
* **Assembly:** A final node concatenates the finished chunks into the final video files.



## 🏗️ Setup
1. Clone the repo: `git clone https://github.com/[Vibhin-818/supernan-dubbing`
2. Install dependencies: `pip install -r requirements.txt`
3. Download checkpoints for Wav2Lip and GFPGAN.
4. Run the pipeline: `python dub_video.py`

## 🚧 Known Limitations
* **Numba Compilation:** Initial processing of the first frame takes ~60 seconds due to mathematical compilation.
* **Hardware:** Requires minimum 16GB VRAM for stable GFPGAN execution at 1080p.
