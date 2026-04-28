# Sonic-ID: Autonomous Acoustic Calibration & Guitar Isolation System 🎸🤖

**Sonic-ID** is a Deep Learning-based Digital Signal Processing (DSP) system designed to operate autonomously on a Raspberry Pi 5. It performs real-time acoustic calibration by isolating electric guitar signals from highly complex, distorted, and frequency-masked live audio environments (e.g., Heavy Metal, Grunge mixes).

## 🚀 The Core Problem & Solution
In live music environments, aggressive distortion and heavy drum cymbals cause extreme **frequency masking**. Traditional DSP pedals struggle to differentiate between a snare hit and a guitar chug, leading to muddy EQ profiles. 

Sonic-ID solves this by utilizing a custom **Phase-Blind U-Net architecture**. Instead of blind source separation, the model takes a 4-channel STFT spectrogram input (Dirty Mix + Dry Reference Guitar) to accurately mask out drums, bass, and vocals, generating a precise 10-band EQ map for the target amplifier.

## 🧠 Architecture & Mathematical Approach
The system processes audio not as raw waveforms, but as power spectrograms via Short-Time Fourier Transform (STFT). 

* **Input Channels:** $in\_channels=4$ (2 channels for the live mix + 2 channels for the dry reference "Sonic-ID").
* **Output Channels:** $out\_channels=2$ (Stereo Masking Tensor).
* **Processing:** The model outputs a magnitude mask with values between 0.0 and 1.0, which is then multiplied with the original mix spectrogram. The final isolated signal is reconstructed using ISTFT.
* **Loss Function & Metaphor:** The curriculum learning approach involved training initially on isolated stems, followed by fine-tuning on the complex **MUSDB18** dataset.

## 📊 Scientific Benchmarks & Crash Test Results
Isolating electric guitars (categorized as "Other" in standard MSS benchmarks) is notoriously difficult. Industry standards like Spotify's Spleeter typically achieve an SDR of ~4.55 dB in this category.

Sonic-ID was put through an extreme "Crash Test" using **Pantera's "Cemetery Gates"**—a track known for dense frequency overlapping, aggressive double kicks, and high-gain distortion.

| Metric | Score (Vocal & Heavy Chugging Section) | Interpretation |
| :--- | :--- | :--- |
| **SDR** (Signal-to-Distortion) | **+6.21 dB** | Outperforms standard baselines for the "Other" category. |
| **SIR** (Signal-to-Interference) | **+5.61 dB** | Successfully suppresses vocal and cymbal bleed to a negligible level. |
| **SAR** (Signal-to-Artifact) | **+7.42 dB** | Minimal digital artifacting, strictly maintaining the guitar's mid-range character. |

*(Note: During non-vocal heavy riff sections, the SDR peaks at **+7.17 dB** and SIR reaches **+10.84 dB**).*

## 🛠️ Tech Stack & Hardware
* **AI/ML:** Python 3.x, PyTorch, Torchaudio
* **DSP / Audio Processing:** Soundfile (Bypassing Windows FFmpeg/Torchcodec dependencies), NumPy, Librosa
* **Hardware Target:** Raspberry Pi 5 (Edge Computing) + Audio Interface (e.g., Focusrite / Behringer)

## 💻 Installation & Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/sonic-id.git](https://github.com/yourusername/sonic-id.git)
   cd sonic-id