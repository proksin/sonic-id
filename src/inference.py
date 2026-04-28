import os
import torch
import torchaudio
from model import UNet

def process_for_hrace(audio_tensor, reference_tensor, sample_rate, model, device):
    """
    [H-RACE Pipeline]
    audio_tensor   : Mikrofon sinyali (mixture.wav) - 2 kanal
    reference_tensor: FX Loop sinyali (other.wav / temiz gitar) - 2 kanal
    """
    audio_tensor     = audio_tensor.to(device)
    reference_tensor = reference_tensor.to(device)
    
    window = torch.hann_window(1024).to(device)
    
    def to_power_spec(wav):
        stft = torch.stft(wav, n_fft=1024, hop_length=512, window=window,
                          return_complex=True, pad_mode="constant")
        magnitude = torch.abs(stft)
        phase     = torch.angle(stft)
        power     = (magnitude ** 2).unsqueeze(0)  # (1, C, F, T)
        return power, phase
    
    mix_power, mix_phase = to_power_spec(audio_tensor)
    ref_power, _         = to_power_spec(reference_tensor)
    
    # Model 4 kanallı girdi bekliyor: [mix(2ch) + reference(2ch)]
    combined_input = torch.cat([mix_power, ref_power], dim=1)  # (1, 4, F, T)
    
    with torch.no_grad():
        mask = model(combined_input)
    
    mask = mask.squeeze(0)
    
    isolated_power     = mix_power.squeeze(0) * mask
    isolated_magnitude = torch.sqrt(isolated_power)
    isolated_stft      = isolated_magnitude * torch.exp(1j * mix_phase)
    
    isolated_waveform = torch.istft(
        isolated_stft, n_fft=1024, hop_length=512,
        window=window, return_complex=False
    )
    return isolated_waveform

def separate_source(mix_path, reference_path, model_checkpoint, output_path):
    """
    mix_path       : Mikrofon kaydı (mixture.wav) - ortam sesi
    reference_path : FX Loop kaydı (other.wav / temiz kuru gitar)
    """
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f">> Cihaz: {device}. Model yükleniyor...")
    
    model = UNet(in_channels=4, out_channels=2).to(device)
    try:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=False))
        model.eval()
        print("[BAŞARILI] Ağırlıklar (Checkpoint) okundu.")
    except Exception as e:
        print(f"[HATA!] Checkpoint yüklenemedi: {e}")
        return

    import soundfile as sf
    import numpy as np

    def load_wav(path):
        audio_np, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = torch.from_numpy(audio_np).T  # (channels, frames)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # Mono → Stereo
        return wav, sr

    print(f"[*] Mikrofon sinyali: {mix_path}")
    waveform, sample_rate = load_wav(mix_path)

    print(f"[*] FX Loop referansı: {reference_path}")
    reference, _ = load_wav(reference_path)

    # Referans uzunluğunu mix ile eşitle
    min_len = min(waveform.shape[1], reference.shape[1])
    waveform  = waveform[:, :min_len]
    reference = reference[:, :min_len]

    print(">> Sinyal İşleniyor...")
    
    chunk_sec    = 15
    chunk_length = int(chunk_sec * sample_rate)
    total_samples = waveform.shape[1]
    isolated_chunks = []
    
    for start_idx in range(0, total_samples, chunk_length):
        end_idx      = min(start_idx + chunk_length, total_samples)
        mix_chunk    = waveform[:, start_idx:end_idx]
        ref_chunk    = reference[:, start_idx:end_idx]
        
        if mix_chunk.shape[1] > 2048:
            out_chunk = process_for_hrace(mix_chunk, ref_chunk, sample_rate, model, device)
            isolated_chunks.append(out_chunk.cpu())
            yuzde = (end_idx / total_samples) * 100
            print(f"   [+] %{yuzde:.1f} tamamlandı.")
    
    isolated_waveform = torch.cat(isolated_chunks, dim=1)
    out_np = isolated_waveform.cpu().numpy().T
    sf.write(output_path, out_np, sample_rate)
    
    print(f"\n=========== HARİKA! ===========")
    print(f"Izole ses şuraya kaydedildi:\n> {output_path}")

if __name__ == "__main__":
    # mixture.wav  = mikrofon kaydı (ortam sesi, tüm mix)
    # other.wav    = FX loop referansı (temiz, kuru gitar)
    test_miks      = r"C:\Users\jiyan\Desktop\sonic-id\test_song\mixture.wav"
    test_referans  = r"C:\Users\jiyan\Desktop\sonic-id\test_song\other.wav"
    test_model     = r"C:\Users\jiyan\Desktop\sonic-id\sonic_id_best_model.pth"
    cikis_yolu     = r"C:\Users\jiyan\Desktop\sonic-id\izole_gitar_ornek.wav"
    
    print("Sonic ID - Inference Test")
    
    for f in [test_miks, test_referans, test_model]:
        if not os.path.exists(f):
            print(f"\n[EKSIK DOSYA] {f}")
            exit()
    
    separate_source(
        mix_path=test_miks,
        reference_path=test_referans,
        model_checkpoint=test_model,
        output_path=cikis_yolu
    )
