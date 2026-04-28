import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from model import UNet 

# ── 10-Bant EQ Ayarları ───────────────────────────────────────────────
EQ_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# ── Ses Yükleme/Kaydetme Yardımcıları (Torchaudio yerine Soundfile) ───
def load_audio(path):
    data, samplerate = sf.read(path)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    tensor = torch.from_numpy(data.T).float()
    return tensor, samplerate

def save_audio(path, tensor, samplerate):
    data = tensor.cpu().numpy().T
    sf.write(path, data, samplerate)

# ── DSP ve Metrik Fonksiyonları ───────────────────────────────────────
def get_band_energies(signal, sr):
    if signal.ndim > 1: signal = signal.mean(axis=0) # Mono'ya çevir
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    fft_mag = np.abs(np.fft.rfft(signal))
    
    energies = []
    for f in EQ_BANDS:
        mask = (freqs >= f*0.7) & (freqs <= f*1.4)
        energies.append(np.mean(fft_mag[mask]) if np.any(mask) else 0)
    
    return np.array(energies) / (np.max(energies) + 1e-8)

def si_snr(estimated, reference):
    ref = reference - reference.mean()
    est = estimated - estimated.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    s_target = alpha * ref
    e_noise = est - s_target
    ratio = np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8)
    return 10 * np.log10(ratio + 1e-8)

def bss_eval(estimated, reference, interference):
    eps = 1e-8
    alpha_ref = np.dot(estimated, reference) / (np.dot(reference, reference) + eps)
    alpha_itr = np.dot(estimated, interference) / (np.dot(interference, interference) + eps)
    
    s_target = alpha_ref * reference
    e_interf = alpha_itr * interference
    e_artif = estimated - s_target - e_interf
    
    sdr = 10 * np.log10(np.sum(s_target**2) / (np.sum((e_interf + e_artif)**2) + eps) + eps)
    sir = 10 * np.log10(np.sum(s_target**2) / (np.sum(e_interf**2) + eps) + eps)
    sar = 10 * np.log10(np.sum((s_target + e_interf)**2) / (np.sum(e_artif**2) + eps) + eps)
    return sdr, sir, sar

def plot_fft_poles(est_e, ref_e, mix_e, score_db, save_path="eq_analiz_raporu.png"):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(EQ_BANDS))
    width = 0.25

    plt.bar(x - width, mix_e, width, label='Giriş (Ham Mix)', color='gray', alpha=0.4)
    plt.bar(x, ref_e, width, label='Hedef (Gerçek Gitar)', color='green', alpha=0.6)
    plt.bar(x + width, est_e, width, label='Sonic-ID (Model)', color='blue')

    plt.xticks(x, [f"{b/1000}k" if b >= 1000 else int(b) for b in EQ_BANDS])
    plt.ylabel('Normalize Edilmiş Akustik Güç')
    plt.title(f'Sonic-ID Spektral Kalibrasyon Analizi (SI-SNR: {score_db:.2f} dB)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.savefig(save_path)
    print(f"\n[ BAŞARILI ] EQ Raporu kaydedildi: {save_path}")

# ── Ana Yürütme (Otomatik Miks + Test) ────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Eğitilmiş .pth modeli")
    parser.add_argument("--folder", type=str, required=True, help="MoisesDB stem'lerinin olduğu klasör")
    parser.add_argument("--duration", type=int, default=10, help="Test edilecek saniye")
    parser.add_argument("--offset", type=int, default=0, help="Şarkıya kaçıncı saniyeden başlanacak")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ BİLGİ ] Model mimarisi ve ağırlıkları yükleniyor... ({device})")
    
    model = UNet(in_channels=4, out_channels=2).to(device)
    state_dict = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[ BİLGİ ] Klasör taranıyor ve {args.offset}. saniyeden itibaren Mixture dalgaları toplanıyor...")
    
    wav_files = []
    for root, dirs, files in os.walk(args.folder):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    
    target_wav = None
    mix_wav = None
    sr = None
    
    for file in wav_files:
        if "mixture_created" in file.lower() or "izole_edilen" in file.lower(): 
            continue 
        
        wav, current_sr = load_audio(file)
        if sr is None: sr = current_sr
        
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        
        if "guitar" in file.lower():
            if target_wav is None:
                target_wav = wav
            else:
                min_len_t = min(target_wav.shape[1], wav.shape[1])
                target_wav = target_wav[:, :min_len_t] + wav[:, :min_len_t]
            print(f"  -> Hedef referans (Gitar) eklendi: {os.path.basename(file)}")
            
        if mix_wav is None:
            mix_wav = wav
        else:
            min_len = min(mix_wav.shape[1], wav.shape[1])
            mix_wav = mix_wav[:, :min_len] + wav[:, :min_len]

    if target_wav is None:
        print("[ HATA ] Klasörde veya alt klasörlerde 'guitar' dosyası bulunamadı!")
        return

    # Sesi kısaltma ve atlama (Offset) işlemi
    start_sample = sr * args.offset
    end_sample = start_sample + (sr * args.duration)
    
    end_sample = min(end_sample, mix_wav.shape[1])
    if start_sample >= mix_wav.shape[1]:
        print(f"[ HATA ] Offset ({args.offset}s) şarkının toplam süresinden daha büyük!")
        return

    mix_wav = mix_wav[:, start_sample:end_sample]
    target_wav = target_wav[:, start_sample:end_sample]

    mix_save_path = os.path.join(args.folder, f"mixture_created_{args.offset}s.wav")
    save_audio(mix_save_path, mix_wav, sr)
    print(f"  -> Mixture başarıyla oluşturuldu: {mix_save_path}")

    # ─────────────────────────────────────────────────────────────
    # Ses Dalgasını Spektrograma Çevirme İşlemi
    # ─────────────────────────────────────────────────────────────
    print(f"\n[ İŞLEM ] İzolasyon yapılıyor ({args.duration} saniyelik kesit)...")
    mix_gpu = mix_wav.to(device)
    ref_gpu = target_wav.to(device)

    window = torch.hann_window(1024).to(device)
    
    def to_power_spec(wav_tensor):
        stft = torch.stft(wav_tensor, n_fft=1024, hop_length=512, window=window, return_complex=True, pad_mode="constant")
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        power = (magnitude ** 2).unsqueeze(0)  
        return power, phase

    with torch.no_grad():
        mix_power, mix_phase = to_power_spec(mix_gpu)
        ref_power, _ = to_power_spec(ref_gpu)
        
        combined_input = torch.cat([mix_power, ref_power], dim=1)
        mask = model(combined_input).squeeze(0)
        
        isolated_power = mix_power.squeeze(0) * mask
        isolated_magnitude = torch.sqrt(isolated_power)
        isolated_stft = isolated_magnitude * torch.exp(1j * mix_phase)
        
        est_wav_tensor = torch.istft(isolated_stft, n_fft=1024, hop_length=512, window=window, return_complex=False)

    est_wav = est_wav_tensor.cpu().numpy()
    target_np = target_wav.numpy()
    mix_np = mix_wav.numpy()

    min_final_len = min(est_wav.shape[-1], target_np.shape[-1], mix_np.shape[-1])
    est_wav = est_wav[..., :min_final_len]
    target_np = target_np[..., :min_final_len]
    mix_np = mix_np[..., :min_final_len]
    interf_np = mix_np - target_np

    # Çıktıyı kaydet
    out_tensor = torch.tensor(est_wav)
    if out_tensor.ndim == 1: out_tensor = out_tensor.unsqueeze(0)
    save_audio(os.path.join(args.folder, f"izole_edilen_test_gitari_{args.offset}s.wav"), out_tensor, sr)

    if est_wav.ndim > 1: est_wav = est_wav.mean(axis=0)
    if target_np.ndim > 1: target_np = target_np.mean(axis=0)
    if mix_np.ndim > 1: mix_np = mix_np.mean(axis=0)
    if interf_np.ndim > 1: interf_np = interf_np.mean(axis=0)

    print("\n" + "="*45)
    print("      SONIC-ID BİLİMSEL TEST SONUÇLARI")
    print("="*45)
    
    si_snr_val = si_snr(est_wav, target_np)
    sdr, sir, sar = bss_eval(est_wav, target_np, interf_np)
    
    print(f"  SI-SNR (İzolasyon Kalitesi): {si_snr_val:+.2f} dB")
    print(f"  SDR (Genel Bozulma Oranı)  : {sdr:+.2f} dB")
    print(f"  SIR (Gürültü Bastırma)     : {sir:+.2f} dB")
    print(f"  SAR (Yapay/Artifakt Etkisi): {sar:+.2f} dB")
    print("="*45)

    print("\n[ İŞLEM ] FFT Direkleri hesaplanıyor ve grafik çiziliyor...")
    est_e = get_band_energies(est_wav, sr)
    ref_e = get_band_energies(target_np, sr)
    mix_e = get_band_energies(mix_np, sr)
    
    plot_path = os.path.join(args.folder, f"eq_analiz_raporu_{args.offset}s.png")
    plot_fft_poles(est_e, ref_e, mix_e, si_snr_val, plot_path)

if __name__ == "__main__":
    main()