import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ── FFT Direkleri (10-Bant EQ) İçin Ayarlar ──────────────────────────────────
EQ_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

def get_band_energies(signal, sr):
    """Sinyalin 10 EQ bandındaki enerji dağılımını hesaplar (FFT Ortalaması)"""
    n_fft = 4096
    # Sinyali mono yap ve FFT al
    if signal.ndim > 1: signal = signal.mean(axis=1)
    
    # Hızlı Fourier Dönüşümü (FFT)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    fft_mag = np.abs(np.fft.rfft(signal))
    
    energies = []
    for f in EQ_BANDS:
        # Her bandın %40 genişliğindeki çevresini topla (Kova mantığı)
        mask = (freqs >= f*0.7) & (freqs <= f*1.4)
        energies.append(np.mean(fft_mag[mask]) if np.any(mask) else 0)
    
    # Görselleştirme için normalize et
    return np.array(energies) / (np.max(energies) + 1e-8)

def plot_fft_poles(est_energies, ref_energies, mix_energies, score_db):
    """10-Bant Karşılaştırma Grafiği Çizer"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(EQ_BANDS))
    width = 0.25

    plt.bar(x - width, mix_energies, width, label='Giriş (Ham Mix)', color='gray', alpha=0.4)
    plt.bar(x, ref_energies, width, label='Hedef (Temiz Gitar)', color='green', alpha=0.6)
    plt.bar(x + width, est_energies, width, label='Sonic-ID (Model Çıktısı)', color='blue')

    plt.xticks(x, [f"{b/1000}k" if b >= 1000 else int(b) for b in EQ_BANDS])
    plt.ylabel('Normalize Edilmiş Akustik Güç')
    plt.title(f'Sonic-ID Spektral Analiz Raporu (SI-SNR: {score_db:.2f} dB)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.savefig('sonuc_analiz.png')
    print("\n[ GÖRSEL ] Analiz grafiği 'sonuc_analiz.png' olarak kaydedildi.")

# ── Senin Mevcut Metrik Fonksiyonların (Aynen Kalıyor) ───────────────────────
def si_snr(estimated, reference):
    ref = reference - reference.mean()
    est = estimated - estimated.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    s_target = alpha * ref
    e_noise   = est - s_target
    ratio = np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8)
    return 10 * np.log10(ratio + 1e-8)

def bss_eval(estimated, reference, interference):
    if estimated.ndim > 1:    estimated    = estimated.mean(axis=1)
    if reference.ndim > 1:    reference    = reference.mean(axis=1)
    if interference.ndim > 1: interference = interference.mean(axis=1)
    min_len = min(len(estimated), len(reference), len(interference))
    est, ref, itr = estimated[:min_len], reference[:min_len], interference[:min_len]
    eps = 1e-8
    alpha_ref = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    alpha_itr = np.dot(est, itr) / (np.dot(itr, itr) + eps)
    s_target, e_interf = alpha_ref * ref, alpha_itr * itr
    e_artif  = est - s_target - e_interf
    sdr = 10 * np.log10(np.sum(s_target**2) / (np.sum((e_interf + e_artif)**2) + eps) + eps)
    sir = 10 * np.log10(np.sum(s_target**2) / (np.sum(e_interf**2) + eps) + eps)
    sar = 10 * np.log10(np.sum((s_target + e_interf)**2) / (np.sum(e_artif**2) + eps) + eps)
    return sdr, sir, sar

# ── Ana Akış (Güncellendi) ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--mixture",   type=str, required=True)
    args = parser.parse_args()

    # Sesleri oku
    est, sr1 = sf.read(args.estimated)
    ref, sr2 = sf.read(args.reference)
    mix, sr3 = sf.read(args.mixture)

    # Uzunluk eşitle
    min_len = min(len(est), len(ref), len(mix))
    est, ref, mix = est[:min_len], ref[:min_len], mix[:min_len]

    # 1. Metrikleri hesapla ve yazdır
    print("\n" + "="*40)
    print("  SONIC ID - PERFORMANS RAPORU")
    print("="*40)
    
    sdr, sir, sar = bss_eval(est, ref, mix - ref)
    si_snr_val = si_snr(est.mean(axis=1) if est.ndim > 1 else est, 
                        ref.mean(axis=1) if ref.ndim > 1 else ref)
    
    print(f"  SI-SNR (İzolasyon) : {si_snr_val:+.2f} dB")
    print(f"  SDR (Genel Kalite) : {sdr:+.2f} dB")
    print(f"  SIR (Bastırma)     : {sir:+.2f} dB")
    print(f"  SAR (Artifakt)     : {sar:+.2f} dB")

    # 2. FFT Direklerini Hesapla ve Çiz
    print("\n[ ANALİZ ] Frekans direkleri hesaplanıyor...")
    est_e = get_band_energies(est, sr1)
    ref_e = get_band_energies(ref, sr2)
    mix_e = get_band_energies(mix, sr3)
    
    plot_fft_poles(est_e, ref_e, mix_e, si_snr_val)

if __name__ == "__main__":
    main()