import os
import torch
import torchaudio
from model import UNet

def process_for_hrace(audio_tensor, sample_rate, model, device):
    """
    [GELECEK FAZ - MODÜLERLİK]
    Bu fonksiyon ileride "H-RACE (Dinamik EQ/Kalibrasyon)" modülü için yazılmıştır.
    Miksajın ses dalgasını tensor olarak alır ve izole edilmiş tensöre dönüştürerek 
    H-RACE pipeline'ına iletir. 
    Dosyaya yazmaz, sadece memory (RAM/VRAM) üzerinde tensör çevirir.
    """
    audio_tensor = audio_tensor.to(device)
    
    # 1. Karmaşık STFT hesaplayarak Faz (Phase) ve Genliği (Magnitude) ayırma
    window = torch.hann_window(1024).to(device)
    stft_complex = torch.stft(
        audio_tensor,
        n_fft=1024,
        hop_length=512,
        window=window,
        return_complex=True,
        pad_mode="constant" # Kenar bozulmalarından kaçınmak için
    )
    
    magnitude = torch.abs(stft_complex)
    phase = torch.angle(stft_complex)
    
    # Eğitim 'power=2.0' (Güç spektrogramı) üzerinden yapıldı.
    # U-Net'in algılayabilmesi için güce çevir. (Batch boyutu eklendi = unsqueeze)
    power_spec = (magnitude ** 2).unsqueeze(0)
    
    # 2. Model Çıkarımı (Maske Tahmini)
    with torch.no_grad():
        mask = model(power_spec)
    
    mask = mask.squeeze(0) # Batch boyutunu çıkar
    
    # 3. İzole Edilmiş Sesin İnşası
    isolated_power = power_spec.squeeze(0) * mask
    isolated_magnitude = torch.sqrt(isolated_power)
    
    # 4. KOPMUŞ FAZI (PHASE) GERİ YAPIŞTIRMA (Crucial Step!)
    # Burada sadece genlik üzerinden giden Griffin-Lim yerine;
    # Orijinal miksteki faz bilgisini (exp(1j * phase)) genlikle çarpıp net bir çıkış alıyoruz.
    isolated_stft = isolated_magnitude * torch.exp(1j * phase)
    
    # Inverse STFT (iSTFT) - Spektrogramı tekrar dinlenebilir sese çevirme
    isolated_waveform = torch.istft(
        isolated_stft,
        n_fft=1024,
        hop_length=512,
        window=window,
        return_complex=False
    )
    
    return isolated_waveform

def separate_source(mix_path, model_checkpoint, output_path):
    """
    Kullanıcıya Yönelik Çıkarım Fonksiyonu:
    Verilen .wav dosyasını izole eder ve hedef diske .wav olarak geri yazar.
    """
    # Donanım kararı
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f">> Cihaz: {device}. Model yükleniyor...")
    
    # Modeli Başlat ve Önceden Öğrenilmiş Ağırlıkları (Weights) Yükle
    model = UNet(in_channels=2, out_channels=2).to(device)
    try:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model.eval()
        print("[BAŞARILI] Ağırlıklar (Checkpoint) okundu ve beyin nakli yapıldı.")
    except Exception as e:
        print(f"[HATA!] Checkpoint dosyası yüklenemedi: {e}")
        return

    # Sesi Yükle (Windows DLL yamasını uyguluyoruz: torchaudio.load yerine soundfile.read)
    print(f"[*] İşlenecek Müzik: {mix_path}")
    import soundfile as sf
    import numpy as np

    audio_np, sample_rate = sf.read(mix_path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(audio_np).T # numpy (Frames, Channels) -> torch (Channels, Frames)
    
    # Modelin doğru çalışması için kanal sayısını eşitle (Eğer Mono ise Stereo yap)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    print(">> Sinyal İşleniyor (Yapay Zeka ve iSTFT Devrede)...")
    
    # VRAM (Ekran Kartı Belleği) 8GB ve altında olduğunda, 4-5 dakikalık bir şarkının devasa 
    # spektrogramı tek seferde GPU'ya sığmaz (CUDA Out of Memory).
    # O yüzden parçayı 15 saniyelik küçük bloklara (chunks) bölüp işleyerek GPU'yu kurtarıyoruz.
    chunk_sec = 15
    chunk_length = int(chunk_sec * sample_rate)
    total_samples = waveform.shape[1]
    
    isolated_chunks = []
    
    for start_idx in range(0, total_samples, chunk_length):
        end_idx = min(start_idx + chunk_length, total_samples)
        chunk_tensor = waveform[:, start_idx:end_idx]
        
        # Son kalan parça çok küçük (STFT yapılamayacak kadar) ise atla
        if chunk_tensor.shape[1] > 2048:
            out_chunk = process_for_hrace(chunk_tensor, sample_rate, model, device)
            isolated_chunks.append(out_chunk.cpu())
            
            yuzde = (end_idx / total_samples) * 100
            print(f"   [+] Parça işlendi... %{yuzde:.1f} tamamlandı.")

    # Çıkan 15 saniyelik temiz blokları tek bir müzik olarak uç uca birleştir
    isolated_waveform = torch.cat(isolated_chunks, dim=1)
    
    # Diske Yaz (Windows DLL hatasını yinelememek için torchaudio.save yerine soundfile.write)
    out_np = isolated_waveform.cpu().numpy().T
    sf.write(output_path, out_np, sample_rate)
    
    print(f"=========== HARİKA! ===========")
    print(f"İzole edilmiş temiz (phase kurtarılmış) gitar/other .wav formatı şuraya oluşturuldu:\n> {output_path}")

# Test Çalıştırması
if __name__ == "__main__":
    # Parametre yolları (Gerçek test sırasında buraları değiştirirsiniz)
    test_miks = r"c:\Users\jiyan\Desktop\blackened.wav"
    test_model = r"C:\Users\jiyan\Desktop\sonic-id\checkpoints\sonic_id_best_model.pth"
    cikis_yolu = r"C:\Users\jiyan\Desktop\sonic-id\data\izole_gitar_ornek.wav"
    
    print("Sonic ID (Phase 1) - Inference Test")
    
    # Küçük bir güvenlik: model dosyası yoksa denemeyelim
    if not os.path.exists(test_miks):
        print("\n(BİLGİ) Test için belirlenen miks dosyası bulunamadı. Lütfen test_miks yolunu güncelleyin.")
    elif not os.path.exists(test_model):
        print("\n(BİLGİ) Seçili checkpoint (model) ağırlığı henüz yok. Önce train.py'yi çalıştırmalısınız.\nBeklenen yol:", test_model)
    else:
        separate_source(
            mix_path=test_miks,
            model_checkpoint=test_model,
            output_path=cikis_yolu
        )
