import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class SonicID_Dataset(Dataset):
    def __init__(self, root_dir, target_file="gitar.wav", window_size=3.0, sample_rate=44100):
        self.root_dir = root_dir
        self.target_file = target_file
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.chunk_samples = int(window_size * sample_rate)

        # ---------------------------------------------------------
        # YENİ EKLENEN KISIM: STFT (Spektrogram) Ayarları
        # Sesi yapay zekanın görebileceği bir "ısı haritasına" çevirir.
        # ---------------------------------------------------------
        self.n_fft = 1024
        self.hop_length = 512
        self.spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0 # Güç spektrogramı
        )

        self.tracks = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]

    def apply_stage_simulation(self, waveform, cutoff_freq, noise_level):
        """
        Canlı sahneyi simüle eder:
        1. Basit Oda EQ'su (Lowpass) - Kabin ve odadaki tiz yutulması
        2. Zemin Gürültüsü (Noise) - Sahnede oluşan dip ses
        """
        import torchaudio.functional as F
        
        # Biquad lowpass kullanarak sahne boğukluğunu veriyoruz
        waveform = F.lowpass_biquad(waveform, self.sample_rate, cutoff_freq)
        
        # Çok hafif dip ses (Noise)
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        return waveform

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]

        mix_path = os.path.join(track_path, "mixture.wav")
        target_path = os.path.join(track_path, self.target_file)

        import soundfile as sf
        info = sf.info(mix_path)
        total_samples = info.frames

        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        # torchaudio.load Windows'ta FFmpeg veya torchcodec DLL hatalarına (Libtorchcodec Error) yol açtığından,
        # en hatasız ve stabil çalışan "soundfile.read" metodunu kullanıyoruz.
        mix_np, _ = sf.read(mix_path, start=start_frame, frames=self.chunk_samples, dtype='float32', always_2d=True)
        target_np, _ = sf.read(target_path, start=start_frame, frames=self.chunk_samples, dtype='float32', always_2d=True)
        
        # soundfile veriyi (frames, channels) numpy formatında verir. 
        # Modellerimiz Tensor kullandığı için Pytorch'un beklentisi olan (channels, frames) formatına ".T" (Transpose) ile çeviriyoruz.
        mix_chunk = torch.from_numpy(mix_np).T
        target_chunk = torch.from_numpy(target_np).T

        if mix_chunk.shape[1] < self.chunk_samples:
            pad_amount = self.chunk_samples - mix_chunk.shape[1]
            mix_chunk = torch.nn.functional.pad(mix_chunk, (0, pad_amount))
            target_chunk = torch.nn.functional.pad(target_chunk, (0, pad_amount))

        # ---------------------------------------------------------
        # YENİ EKLENEN KISIM: Sahne Simülasyonu ve Referans Kopyası
        # ---------------------------------------------------------
        # 1. Orijinal FX loop sesi (Tertemiz, odaya çıkmamış izole gitar)
        reference_chunk = target_chunk.clone()
        
        # 2. Sahne / Oda akustiğini temsil edecek rastgele değerler
        cutoff_freq = random.uniform(3000.0, 8000.0)
        noise_level = random.uniform(0.001, 0.005)
        
        # 3. Miks ve Hedef sese sahne simülasyonunu uyguluyoruz
        mix_chunk = self.apply_stage_simulation(mix_chunk, cutoff_freq, noise_level)
        target_chunk = self.apply_stage_simulation(target_chunk, cutoff_freq, noise_level)

        # 4. Fiziksel Gecikme (Time Delay) Simülasyonu
        # Havada yayılan sesin (mikrofonun) kabloya (FX Loop) göre gecikmesi
        # 0 ila 30 ms arası rastgele bir gecikme ekleyelim (yaklaşık 0-10 metre mesafe)
        delay_ms = random.uniform(0, 30.0)
        delay_samples = int((delay_ms / 1000.0) * self.sample_rate)
        
        if delay_samples > 0:
            # Sinyali sağa kaydırmak için başa sıfır ekleyip sondan kırpıyoruz (Shape bozulmasın diye)
            mix_chunk = torch.nn.functional.pad(mix_chunk, (delay_samples, 0))[:, :-delay_samples]
            target_chunk = torch.nn.functional.pad(target_chunk, (delay_samples, 0))[:, :-delay_samples]

        # ---------------------------------------------------------
        # Dönüşümü Uygulama (Spektrogramlara Çevirme)
        # ---------------------------------------------------------
        mix_spec = self.spectrogram(mix_chunk)
        reference_spec = self.spectrogram(reference_chunk)
        target_spec = self.spectrogram(target_chunk)

        return mix_spec, reference_spec, target_spec

# Test Kodu
if __name__ == "__main__":
    # İŞTE BURASI: İndirdiğin MUSDB18-HQ verisinin "train" klasörünün yolunu buraya yazıyoruz.
    # Windows yollarında ters slash (\) hata vermesin diye başa 'r' koyuyoruz.
    veri_yolu = r"C:\Users\jiyan\Desktop\sonic-id\data\train" 
    
    print(f"Veri şu konumda aranıyor: {veri_yolu}")
    
    try:
        # Sınıfı çağırıyoruz ve yolu içine gönderiyoruz
        dataset = SonicID_Dataset(root_dir=veri_yolu, window_size=3.0)
        print(f"Harika! Klasörde toplam {len(dataset)} şarkı bulundu.")
        
        # İlk şarkıdan 3 saniyelik bir örnek çekip dönüşümü test edelim
        mix_spec, reference_spec, target_spec = dataset[0]
        print(f"Mix Spektrogram Boyutu (Girdi 1): {mix_spec.shape}")
        print(f"Referans Spektrogram Boyutu (Girdi 2): {reference_spec.shape}")
        print(f"Target Spektrogram Boyutu (Hedef): {target_spec.shape}")
        print("1. Adım başarıyla tamamlandı, veriler sahne simülasyonuyla birlikte resimlere dönüştü!")
        
    except Exception as e:
        print(f"Bir hata oluştu. Klasör yolunu doğru yazdığından emin ol.\nHata detayı: {e}")