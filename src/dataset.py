import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class SonicID_Dataset(Dataset):
    def __init__(self, root_dir, window_size=3.0, sample_rate=44100):
        self.root_dir = root_dir
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

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]

        mix_path = os.path.join(track_path, "mixture.wav")
        target_path = os.path.join(track_path, "other.wav")

        info = torchaudio.info(mix_path)
        total_samples = info.num_frames

        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        mix_chunk, _ = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=self.chunk_samples)
        target_chunk, _ = torchaudio.load(target_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        if mix_chunk.shape[1] < self.chunk_samples:
            pad_amount = self.chunk_samples - mix_chunk.shape[1]
            mix_chunk = torch.nn.functional.pad(mix_chunk, (0, pad_amount))
            target_chunk = torch.nn.functional.pad(target_chunk, (0, pad_amount))

        # ---------------------------------------------------------
        # YENİ EKLENEN KISIM: Dönüşümü Uygulama
        # ---------------------------------------------------------
        mix_spec = self.spectrogram(mix_chunk)
        target_spec = self.spectrogram(target_chunk)

        return mix_spec, target_spec

# Test Kodu
if __name__ == "__main__":
    # İŞTE BURASI: İndirdiğin MUSDB18-HQ verisinin "train" klasörünün yolunu buraya yazıyoruz.
    # Windows yollarında ters slash (\) hata vermesin diye başa 'r' koyuyoruz.
    veri_yolu = r"C:\Users\jiyan\Desktop\musdb18hq\train" 
    
    print(f"Veri şu konumda aranıyor: {veri_yolu}")
    
    try:
        # Sınıfı çağırıyoruz ve yolu içine gönderiyoruz
        dataset = SonicID_Dataset(root_dir=veri_yolu, window_size=3.0)
        print(f"Harika! Klasörde toplam {len(dataset)} şarkı bulundu.")
        
        # İlk şarkıdan 3 saniyelik bir örnek çekip dönüşümü test edelim
        mix_spec, target_spec = dataset[0]
        print(f"Mix Spektrogram Boyutu: {mix_spec.shape}")
        print(f"Target Spektrogram Boyutu: {target_spec.shape}")
        print("1. Adım başarıyla tamamlandı, veriler yapay zekanın okuyacağı resimlere dönüştü!")
        
    except Exception as e:
        print(f"Bir hata oluştu. Klasör yolunu doğru yazdığından emin ol.\nHata detayı: {e}")