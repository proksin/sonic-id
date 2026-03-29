import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class SonicID_Dataset(Dataset):
    def __init__(self, root_dir, window_size=3.0, sample_rate=44100):
        """
        root_dir: MUSDB18-HQ klasörünün yolu (örneğin: 'data/train')
        window_size: Kayan pencere boyutu (saniye cinsinden)
        sample_rate: Hedeflenen örnekleme hızı (WAV için genelde 44100)
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.chunk_samples = int(window_size * sample_rate) # 3 saniye * 44100 = 132300 sample

        # Klasördeki tüm şarkı isimlerini (alt klasörleri) bul
        self.tracks = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        # Toplam şarkı sayısını döndürür
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]

        # Sahnede duyduğumuz o kirli, her şeyin birbirine girdiği ses
        mix_path = os.path.join(track_path, "mixture.wav")
        # Bizim H-RACE için "Zemin Gerçeği" (Sadece gitar/klavye gibi diğer sesler)
        target_path = os.path.join(track_path, "other.wav")

        # Şarkının toplam uzunluğunu bul (RAM'e yüklemeden sadece bilgisini alıyoruz)
        info = torchaudio.info(mix_path)
        total_samples = info.num_frames

        # Şarkının içinden rastgele 3 saniyelik bir başlangıç noktası seç
        if total_samples > self.chunk_samples:
            start_frame = random.randint(0, total_samples - self.chunk_samples)
        else:
            start_frame = 0

        # SADECE o 3 saniyelik kısmı RAM'e yükle (Büyük optimizasyon burası)
        mix_chunk, _ = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=self.chunk_samples)
        target_chunk, _ = torchaudio.load(target_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        # Eğer parça 3 saniyeden kısaysa sıfırlarla doldur (Padding)
        if mix_chunk.shape[1] < self.chunk_samples:
            pad_amount = self.chunk_samples - mix_chunk.shape[1]
            mix_chunk = torch.nn.functional.pad(mix_chunk, (0, pad_amount))
            target_chunk = torch.nn.functional.pad(target_chunk, (0, pad_amount))

        return mix_chunk, target_chunk

# Test Kodu
if __name__ == "__main__":
    # Veri setini tanımla (Dizin yolunu kendi indirdiğin yere göre güncellemelisin)
    # Örnek: dataset = SonicID_Dataset(root_dir="../data/musdb18hq/train", window_size=3.0)
    print("Dataset sınıfı başarıyla oluşturuldu!")