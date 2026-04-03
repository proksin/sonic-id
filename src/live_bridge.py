import sys
import os
import time
import torch
import numpy as np

# Projedeki diğer modüllerimizi içe aktarıyoruz
from model import UNet
from eq_matcher import EQMatcher
from inference import process_for_hrace

class LivePedalProcessor:
    def __init__(self, sample_rate=44100, buffer_duration=3.0):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.chunk_samples = int(sample_rate * buffer_duration)
        
        # Ekran Kartı (Eğer yoksa CPU/MPS)
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")
        
        print(f"[BOOT] Cihaz: {self.device}")
        
        # Modelleri (Beyinleri) Başlat
        print("[BOOT] U-Net AI Beyni (Ayırıcı) Yükleniyor...")
        self.ai_model = UNet(in_channels=2, out_channels=2).to(self.device)
        self.ai_model.eval() # Öğrenme modunu kapat, sadece kullan.
        
        print("[BOOT] EQ Karşılaştırıcı (DSP) Yükleniyor...")
        self.eq_matcher = EQMatcher(sample_rate=self.sample_rate)
        
    def simulate_audio_capture(self):
        """
        Gerçek mikrofon yerine sistemi test etmek için 3 saniyelik sahte gitar sesi üretir.
        """
        t = np.linspace(0, self.buffer_duration, self.chunk_samples, endpoint=False)
        
        # 1. Kaynak: FX Loop'tan Gelen Direkt Ses (Clean DI) - Örneğin 400Hz ve 1200Hz barındırıyor
        di_signal = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)
        
        # 2. Kaynak: Odayı Dinleyen Mikrofon (Mix) Sesi 
        # (İçinde davul zili var 8000Hz, ve gitarın 1200Hz olan yeri akustik sebebiyle odada çok zayıf çıkıyor)
        mix_signal = np.sin(2 * np.pi * 400 * t) + 0.05 * np.sin(2 * np.pi * 1200 * t) + 1.5 * np.sin(2 * np.pi * 8000 * t)
        
        return mix_signal, di_signal

    def process_buffer(self, mix_audio_np, di_audio_np):
        """
        Gelen paketleri (3 Sn) alır, AI'a atar ve sonrasında EQ eşleştiriciye paslar.
        """
        # U-Net'in kullanması için numpy'dan torch tensörüne çevir: (Channels, Frames)
        # Sinyali stereo gibi göstermek için repeat(2,1) yapıyoruz çünkü model stereo çalışıyor
        mix_tensor = torch.from_numpy(mix_audio_np).float().unsqueeze(0).repeat(2, 1)
        
        # 1. Aşama: AI (U-Net) Sesi Ayıklar
        with torch.no_grad():
            isolated_tensor = process_for_hrace(mix_tensor, self.sample_rate, self.ai_model, self.device)
            # İzole edilmiş sesi Numpy'a ve tek kanala(Mono) geri döndür
            isolated_np = isolated_tensor.cpu().numpy().mean(axis=0) 
            
        # 2. Aşama: EQ Karşılaştırması 
        # (Target/Hedef = AI'dan çıkan izole edilmiş mikrofon sesi)
        # (Source/Kaynak = FX Loop'taki DI gitar sesi)
        eq_settings = self.eq_matcher.get_eq_match(target_audio=isolated_np, source_audio=di_audio_np)
        
        return eq_settings

    def run_live_loop(self):
        """
        Pedalın sürekli / sonsuz döngüde çalıştığı yer "Always On"
        """
        print("\n" + "="*50)
        print("🟢 CANLI MOD AKTİF: Sisteme Ses Geliyor (Simülasyon)...")
        print("="*50 + "\n")
        
        try:
            loop_count = 1
            while True:  # Sonsuz Pedal Döngüsü
                start_time = time.time()
                
                # A. Sesi Yakala (Ses kartından 3 saniyelik paket alır)
                mix_buffer, di_buffer = self.simulate_audio_capture()
                
                # B. İşlemi Başlat (AI -> EQ)
                eq_recommendation = self.process_buffer(mix_buffer, di_buffer)
                
                # Terminalde Gösterim
                process_time = time.time() - start_time
                print(f"--- [ Paket {loop_count} | İşlem Süresi: {process_time:.2f} Saniye ] ---")
                
                # Sadece en çok müdahale gerektiren (örneğin +/- 5 dB'den büyük) bantları göstererek kalabalığı önle
                for hz, db in zip(self.eq_matcher.band_centers, eq_recommendation):
                    if abs(db) > 1.0:
                        bar = "#" * int(abs(db))
                        yön = "ARTTIR (+)" if db > 0 else "AZALT  (-)"
                        print(f"{hz:>6} Hz Bant: {db:>6.2f} dB  {yön} | {bar}")
                print("\n")
                
                loop_count += 1
                
                # CPU'yu çok boğmamak için ve 3 saniyelik bufferın dolmasını beklemek için bekle
                time.sleep(3.0) 
                
        except KeyboardInterrupt:
            print("\n🔴 Pedal Kapatıldı (Güç Kesildi).")

if __name__ == "__main__":
    pedal = LivePedalProcessor(buffer_duration=3.0)
    pedal.run_live_loop()
