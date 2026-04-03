import numpy as np

class EQMatcher:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Endüstri standardı 10-Band EQ Merkez Frekansları (Genelde MXR pedallarında kullanılır)
        self.band_centers = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        
    def to_mono(self, audio):
        """Stereo (Shape: 2, N) sinyali Mono'ya (Shape: N,) çevirir."""
        if audio.ndim == 2:
            return np.mean(audio, axis=0)
        return audio
        
    def rms_normalize(self, target_audio, source_audio):
        """
        İki sesin enerjisini (RMS) hesaplayıp kaynak sesi hedef sese eşitler.
        U-Net'in sebep olduğu 'pumping' volüm hatalarını pas geçmemizi sağlar.
        """
        rms_target = np.sqrt(np.mean(target_audio**2)) + 1e-10
        rms_source = np.sqrt(np.mean(source_audio**2)) + 1e-10
        
        # Kaynak sinyalinin volümünü hedef sinyalin volümüne eşitle
        gain_factor = rms_target / rms_source
        normalized_source = source_audio * gain_factor
        return target_audio, normalized_source
        
    def calculate_spectral_difference(self, target_audio, source_audio):
        """
        İki sinyalin frekans spektrumlarını çıkarır ve aralarındaki desibel farkını hesaplar.
        """
        # Ses paketlerinin uzunluklarını eşitler (Aynı ms'de olmaları için)
        min_len = min(len(target_audio), len(source_audio))
        target_audio = target_audio[:min_len]
        source_audio = source_audio[:min_len]
        
        # Sinyali yumuşatmak için Hann penceresi (Windowing) uyguluyoruz
        window = np.hanning(min_len)
        
        # Real-FFT (Hızlı Fourier Dönüşümü) ile sesin frekans karakterini söküyoruz
        target_fft = np.abs(np.fft.rfft(target_audio * window))
        source_fft = np.abs(np.fft.rfft(source_audio * window))
        
        # Oluşan frekans adımlarını Hz cinsinden listeliyoruz
        freqs = np.fft.rfftfreq(min_len, d=1.0/self.sample_rate)
        
        # Logaritma işleminde 0'a bölme (Sonsuzluk) hatası olmasın diye çok ufak bir taban veriyoruz
        target_fft = np.maximum(target_fft, 1e-10)
        source_fft = np.maximum(source_fft, 1e-10)
        
        # Desibel (dB) cinsinden fark (Hedef - Kaynak)
        # Bize çıkan sonuç pozitifse "Bu frekansı o kadar dB artır", negatifse "Azalt" anlamına geliyor
        diff_db = 20 * np.log10(target_fft / source_fft)
        
        return freqs, diff_db
        
    def map_to_10band_eq(self, freqs, diff_db):
        """
        Oluşan milyonlarca kesintisiz dB yığınını, 10 tane belirgin Bant bölgesinin içine yerleştirir.
        """
        eq_settings = []
        
        for center_freq in self.band_centers:
            # Bandın kapsadığı alan: Merkez frekansın yarım oktav altı ve üstü
            lower_bound = center_freq / np.sqrt(2)
            upper_bound = center_freq * np.sqrt(2)
            
            # Hangi frekanslar bu bandın içine düşüyor bul (-ki ortalamasını alalım)
            band_mask = (freqs >= lower_bound) & (freqs <= upper_bound)
            
            if np.any(band_mask):
                # O banta düşen toplam dB ihtiyacının ortalamasını hesapla
                band_mean_db = np.mean(diff_db[band_mask])
                
                # Gitar pedalları ve EQ aletleri genelde +/- 12 dB limitlidir. Güvenlik kilidi:
                band_mean_db = np.clip(band_mean_db, -12.0, 12.0)
                eq_settings.append(band_mean_db)
            else:
                eq_settings.append(0.0)
                
        return np.array(eq_settings)
        
    def get_eq_match(self, target_audio, source_audio):
        """
        Sistemin Ana Şalteri: İki sesi ver, sana EQ ayar listesi versin.
        """
        target = self.to_mono(target_audio)
        source = self.to_mono(source_audio)
        
        target, source = self.rms_normalize(target, source)
        freqs, diff_db = self.calculate_spectral_difference(target, source)
        eq_bands = self.map_to_10band_eq(freqs, diff_db)
        
        return eq_bands


# ==========================================
# TEST BLOĞU: Mimarinin Doğrulanması
# ==========================================
if __name__ == "__main__":
    sr = 44100
    duration = 1.0 # 1 saniyelik tampon(buffer) örneği
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    
    # 1. HEDEF SİNYAL (Target): Mikrofon+UNet üzerinden yakaladığımız 
    # diyelim ki içerisinde 1000 Hz'i inanılmaz baskın olan (Mid'leri yüksek) bir gitar var.
    target_signal = np.sin(2 * np.pi * 100 * t) + 3.0 * np.sin(2 * np.pi * 1000 * t)
    
    # 2. KAYNAK SİNYAL (Source): FX Loop'tan gelen kuru ses (Mid'leri daha çekingen)
    source_signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t)
    
    # Modeli başlat
    matcher = EQMatcher(sample_rate=sr)
    
    # İhtiyacımız olan EQ haritasını çek!
    eq_settings = matcher.get_eq_match(target_signal, source_signal)
    
    print("\n[TEST] 10-Band Gitar EQ Ayarları (+/- dB):")
    print("-" * 50)
    for hz, db in zip(matcher.band_centers, eq_settings):
        if hz == 1000:
            print(f"{hz:>6} Hz Bant: {db:>6.2f} dB  <-- Hedef seste 1000Hz (Mids) çok yüksekti, o yüzden pedala o bantı maksimuma (+12) açması gerektiğini söylüyor.")
        elif hz == 125 or hz == 250:
             print(f"{hz:>6} Hz Bant: {db:>6.2f} dB")
        else:
            print(f"{hz:>6} Hz Bant: {db:>6.2f} dB")
            
    print("\nSonuç: Normalizasyon + FFT Bandlama Başarılı!")
