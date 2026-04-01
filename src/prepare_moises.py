import os
import soundfile as sf
import numpy as np
import json
import glob

def prepare_moises_data(moises_dir, output_dir, target_instrument="vocals"):
    """
    MoisesDB formatındaki klasörleri (bass, drums, vb. alt klasörler), 
    SonicID modelinin eğitimde beklediği formata (mixture.wav ve diğer) dönüştürür.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    song_dirs = [d for d in os.listdir(moises_dir) if os.path.isdir(os.path.join(moises_dir, d))]
    
    print(f"Toplam {len(song_dirs)} şarkı bulundu. Dönüştürülüyor...")
    
    for idx, song_folder in enumerate(song_dirs):
        song_path = os.path.join(moises_dir, song_folder)
        
        # Eğer klasör değilse atla
        if not os.path.isdir(song_path): continue
            
        out_song_path = os.path.join(output_dir, f"song_{idx:03d}")
        if not os.path.exists(out_song_path):
            os.makedirs(out_song_path)
            
        # Şarkı içindeki tüm wav dosyalarını bulalım
        all_wavs = glob.glob(os.path.join(song_path, "*", "*.wav"))
        if len(all_wavs) == 0:
            continue
            
        print(f"[{idx+1}/{len(song_dirs)}] İşleniyor: {song_folder}")
        
        mixture_audio = None
        target_audio = None
        sr = 44100
        
        # Her bir wav dosyasını oku ve mix'in içine ekle
        for wav_file in all_wavs:
            parent_dir = os.path.basename(os.path.dirname(wav_file))
            audio, sr = sf.read(wav_file, always_2d=True)
            
            # Mixture'a ekle (Üst üste bindir - Mixle)
            if mixture_audio is None:
                mixture_audio = np.zeros_like(audio)
            
            # Eğer boyutlar uyuşmuyorsa, en kısa olana göre kes (Normalde hepsi aynı boyuttadır)
            min_len = min(mixture_audio.shape[0], audio.shape[0])
            mixture_audio = mixture_audio[:min_len]
            audio_adj = audio[:min_len]
            
            mixture_audio += audio_adj
            
            # Hedef enstrüman ise (Örn: "vocals" veya "guitar"), target.wav için ayır
            if target_instrument in parent_dir.lower():
                if target_audio is None:
                    target_audio = np.zeros_like(audio)
                target_audio_adj = target_audio[:min_len]
                target_audio_adj += audio_adj
                target_audio = target_audio_adj
                
        # Kalau target_audio hiç bulunamadıysa içini boş (sessiz) dolduralım (Model hata vermesin diye)
        if target_audio is None:
            print(f"  Uyarı: Bu şarkıda '{target_instrument}' bulunamadı. Boş bir dosya oluşturuluyor.")
            target_audio = np.zeros_like(mixture_audio)
            
        # Kaydet
        sf.write(os.path.join(out_song_path, "mixture.wav"), mixture_audio, sr)
        
        # dataset.py içerisinde "other.wav" olarak arandığı için your dosyayı "other.wav" adıyla kaydediyoruz.
        sf.write(os.path.join(out_song_path, "other.wav"), target_audio, sr)

if __name__ == "__main__":
    MOISES_KLASORU = r"C:\Users\jiyan\Desktop\mosie"
    HEDEF_KLASOR = r"C:\Users\jiyan\Desktop\sonic-id\data\train"
    AYRILACAK_ENSTRUMAN = "guitar"  # Veya 'guitar', 'bass' vs. ne istersen yaz.
    
    prepare_moises_data(MOISES_KLASORU, HEDEF_KLASOR, target_instrument=AYRILACAK_ENSTRUMAN)
    print("İşlem tamam! Veriler artık dataset.py'nin okuyabileceği formata çevrildi.")
