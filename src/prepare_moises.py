import os
import soundfile as sf
import numpy as np
import glob
import shutil

def prepare_dataset(input_dir, output_dir, target_instrument="guitar"):
    """
    Hem MUSDB18-HQ (train/test alt klasörlü ve direkt wav'lı) 
    hem de MoisesDB (alt klasörlerde wav'lı) formatlarını otomatik algılar 
    ve modeli besleyeceğimiz standart formata dönüştürür.
    100 şarkının birbirine karışma (Cacophony) bug'ı giderilmiştir!
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    song_dirs = []
    
    # 1. AKILLI KLASÖR ALGILAMA SİSTEMİ
    # Eğer input_dir içinde "train" veya "test" varsa, şarkılar onların içindedir (MUSDB18-HQ Formatı)
    if os.path.exists(os.path.join(input_dir, "train")) or os.path.exists(os.path.join(input_dir, "test")):
        print(">> MUSDB18-HQ Klasör Yapısı Algılandı ('train' / 'test' alt klasörleri var).")
        for sub in ["train", "test"]:
            sub_path = os.path.join(input_dir, sub)
            if os.path.exists(sub_path):
                for d in os.listdir(sub_path):
                    full_path = os.path.join(sub_path, d)
                    if os.path.isdir(full_path):
                        song_dirs.append(full_path)
    else:
        # MoisesDB / Düz Klasör Yapısı
        print(">> MoisesDB / Düz Şarkı Klasör Yapısı Algılandı.")
        for d in os.listdir(input_dir):
            full_path = os.path.join(input_dir, d)
            if os.path.isdir(full_path):
                song_dirs.append(full_path)
                
    print(f"\nToplam {len(song_dirs)} şarkı klasörü tek tek işlenecek...\n")
    
    for idx, song_path in enumerate(song_dirs):
        song_name = os.path.basename(song_path)
        out_song_path = os.path.join(output_dir, f"song_{idx:03d}_{song_name.replace(' ', '_')}")
        
        if not os.path.exists(out_song_path):
            os.makedirs(out_song_path)
            
        # Şarkı klasöründeki wav dosyalarını bul (Hem direkt içinde hem de alt klasörlerinde arar)
        wavs_direct = glob.glob(os.path.join(song_path, "*.wav"))
        wavs_sub = glob.glob(os.path.join(song_path, "*", "*.wav"))
        all_wavs = wavs_direct + wavs_sub
        
        if len(all_wavs) == 0:
            continue
            
        print(f"[{idx+1}/{len(song_dirs)}] İşleniyor: {song_name}")
        
        mixture_audio = None
        target_audio = None
        sr = 44100
        
        # Eğer klasörde zaten orijinalinden "mixture.wav" varsa amelelik yapıp baştan mixlemeye gerek yok
        has_premixed_mixture = any(os.path.basename(w).lower() == "mixture.wav" for w in all_wavs)
        
        for wav_file in all_wavs:
            file_name = os.path.basename(wav_file).lower()
            parent_dir = os.path.basename(os.path.dirname(wav_file)).lower()
            
            # Ses dosyasını RAM'e al
            audio, sr = sf.read(wav_file, always_2d=True)
            
            # ------ TARGET (Hedef) AYRIŞTIRMA ------
            # İsimde veya bulunduğu alt klasörün isminde 'target_instrument' geçiyorsa
            if target_instrument.lower() in file_name or target_instrument.lower() in parent_dir:
                if target_audio is None:
                    target_audio = np.zeros_like(audio)
                min_len = min(target_audio.shape[0], audio.shape[0])
                target_audio = target_audio[:min_len] + audio[:min_len]
                
            # ------ MIXTURE (Miks) OLUŞTURMA ------
            if has_premixed_mixture:
                if file_name == "mixture.wav":
                    mixture_audio = audio
            else:
                # Orijinal miks yoksa, her kanalı (bateri, gitar, vokal) üst üste bindirerek mix yarat (MoisesDB)
                if mixture_audio is None:
                    mixture_audio = np.zeros_like(audio)
                min_len = min(mixture_audio.shape[0], audio.shape[0])
                mixture_audio = mixture_audio[:min_len] + audio[:min_len]

        # Hedef bulunamadıysa model çökmesin diye sessizlik ekle
        if target_audio is None:
            print(f"  --> Uyarı: Bu şarkıda '{target_instrument}' bulunamadı. Boş bir dosya oluşturuluyor.")
            if mixture_audio is not None:
                target_audio = np.zeros_like(mixture_audio)
            else:
                continue

        if mixture_audio is None:
            continue
            
        # Dosyaları diske yaz
        final_len = min(mixture_audio.shape[0], target_audio.shape[0])
        sf.write(os.path.join(out_song_path, "mixture.wav"), mixture_audio[:final_len], sr)
        sf.write(os.path.join(out_song_path, f"{target_instrument}.wav"), target_audio[:final_len], sr)

if __name__ == "__main__":
    # Veri setinizin olduğu yer (MUSDB18-HQ veya MoisesDB)
    GIRDI_KLASORU = r"C:\Users\jiyan\Desktop\mosie" 
    
    # Çıktıların (U-Net'in okuyacağı şekilde) kaydedileceği temiz klasör
    CIKTI_KLASORU = r"C:\Users\jiyan\Desktop\sonic-id\data\train_processed"
    
    # Hangi enstrümanı çekip çıkartmak istiyorsun? (Moises için "guitar", MUSDB18 için "other" tavsiye edilir)
    AYRILACAK_ENSTRUMAN = "guitar"  
    
    prepare_dataset(GIRDI_KLASORU, CIKTI_KLASORU, target_instrument=AYRILACAK_ENSTRUMAN)
    print("\n[BAŞARILI] İşlem tamam! Veriler artık kusursuzca ayrıştırıldı.")
