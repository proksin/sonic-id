import os
import soundfile as sf
import numpy as np
import glob

def prepare_dataset(input_dir, output_dir, target_instrument="guitar"):
    """
    Hem MUSDB18-HQ hem de MoisesDB formatlarını, klasör yapısı ne kadar derin olursa olsun
    otomatik algılar ve modeli besleyeceğimiz standart formata dönüştürür.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(">> Veri setindeki tüm ses dosyaları derinlemesine taranıyor (Recursive Search)...")
    
    # 1. KUSURSUZ KLASÖR ALGILAMA SİSTEMİ
    # Klasör derinliği ne olursa olsun içindeki tüm .wav dosyalarını X-Ray gibi bulur.
    all_wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    if not all_wav_files:
        print("HATA: Belirtilen klasörde hiçbir .wav dosyası bulunamadı!")
        return

    # İçinde .wav barındıran benzersiz klasörleri (yani şarkıların ana dizinlerini) tespit et
    unique_dirs = list(set(os.path.dirname(w) for w in all_wav_files))
    
    song_dirs = []
    # Sadece içinde birden fazla .wav olanları "şarkı" olarak kabul et (hedef + miks veya diğer stemler)
    for d in unique_dirs:
        wavs_in_dir = glob.glob(os.path.join(d, "*.wav"))
        if len(wavs_in_dir) >= 2:
            song_dirs.append(d)
            
    # Her çalıştırmada aynı sırayla işlemesi için klasörleri alfabetik diz
    song_dirs = sorted(song_dirs)
    
    print(f"\nToplam {len(song_dirs)} şarkı klasörü bulundu ve tek tek işlenecek...\n")
    
    for idx, song_path in enumerate(song_dirs):
        song_name = os.path.basename(song_path)
        out_song_path = os.path.join(output_dir, f"song_{idx:03d}_{song_name.replace(' ', '_')}")
        
        if not os.path.exists(out_song_path):
            os.makedirs(out_song_path)
            
        all_wavs = glob.glob(os.path.join(song_path, "*.wav"))
        
        print(f"[{idx+1}/{len(song_dirs)}] İşleniyor: {song_name} ({len(all_wavs)} stem bulundu)")
        
        mixture_audio = None
        target_audio = None
        sr = 44100
        
        # Eğer klasörde zaten orijinalinden "mixture.wav" varsa baştan mixlemeye gerek yok
        has_premixed_mixture = any(os.path.basename(w).lower() == "mixture.wav" for w in all_wavs)
        
        for wav_file in all_wavs:
            file_name = os.path.basename(wav_file).lower()
            parent_dir = os.path.basename(os.path.dirname(wav_file)).lower()
            
            # Ses dosyasını RAM'e al
            audio, sr = sf.read(wav_file, always_2d=True)
            
            # ------ TARGET (Hedef) AYRIŞTIRMA ------
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
                # Orijinal miks yoksa, her kanalı üst üste bindirerek mix yarat (MoisesDB)
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
    
    # Hangi enstrümanı çekip çıkartmak istiyorsun?
    AYRILACAK_ENSTRUMAN = "guitar"  
    
    prepare_dataset(GIRDI_KLASORU, CIKTI_KLASORU, target_instrument=AYRILACAK_ENSTRUMAN)
    print("\n[BAŞARILI] İşlem tamam! Veriler artık kusursuzca ayrıştırıldı.")