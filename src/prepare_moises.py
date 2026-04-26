import os
import soundfile as sf
import numpy as np
import glob

def prepare_moises_data(moises_dir, output_dir, target_instrument="vocals"):
    """
    MUSDB18-HQ formatındaki klasörleri (train ve test altındaki şarkıları), 
    SonicID modelinin eğitimde beklediği formata (mixture.wav ve other.wav) dönüştürür.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("MUSDB18-HQ veri seti taranıyor...")
    
    # İçinde '.wav' dosyası olan tüm klasörleri şarkı klasörü olarak bulalım
    song_dirs = []
    for root, dirs, files in os.walk(moises_dir):
        if any(f.endswith('.wav') for f in files):
            song_dirs.append(root)
            
    song_dirs = sorted(song_dirs)
    print(f"Toplam {len(song_dirs)} şarkı bulundu. Dönüştürülüyor...")
    
    for idx, song_path in enumerate(song_dirs):
        song_name = os.path.basename(song_path)
        out_song_path = os.path.join(output_dir, f"song_{idx:03d}")
        
        if not os.path.exists(out_song_path):
            os.makedirs(out_song_path)
            
        all_wavs = glob.glob(os.path.join(song_path, "*.wav"))
        if len(all_wavs) == 0:
            continue
            
        print(f"[{idx+1}/{len(song_dirs)}] İşleniyor: {song_name}")
        
        mixture_audio = None
        target_audio = None
        sr = 44100
        
        has_mixture = any(os.path.basename(w).lower() in ["mixture.wav", "mix.wav"] for w in all_wavs)
        
        for wav_file in all_wavs:
            file_name = os.path.basename(wav_file).lower()
            
            if target_instrument.lower() in file_name:
                target_audio, sr = sf.read(wav_file, always_2d=True)
                
            if has_mixture and file_name in ["mixture.wav", "mix.wav"]:
                mixture_audio, sr = sf.read(wav_file, always_2d=True)
            elif not has_mixture:
                audio, sr = sf.read(wav_file, always_2d=True)
                if mixture_audio is None:
                    mixture_audio = np.zeros_like(audio)
                min_len = min(mixture_audio.shape[0], audio.shape[0])
                mixture_audio = mixture_audio[:min_len]
                mixture_audio += audio[:min_len]
                
        if target_audio is None:
            print(f"  Uyarı: '{target_instrument}' bulunamadı. Boş dosya oluşturuluyor.")
            if mixture_audio is not None:
                target_audio = np.zeros_like(mixture_audio)
            else:
                continue
                
        if mixture_audio is None:
            continue
            
        min_len = min(mixture_audio.shape[0], target_audio.shape[0])
        mixture_audio = mixture_audio[:min_len]
        target_audio = target_audio[:min_len]
            
        sf.write(os.path.join(out_song_path, "mixture.wav"), mixture_audio, sr)
        sf.write(os.path.join(out_song_path, "other.wav"), target_audio, sr)

if __name__ == "__main__":
    MOISES_KLASORU = "/workspace/sonic-id/data/moises"
    HEDEF_KLASOR = "/workspace/sonic-id/data/train"
    AYRILACAK_ENSTRUMAN = "vocals"
    
    prepare_moises_data(MOISES_KLASORU, HEDEF_KLASOR, target_instrument=AYRILACAK_ENSTRUMAN)
    print("İşlem tamam! Veriler artık dataset.py'nin okuyabileceği formata çevrildi.")
