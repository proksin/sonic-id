import os
import soundfile as sf
import numpy as np
import glob
import argparse

def prepare_universal_dataset(input_dir, output_dir, target_instrument="gitar"):
    """
    Hem MUSDB18 (train/test alt klasörlü) hem de MoisesDB (iç içe klasörlü) 
    veri setlerini anlayan ve modelin istediği formata çeviren AKILLI betik.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[*] {input_dir} dizinindeki tüm şarkılar taranıyor...")
    
    song_dirs = []
    # os.walk ile tüm alt klasörlere (train, test vs) in
    for root, dirs, files in os.walk(input_dir):
        # İçinde birden fazla .wav dosyası barındıran veya alt klasörlerinde .wav olan 
        # en üst şarkı kök klasörünü bulmamız lazım.
        wav_files = glob.glob(os.path.join(root, "*.wav"))
        sub_wav_files = glob.glob(os.path.join(root, "*", "*.wav"))
        
        # Eğer bu klasör bir şarkının ana klasörüyse (örn: "Song 1")
        if len(wav_files) >= 2 or len(sub_wav_files) >= 2:
            # Sadece bir kere eklemek için kontrol
            if not any(root.startswith(existing) for existing in song_dirs):
                song_dirs.append(root)
            
    print(f"[+] Toplam {len(song_dirs)} benzersiz şarkı klasörü bulundu. Dönüştürülüyor...\n")
    
    for idx, song_path in enumerate(song_dirs):
        song_folder_name = os.path.basename(song_path)
        out_song_path = os.path.join(output_dir, f"song_{idx:03d}_{song_folder_name.replace(' ', '_')}")
        
        if not os.path.exists(out_song_path):
            os.makedirs(out_song_path)
            
        print(f"[{idx+1}/{len(song_dirs)}] İşleniyor: {song_folder_name}")
        
        # Klasörün içindeki ve bir altındaki tüm wav dosyalarını bul
        all_wavs = glob.glob(os.path.join(song_path, "**", "*.wav"), recursive=True)
        
        # Eğer orijinal dataset zaten "mixture.wav" verdiyse, sesleri amele gibi tekrar toplamamıza gerek yok!
        has_mixture = any(os.path.basename(w).lower() == "mixture.wav" for w in all_wavs)
        
        mixture_audio = None
        target_audio = None
        sr = 44100
        
        for wav_file in all_wavs:
            file_name = os.path.basename(wav_file).lower()
            parent_dir = os.path.basename(os.path.dirname(wav_file)).lower()
            
            # Eğer halihazırda mixture.wav varsa, onu sadece oku ve mix'e ekleme
            if has_mixture and file_name == "mixture.wav":
                mixture_audio, sr = sf.read(wav_file, always_2d=True)
                continue
                
            audio, sr = sf.read(wav_file, always_2d=True)
            
            # Eğer mixture.wav yoksa (MoisesDB gibi), tüm parçaları üst üste bindirip miksleyeceğiz
            if not has_mixture:
                if mixture_audio is None:
                    mixture_audio = np.zeros_like(audio)
                min_len = min(mixture_audio.shape[0], audio.shape[0])
                mixture_audio = mixture_audio[:min_len]
                audio_adj = audio[:min_len]
                mixture_audio += audio_adj
                
            # İstenen hedef enstrüman (örn: gitar.wav, other.wav veya klasör adı 'guitar' ise)
            if target_instrument in file_name or target_instrument in parent_dir:
                if target_audio is None:
                    target_audio = np.zeros_like(audio)
                min_len = min(target_audio.shape[0], audio.shape[0])
                target_audio = target_audio[:min_len]
                audio_adj = audio[:min_len]
                target_audio += audio_adj
                
        # Eğer o şarkıda hiç gitar/hedef yoksa, boş bir wav yarat (Hata vermesin diye)
        if target_audio is None:
            print(f"  [!] Uyarı: '{target_instrument}' bulunamadı. Boş dosya oluşturuluyor.")
            if mixture_audio is not None:
                target_audio = np.zeros_like(mixture_audio)
            else:
                target_audio = np.zeros((44100, 2))
                
        # Disk'e yaz!
        if mixture_audio is not None:
            sf.write(os.path.join(out_song_path, "mixture.wav"), mixture_audio, sr)
            
        sf.write(os.path.join(out_song_path, f"{target_instrument}.wav"), target_audio, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evrensel Dataset Hazırlayıcı (Moises & MUSDB18)")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="İndirdiğin ham dataset klasörü")
    parser.add_argument("--output_dir", type=str, default="data/train", help="Eğitime girecek temiz formatın çıkacağı yer")
    parser.add_argument("--target", type=str, default="gitar", help="Ayırmak istediğin enstrüman adı (gitar, other, vocals vs.)")
    
    args = parser.parse_args()
    
    prepare_universal_dataset(args.input_dir, args.output_dir, args.target)
    print("\n[BAŞARILI] İşlem tamam! Veriler artık train.py'nin okuyabileceği formata çevrildi.")
