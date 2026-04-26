import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SonicID_Dataset
from model import UNet

def train_sonic_id():
    """
    Sonic ID (Phase 1) - U-Net Model Eğitimi 
    Amacımız elektro gitar (veya ayarlanan target) ağırlıklarını ayıracak 
    maskeyi modelin öğrenmesini sağlamak. 
    İleride özel veri setiyle Fine-Tune çalışmasına hazır (esnek) yazılmıştır.
    """
    # ------------------ HIPERPARAMETRELER ------------------
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WINDOW_SIZE = 3.0
    
    # Veri setimiz nerede? (dataset.py'daki yol ile uyumlu olmalı)
    DATA_PATH = r"C:\Users\jiyan\Desktop\sonic-id\data\train" 
    
    # Kaydedilecek weights noktaları
    CHECKPOINT_DIR = "../checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # -------------------------------------------------------

    # 1. Cihaz (Hardware) Tercihi: CUDA (Nvidia), MPS (Mac), CPU (Diğer)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(">> Eğitim [CUDA - NVIDIA GPU] ile başlayacak. Süper hızlı olacak.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">> Eğitim [MPS - APPLE SILICON] ile başlayacak. (M1/M2/M3).")
    else:
        device = torch.device("cpu")
        print(">> UYARI! GPU Bulunamadı! Eğitim CPU ile devam edecek. İşlem uzun sürebilir.")

    # 2. Veri Yükleyici (DataLoader) 
    print(f"[{DATA_PATH}] konumundan parçalar RAM'e yükleniyor...")
    try:
        dataset = SonicID_Dataset(root_dir=DATA_PATH, window_size=WINDOW_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    except Exception as e:
        print(f"HATA! Veri seti yüklenemedi. Yol (Path) doğru mu? Detay: {e}")
        return

    print(f"Harika! Eğitim verisi hazır. Toplam klasör: {len(dataset)}")

    # 3. Model ve Motorlar (Optimizasyon vs)
    model = UNet(in_channels=4, out_channels=2).to(device)
    
    import torchaudio.functional as F_audio
    
    # SNR Loss'u desteklemek ve çok ince cızırtıları sıfırlamak için küçük bir L1 destekleyici kenarda duracak.
    criterion_l1 = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Eğitim Döngüsü (Training Loop)
    print("\n---------- EĞİTİM (TRAIN) BAŞLIYOR ----------")
    
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (mix_spec, reference_spec, target_spec) in enumerate(dataloader):
            # Tensörleri GPU'ya / CPU'ya at
            mix_spec = mix_spec.to(device)
            reference_spec = reference_spec.to(device)
            target_spec = target_spec.to(device)
            
            # Girdi 1 (Mix) ve Girdi 2 (Referans) birleştirilir (Concatenation)
            combined_input = torch.cat([mix_spec, reference_spec], dim=1)
            
            # Forward Pass: Model 4 kanala bakıp miks üzerinden bir Maske (0-1 arası) üretiyor.
            mask = model(combined_input)
            
            # Üretilen maskeyi miksin genliğiyle çarparak kendi ayrıştırdığı vizyonu elde ediyoruz
            isolated_spec = mix_spec * mask
            
            # Gerçekte (target) olması gereken spektrograma ne kadar benziyor? Hesapla!
            # [YENİ] L1 Loss yerine Makaledeki Scale-Invariant SNR (dB Loss) kullanıyoruz (Denklem 2).
            # Model piksellerin rengini değil, resmin içindeki "Ses Enerjisini (Akustik Güç)" çözmek zorunda bırakılıyor.
            B = isolated_spec.shape[0]
            preds_flat = isolated_spec.view(B, -1)
            targets_flat = target_spec.view(B, -1)
            
            # SI-SNR (Makale Denklem 2) - Manuel (Manuel hesaplama kütüphane bağımsızdır)
            eps = 1e-8
            target_energy = torch.sum(targets_flat ** 2, dim=-1, keepdim=True)
            alpha = torch.sum(preds_flat * targets_flat, dim=-1, keepdim=True) / (target_energy + eps)
            s_target = alpha * targets_flat
            e_noise = preds_flat - s_target
            
            s_target_energy = torch.sum(s_target ** 2, dim=-1)
            e_noise_energy = torch.sum(e_noise ** 2, dim=-1)
            si_snr_db = 10 * torch.log10(s_target_energy / (e_noise_energy + eps) + eps)
            
            # Negatife alıp Loss olarak optimize ediyoruz
            loss_snr = -torch.mean(si_snr_db) 
            
            # Çok ufak (0.1) bir L1 pürüzleri temizler (Sparsity)
            loss_l1 = criterion_l1(isolated_spec, target_spec)
            
            loss = loss_snr + (0.1 * loss_l1)
            
            # Backward Pass (Zamanda Geri Sarıp Hatayı Düzeltme)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Görsel takip (Terminalde)
            if batch_idx % 5 == 0:
                print(f"Epoch: [{epoch}/{EPOCHS}] | Batch: [{batch_idx}/{len(dataloader)}] | Loss (Hata Payı): {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"=== EPOCH {epoch} TAMAMLANDI === | Ortalama Loss: {avg_loss:.4f}\n")
        
        # 5. Her epoch'ta modeli kaydet (Bulut sunucuda yer problemi olmayacağı için)
        model_save_path = os.path.join(CHECKPOINT_DIR, f"sonic_id_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"[*] Model kaydedildi -> {model_save_path}")
        
        # En iyi modeli de ayrıca güncelleyelim
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_save_path = os.path.join(CHECKPOINT_DIR, "sonic_id_best_model.pth")
            torch.save(model.state_dict(), best_model_save_path)
            print(f"[+] Yeni EN İYİ loss seviyesi! (Loss: {best_loss:.4f})")

    print("=== TAMAMLANDI: Otonom EQ Kalibrasyonu için H-RACE modülüne gönderilmeye hazır. ===")

if __name__ == "__main__":
    train_sonic_id()
