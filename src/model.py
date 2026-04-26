import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    U-Net mimarisinde her katmanda tekrar eden iki ardışık Evrişim -> Normalizasyon -> LeakyReLU bloğu.
    Boyutların (Height x Width) korunması için padding=1 kullanıyoruz.
    Phase 1.5: 'ReLU' yerine 'LeakyReLU' kullanılarak susturulan titreşimlerdeki 'Ölü Nöron'
    (Gradient ölümü) ihtimali ortadan kaldırıldı; davullar silinirken cılız gitar tıngırtıları kaybolmayacak.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Sonic ID (Phase 1.5) - Gelişmiş Saf Gitar Modeli (Artifact / Çınlama Önleyici)
    Mikslenmiş ses spektrogramını alıp maske (mask) üretir. 
    İçerisinde Soft-Masking (Yumuşak Geçiş) algoritması ve daha derin özellik katmanları barındırır.
    """
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512, 1024]):
        # Model 1 katman daha derinleştirildi (1024'e kadar) -> Metallica (Distortion) gibi 
        # frekansı aşırı karmaşık müzikleri daha iyi hazmedebilmesi için.
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ENCODER (Aşağı Örnekleme / Downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # DECODER (Yukarı Örnekleme / Upsampling - Skip Bağlantılarıyla Birlikte)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # BOĞAZ (Bottleneck) Katmanı
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # ÇIKIŞ KATMANI: Sigmoid maskesi. Değerler [0, 1] arası.
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # [YENİ]: Soft-Masking Bulanıklaştırması (Çınlama / Birdies Giderici)
        # Bıçak gibi kesen frekans uçurumlarını blur'layarak organik miks (ve iSTFT esnekliği) yaratır.
        self.smooth_mask = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Gelişmiş Auto-Padding (32'nin Katı) ile İleri Sürüm (Forward Pass).
        """
        B, C, H, W = x.size()
        
        # 1. Adım: x ve y ekseninde 32'nin katı boyuta tamamlamak için Padding
        # U-Net'te 5 kez (önceki 4 idi) 2x2 pooling var, yani boyutların 2^5 = 32'ye tam bölünmesi şart.
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        
        # Formüller: (left, right, top, bottom)
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))

        skip_connections = []
        out = x_padded

        # Encoder (Özellik Çıkarımı)
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Skip listesini baştan sona (Decoder akışına göre) ters çeviriyoruz.
        skip_connections = skip_connections[::-1]

        # Decoder (Geri İnşa Süreci)
        for idx in range(0, len(self.ups), 2):
            out = self.ups[idx](out)   # ConvTranspose2d ile Boyut Büyütme
            skip_connection = skip_connections[idx//2]
            
            # Dinamik boyut uyarlaması
            if out.shape != skip_connection.shape:
                out = F.interpolate(out, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
                
            # Özellik Haritalarının Birleştirilmesi (Concatenate)
            out = torch.cat((skip_connection, out), dim=1)
            out = self.ups[idx+1](out) # İkili Evrişim (DoubleConv)

        # Ham (Keskin) Maskenin üretilmesi
        raw_mask = self.final_conv(out)
        
        # [Phase 1.5 YAMASI] Soft-Masking: Frekansları çınlamaması için organikleştir (Blur)
        smoothed_mask = self.smooth_mask(raw_mask)
        
        # 2. Adım: Yapısını korumak adına başta eklediğimiz pad kısımlarını makaslıyoruz.
        final_mask = smoothed_mask[:, :, :H, :W]

        return final_mask

# Kendi kendini test edebilmesi için minik bir blok:
if __name__ == "__main__":
    # Test için mock/dummy tensör = Batch:1, Kanal:4 (Miks + Referans), Frekans: 513, Zaman: 259
    test_input = torch.randn((1, 4, 513, 259)) 
    model = UNet(in_channels=4, out_channels=2)
    output_mask = model(test_input)
    
    print(f"[Phase 1.5 TEST BAŞARILI] Girdi: {test_input.shape} | Soft-Mask Çıktısı: {output_mask.shape}")
