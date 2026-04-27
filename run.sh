#!/bin/bash
# Sonic ID - RunPod Eğitim Başlatıcı
# Kullanım: bash run.sh
# Resume için: bash run.sh --resume

CHECKPOINT_DIR="/workspace/sonic-id/checkpoints"
DATA_PATH="/workspace/sonic-id/data/train"
TARGET_FILE="other.wav"
EPOCHS=200

cd /workspace/sonic-id

if [ "$1" == "--resume" ]; then
    # Son tamamlanan epoch'u otomatik bul
    LAST_CKPT=$(ls -v $CHECKPOINT_DIR/sonic_id_epoch_*.pth 2>/dev/null | tail -1)

    if [ -z "$LAST_CKPT" ]; then
        echo "[!] Checkpoint bulunamadi, sifirdan basliyor..."
        python src/train.py \
            --data_path $DATA_PATH \
            --target_file $TARGET_FILE \
            --epochs $EPOCHS
    else
        # Epoch numarasini dosya adindan cek
        LAST_EPOCH=$(echo $LAST_CKPT | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
        START_EPOCH=$((LAST_EPOCH + 1))
        echo "[*] Son checkpoint: $LAST_CKPT (Epoch $LAST_EPOCH)"
        echo "[*] Epoch $START_EPOCH'den devam ediliyor..."
        python src/train.py \
            --data_path $DATA_PATH \
            --target_file $TARGET_FILE \
            --epochs $EPOCHS \
            --resume $LAST_CKPT \
            --start_epoch $START_EPOCH
    fi
else
    echo "[*] Sifirdan basliyor..."
    python src/train.py \
        --data_path $DATA_PATH \
        --target_file $TARGET_FILE \
        --epochs $EPOCHS
fi
