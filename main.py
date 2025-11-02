import cv2
import numpy as np

def yuz_sansurle(frame, yuzler, sansur_tipi='blur'):
    """
    Tespit edilen yüzleri sansürler
    
    Args:
        frame: Görüntü frame'i
        yuzler: Tespit edilen yüz koordinatları
        sansur_tipi: 'blur' veya 'pixelate'
    """
    for (x, y, w, h) in yuzler:
        
        yuz_bolgesi = frame[y:y+h, x:x+w]
        
        if sansur_tipi == 'blur':
            
            blurlenmis = cv2.GaussianBlur(yuz_bolgesi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurlenmis
            
        elif sansur_tipi == 'pixelate':
           
            
            kucuk = cv2.resize(yuz_bolgesi, (w//15, h//15), interpolation=cv2.INTER_LINEAR)
            
            piksel = cv2.resize(kucuk, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = piksel
    
    return frame


def main():
    print("🎥 Yüz Sansürleme Programı Başlatılıyor...")
    print("=" * 50)
    
    
    yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    
    kamera = cv2.VideoCapture(0)
    
    if not kamera.isOpened():
        print("❌ HATA: Kamera açılamadı!")
        print("Lütfen kameranızın bağlı olduğundan emin olun.")
        return
    
    print("✅ Kamera başarıyla açıldı!")
    print("\n📋 Kontroller:")
    print("  - ESC: Çıkış")
    print("  - 'b': Blur modu")
    print("  - 'p': Piksel modu")
    print("  - 's': Ekran görüntüsü kaydet")
    print("=" * 50)
    
    sansur_tipi = 'blur'
    
    while True:
        
        ret, frame = kamera.read()
        
        if not ret:
            print("❌ Frame okunamadı!")
            break
        
       
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
       
        yuzler = yuz_cascade.detectMultiScale(
            gri,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # sansür
        frame = yuz_sansurle(frame, yuzler, sansur_tipi)
        
        
        cv2.putText(frame, f"Tespit: {len(yuzler)} yuz | Mod: {sansur_tipi.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
       
        cv2.imshow('Yuz Sansurleme - ESC ile cikis', frame)
        
        
        tus = cv2.waitKey(1) & 0xFF
        
        if tus == 27: 
            print("\n👋 Program kapatılıyor...")
            break
        elif tus == ord('b'):
            sansur_tipi = 'blur'
            print("✨ Blur modu aktif")
        elif tus == ord('p'):
            sansur_tipi = 'pixelate'
            print("🔲 Piksel modu aktif")
        elif tus == ord('s'):
            dosya_adi = f"sansurlu_goruntu_{np.random.randint(1000, 9999)}.jpg"
            cv2.imwrite(dosya_adi, frame)
            print(f"📸 Görüntü kaydedildi: {dosya_adi}")
    
    
    kamera.release()
    cv2.destroyAllWindows()
    print("✅ Kamera kapatıldı. Görüşmek üzere!")


if __name__ == "__main__":
    main()
