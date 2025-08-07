from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response # <--- DIUBAH: Gunakan Response untuk gambar
import numpy as np
import io
from PIL import Image
import cv2
from mtcnn import MTCNN

# --- Inisialisasi Model di Awal ---
# Model dimuat sekali saat aplikasi dimulai, bukan setiap ada request
detector = MTCNN(device='CPU')

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Detection Test API!"}

@app.post("/face_detection")
async def face_detection(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # 1. Baca file yang diupload sebagai bytes
        contents = await file.read()
        
        # 2. Ubah bytes menjadi array NumPy yang bisa dibaca cv2
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Jika gambar tidak berhasil di-decode
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        # 3. Konversi warna dari BGR (OpenCV) ke RGB (MTCNN)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4. Deteksi dan potong wajah
        cropped_face = crop_face_square(img_rgb) # <--- DIUBAH: Kirim array gambar, bukan path
        
        if cropped_face is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # 5. Konversi balik hasil potongan (RGB) ke BGR untuk encoding cv2
        cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # 6. Encode gambar hasil potongan ke format JPEG di memori
        is_success, buffer = cv2.imencode(".jpg", cropped_face_bgr)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode cropped image")

        # 7. Kembalikan gambar sebagai response dengan media type yang benar
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except HTTPException as e:
        # Re-raise HTTPException agar FastAPI menanganinya
        raise e
    except Exception as e:
        # Tangani error lainnya
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Fungsi Helper yang Sudah Diperbaiki ---
def crop_face_square(image_array): # <--- DIUBAH: Menerima array gambar, bukan path
    """
    Mendeteksi wajah dari array gambar NumPy (format RGB) dan mengembalikannya
    sebagai potongan gambar persegi.
    """
    result = detector.detect_faces(image_array)
    
    if result:
        # Ambil wajah pertama yang paling confidence
        x, y, w, h = result[0]['box']
        
        # Pastikan koordinat tidak negatif
        x, y = max(x, 0), max(y, 0)

        # Tentukan sisi terpanjang untuk membuat kotak persegi
        size = max(w, h)
        
        # Center-align the square crop
        center_x, center_y = x + w // 2, y + h // 2
        
        # Hitung koordinat baru untuk crop persegi
        x_new = max(center_x - size // 2, 0)
        y_new = max(center_y - size // 2, 0)
        
        # Potong wajah dari gambar asli
        # Pastikan area potongan tidak melebihi batas gambar
        face_square = image_array[y_new:y_new+size, x_new:x_new+size]
        
        return face_square
        
    return None