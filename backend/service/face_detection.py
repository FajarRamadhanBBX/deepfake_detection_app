from mtcnn import MTCNN

detector = MTCNN(device='CPU')

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
        
        # Center-align untuk crop foto
        center_x, center_y = x + w // 2, y + h // 2
        
        # Hitung koordinat baru untuk crop persegi
        x_new = max(center_x - size // 2, 0)
        y_new = max(center_y - size // 2, 0)
        
        # Potong wajah dari gambar asli
        # Pastikan area potongan tidak melebihi batas gambar
        face_square = image_array[y_new:y_new+size, x_new:x_new+size]
        
        return face_square
        
    return None