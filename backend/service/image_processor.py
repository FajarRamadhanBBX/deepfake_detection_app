import numpy as np
import cv2

def decode_image_from_bytes(contents) -> np.ndarray | None:
    """"
    Melakukan decoding dari byte gambar menjadi array NumPy.
    Jika decoding gagal, mengembalikan None.
    """
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img

def convert_bgr_to_rgb(image) -> np.ndarray:
    """
    Mengonversi gambar dari format BGR ke RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_rgb_to_bgr(image) -> np.ndarray:
    """
    Mengonversi gambar dari format RGB ke BGR.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def encode_image_to_jpg(image) -> bytes | None:
    """
    Mengonversi gambar array NumPy ke format JPEG.
    Jika encoding gagal, mengembalikan None.
    """
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return None
    return buffer.tobytes()