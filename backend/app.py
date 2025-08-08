import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from face_detection import crop_face_square
from image_processor import decode_image_from_bytes, convert_bgr_to_rgb, convert_rgb_to_bgr, encode_image_to_jpg

# Safety Configuration
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Detection Test API!"}

@app.post("/face_detection")
async def face_detection(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."
            )
        
        image = decode_image_from_bytes(contents)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        height, width, _ = image.shape
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions exceed the maximum limit of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels."
            )

        image_rgb = convert_bgr_to_rgb(image)

        cropped_face = crop_face_square(image_rgb)
        if cropped_face is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image"
            )
        
        cropped_face_bgr = convert_rgb_to_bgr(cropped_face)

        cropped_face_result = encode_image_to_jpg(cropped_face_bgr)
        if cropped_face_result is None:
            raise HTTPException(
                status_code=400,
                detail="Could not encode cropped face image"
            )

        return Response(content=cropped_face_result, media_type="image/jpeg")

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal server error occurred."
        )