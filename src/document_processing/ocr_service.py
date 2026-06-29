import logging

logger = logging.getLogger(__name__)

_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            # Disable GPU for lightweight CPU execution (stable on standard hardware)
            _reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    return _reader

def extract_text_from_image(image_data) -> str:
    """
    Extract text from image path, bytes, or numpy array using local EasyOCR.
    """
    try:
        reader = get_ocr_reader()
        results = reader.readtext(image_data, detail=0)
        return " ".join(results).strip()
    except Exception as e:
        logger.error(f"Local EasyOCR text extraction failed: {e}")
        return ""
