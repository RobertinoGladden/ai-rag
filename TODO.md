# Bug Fix TODO

## Issues (Original)
1. Chroma 500 error after clear vector store + re-upload
2. CV OCR "not enough values to unpack (expected 3, got 2)"
3. CV OCR frontend sends query param instead of JSON body
4. CV Classify frontend response parsing mismatch
5. Missing BaseModel import in cv_module/src/api/routes.py

## Steps (Original)

- [x] 1. Fix `rag_pipeline/src/retrieval/vector_store.py` — use `reset_collection()` instead of destructive `delete_collection()`
- [x] 2. Fix `cv_module/src/processors/ocr_processor.py` — handle both 2-value and 3-value tuples from EasyOCR
- [x] 3. Fix `cv_module/src/api/routes.py` — add missing `BaseModel` import for inline `OCRRequest`
- [x] 4. Fix `frontend/index.html` — fix OCR request format and classify response parsing

## OCR Optimization

- [x] 5. Add image preprocessing pipeline (grayscale, CLAHE contrast, sharpening, denoising)
- [x] 6. Add adaptive binary thresholding for banner/sign text
- [x] 7. Implement multi-pass OCR (original + enhanced + binary) with result merging
- [x] 8. Tune EasyOCR parameters (beamsearch decoder, contrast_ths, text_threshold, etc.)
- [x] 9. Add smart text line reconstruction from bounding box positions
- [x] 10. Add deduplication logic for merged multi-pass results
- [ ] 11. Rebuild and test
