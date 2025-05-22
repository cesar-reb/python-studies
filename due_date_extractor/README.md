# DUE_DATE Extractor

NER (Named Entity Recognition) model training for detecting due dates in documents.

Documents are processed using OCR (Optical Character Recognition) for image-based files, or extracted directly from text-based formats (such as PDFs).

```
pip install spacy easyocr opencv-python-headless
# initial model -> python -m spacy download pt_core_news_sm
