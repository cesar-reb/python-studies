import os
import json
import re
import easyocr
from PIL import Image
import spacy

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_text_easyocr(image_path: str, reader: easyocr.Reader):
    results = reader.readtext(image_path)
    lines = [text for (_, text, prob) in results if prob > 0.5]
    return clean_text(" ".join(lines))

def evaluate_model(model_path, image_folder):
    nlp = spacy.load(model_path)
    reader = easyocr.Reader(['pt'])
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        print(f"Avaliando: {image_name}")

        text = extract_text_easyocr(image_path, reader)
        doc = nlp(text)

        print("Texto OCR extraído:")
        print(text)

        print("Entidades encontradas:")
        found_due_date = False
        for ent in doc.ents:
            label = ent.label_
            print(f"  - {ent.text} [{label}]")
            if label == "DUE_DATE":
                found_due_date = True

        if not found_due_date:
            print("Nenhuma DUE_DATE encontrada!")
        else:
            print("DUE_DATE identificada!")

if __name__ == "__main__":
    evaluate_model("./trained_ner_model", "./training_imgs")