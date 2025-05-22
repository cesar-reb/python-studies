import json
import os
import re
import spacy
import shutil
from spacy.training.example import Example
from spacy.util import minibatch
import easyocr
from datetime import datetime

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_text_easyocr(image_path, reader):
    results = reader.readtext(image_path)
    lines = [text for (_, text, prob) in results if prob > 0.5]
    return clean_text(" ".join(lines))

def load_training_data(image_folder, index_json_path, reader):
    with open(index_json_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    training_data = []

    for filename, (start, end) in index_data.items():
        image_path = os.path.join(image_folder, filename)
        text = extract_text_easyocr(image_path, reader)
        entities = [(start, end, "DUE_DATE")]
        training_data.append((text, {"entities": entities}))
    
    return training_data

def backup_model_dir(model_dir, backup_base_dir, max_backups=5):
    if not os.path.exists(model_dir):
        return 
    
    os.makedirs(backup_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_base_dir, f"backup_{timestamp}")
    shutil.copytree(model_dir, backup_path)
    print(f"Backup created at: {backup_path}")

    backups = sorted(
        [os.path.join(backup_base_dir, d) for d in os.listdir(backup_base_dir)],
        key=os.path.getctime
    )

    while len(backups) > max_backups:
        oldest = backups.pop(0)
        shutil.rmtree(oldest)
        print(f"Old backups removed: {oldest}")

def train_spacy_ner(training_data, output_dir):
    nlp = spacy.load(output_dir)
    ner = nlp.get_pipe("ner")

    if "DUE_DATE" not in ner.labels:
        ner.add_label("DUE_DATE")

    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for i in range(20):
            losses = {}
            batches = minibatch(training_data, size=2)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annotations))
                nlp.update(examples, drop=0.2, losses=losses)
            print(f"Epoch {i+1} - Losses: {losses}")

    nlp.to_disk(output_dir)
    print(f"Model saved at: {output_dir}")

def main():
    image_folder = "./training_imgs"
    json_path = "./due_date_references.json"
    output_dir = "./trained_ner_model"
    backup_base_dir = "./trained_ner_model_backups"

    backup_model_dir(output_dir, backup_base_dir, max_backups=5)

    reader = easyocr.Reader(['pt'])
    training_data = load_training_data(image_folder, json_path, reader)
    train_spacy_ner(training_data, output_dir)

if __name__ == "__main__":
    main()
