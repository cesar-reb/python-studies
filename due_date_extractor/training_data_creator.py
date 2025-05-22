import os
import json
import re
import easyocr
import pdfplumber

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages[:1]:  # só primeira página
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return clean_text(text)
    except Exception as e:
        print(f"Erro ao ler PDF {pdf_path}: {e}")
        return ""

def extract_text_from_image(image_path, reader: easyocr.Reader):
    results = reader.readtext(image_path)
    lines = [text for (_, text, prob) in results if prob > 0.5]
    return clean_text(" ".join(lines))

def main():
    training_source = "./training_imgs"
    json_path = "./due_date_references.json"
    reader = easyocr.Reader(['pt'])

    # Carrega JSON existente (se houver)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    files = [f for f in os.listdir(training_source) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

    for file in files:
        if file in data:
            print(f"Já processado: {file}")
            continue

        path = os.path.join(training_source, file)

        print(f"Processando: {file}")
        if file.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            text = extract_text_from_image(path, reader)

        if not text.strip():
            print("Nenhum texto extraído. Pulando...")
            continue

        print("📝 Texto extraído:")
        print("=" * 60)
        print(text)
        print("=" * 60)

        target = input("Digite exatamente a data de vencimento visível no texto (ou ENTER para pular): ").strip()

        if not target:
            print("Pulando...")
            continue

        if target not in text:
            print(f"'{target}' não encontrado no texto. Pulando...")
            continue

        start = text.index(target)
        end = start + len(target)
        data[file] = [start, end]

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Índices salvos em: {json_path}")

if __name__ == "__main__":
    main()
