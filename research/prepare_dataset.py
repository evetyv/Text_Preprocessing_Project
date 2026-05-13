import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def labels_to_chunks(labels):
    """
    Преобразует список меток (1 - начало нового чанка, 0 - продолжение)
    в список чанков (списки индексов предложений).
    """
    chunks = []
    current_chunk = [0]
    for i in range(1, len(labels)):
        if labels[i] == 1:
            chunks.append(current_chunk)
            current_chunk = [i]
        else:
            current_chunk.append(i)
    chunks.append(current_chunk)
    return chunks

def compute_embeddings(model, sentences):
    """Вычисляет эмбеддинги для списка предложений."""
    return model.encode(sentences, show_progress_bar=False)

def main():
    Path("data").mkdir(exist_ok=True)
    
    print("Загрузка датасета RuTextSegNews (test split)...")
    dataset = load_dataset("mlenjoyneer/RuTextSegNews", split="test")
    print(f"Загружено документов: {len(dataset)}")
    
    # Инициализируем модели
    print("Загрузка модели MiniLM...")
    model_minilm = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Загрузка модели rubert-tiny2...")
    model_rubert = SentenceTransformer('cointegrated/rubert-tiny2')
    
    processed_docs = []
    
    for doc_idx, item in enumerate(dataset):
        sentences = item["sentences"]
        labels = item["labels"]
        chunks_gt = labels_to_chunks(labels)
        
        # Вычисляем эмбеддинги двумя моделями
        emb_minilm = compute_embeddings(model_minilm, sentences)
        emb_rubert = compute_embeddings(model_rubert, sentences)
        
        processed_docs.append({
            "doc_id": doc_idx,
            "sentences": sentences,
            "labels": labels,
            "chunks_gt": chunks_gt,
            "embeddings_minilm": emb_minilm.tolist(),
            "embeddings_rubert": emb_rubert.tolist()
        })
        
        if (doc_idx + 1) % 10 == 0:
            print(f"Обработано {doc_idx + 1} документов...")
    
    # Сохраняем результат
    output_path = Path("data/rutextseg_processed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False)
    
    print(f"Готово! Обработано {len(processed_docs)} документов.")
    print(f"Результат сохранён в {output_path}")

if __name__ == "__main__":
    main()