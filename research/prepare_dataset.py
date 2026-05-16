import json
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

def labels_to_chunks(labels):
    """Преобразует список меток (1 - начало нового чанка, 0 - продолжение)
    в список чанков (списки индексов предложений)."""
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

def compute_embeddings(model, sentences, batch_size=64):
    """Вычисляет эмбеддинги для списка предложений с заданным размером батча."""
    return model.encode(sentences, show_progress_bar=False, batch_size=batch_size)

def process_split(split_name, dataset, model_minilm, model_rubert):
    """Обрабатывает один сплит датасета и возвращает список документов."""
    processed = []
    total = len(dataset)
    for doc_idx, item in enumerate(dataset):
        sentences = item["sentences"]
        labels = item["labels"]
        chunks_gt = labels_to_chunks(labels)

        emb_minilm = compute_embeddings(model_minilm, sentences)
        emb_rubert = compute_embeddings(model_rubert, sentences)

        processed.append({
            "doc_id": doc_idx,
            "sentences": sentences,
            "labels": labels,
            "chunks_gt": chunks_gt,
            "embeddings_minilm": emb_minilm.tolist(),
            "embeddings_rubert": emb_rubert.tolist()
        })

        if (doc_idx + 1) % 100 == 0 or (doc_idx + 1) == total:
            print(f"[{split_name}] Обработано {doc_idx + 1}/{total} документов...")
    return processed

def main():
    # Параметры
    MAX_TRAIN_DOCS = 10_000   
    RANDOM_SEED = 42
    BATCH_SIZE = 64           

    data_dir = Path("research/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Фиксируем генератор случайных чисел
    random.seed(RANDOM_SEED)

    # Загружаем модели (один раз)
    print("Загрузка модели MiniLM...")
    model_minilm = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Загрузка модели rubert-tiny2...")
    model_rubert = SentenceTransformer('cointegrated/rubert-tiny2')

    print("\nЗагрузка test-сплита RuTextSegNews...")
    dataset_test = load_dataset("mlenjoyneer/RuTextSegNews", split="test")
    print(f"Test-сплит: {len(dataset_test)} документов")

    print("Обработка test-сплита...")
    test_docs = process_split("test", dataset_test, model_minilm, model_rubert)
    test_path = data_dir / "rutextseg_test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_docs, f, ensure_ascii=False)
    print(f"Test сохранён: {test_path} ({len(test_docs)} док.)")

    print("\nЗагрузка train-сплита RuTextSegNews...")
    dataset_train_full = load_dataset("mlenjoyneer/RuTextSegNews", split="train")
    total_train = len(dataset_train_full)
    print(f"Полный train-сплит: {total_train} документов")

    if total_train > MAX_TRAIN_DOCS:
        indices = random.sample(range(total_train), MAX_TRAIN_DOCS)
        dataset_train = dataset_train_full.select(indices)
        print(f"Отобрано {len(dataset_train)} документов (ограничение {MAX_TRAIN_DOCS})")
    else:
        dataset_train = dataset_train_full

    # Обрабатываем
    print("Обработка train-подвыборки...")
    train_docs = process_split("train", dataset_train, model_minilm, model_rubert)

    # Разделяем train на части для LogReg и подбора фактора (70/30)
    print("Разделение train на logreg (70%) и factor (30%)...")
    train_logreg, train_factor = train_test_split(
        train_docs, test_size=0.3, random_state=RANDOM_SEED, shuffle=True
    )
    # Перенумерация doc_id для порядка
    for i, doc in enumerate(train_logreg):
        doc["doc_id"] = i
    for i, doc in enumerate(train_factor):
        doc["doc_id"] = i

    logreg_path = data_dir / "rutextseg_train_logreg.json"
    with open(logreg_path, "w", encoding="utf-8") as f:
        json.dump(train_logreg, f, ensure_ascii=False)
    print(f"Train_logreg сохранён: {logreg_path} ({len(train_logreg)} док.)")

    factor_path = data_dir / "rutextseg_train_factor.json"
    with open(factor_path, "w", encoding="utf-8") as f:
        json.dump(train_factor, f, ensure_ascii=False)
    print(f"Train_factor сохранён: {factor_path} ({len(train_factor)} док.)")

    print("\nВсе файлы созданы в папке research/data.")

if __name__ == "__main__":
    main()