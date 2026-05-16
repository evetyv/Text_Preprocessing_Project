# File: prepare_boundary_data.py

import json
import numpy as np
from pathlib import Path

def build_boundary_dataset(processed_docs_path):
    """
    Формирует обучающий набор признаков и меток для логистической регрессии
    из предоставленного файла обработанных документов.
    Возвращает X, y, doc_ids, pair_indices для всех пар.
    """
    with open(processed_docs_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    X = []
    y = []
    doc_ids = []       # идентификатор документа для каждой пары
    pair_indices = []  # порядковый номер пары внутри документа

    for doc in docs:
        sentences = doc['sentences']
        labels = doc['labels']
        emb = np.array(doc['embeddings_minilm'])
        n = len(sentences)
        for i in range(n - 1):
            emb_i = emb[i]
            emb_j = emb[i+1]
            # 1) конкатенация
            concat = np.concatenate([emb_i, emb_j])
            # 2) косинусное сходство (одно число)
            cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-9)
            # 3) разность
            diff = emb_i - emb_j
            # объединяем всё вместе
            feat = np.concatenate([concat, [cos_sim], diff])
            X.append(feat)
            y.append(labels[i+1])
            doc_ids.append(doc['doc_id'])
            pair_indices.append(i)  # i-я пара (предложения i и i+1)

    X = np.array(X)
    y = np.array(y)
    doc_ids = np.array(doc_ids)
    pair_indices = np.array(pair_indices)

    print(f"Всего пар для обучения: {len(X)}")
    print(f"Количество положительных примеров (границ): {y.sum()}")

    return X, y, doc_ids, pair_indices

def main():
    # Путь к файлу с документами для обучения регрессии
    train_path = Path("research/data/rutextseg_train_logreg.json")
    if not train_path.exists():
        print(f"Файл {train_path} не найден. Сначала запустите prepare_dataset.py")
        return

    X_train, y_train, doc_ids_train, pair_idx_train = build_boundary_dataset(train_path)

    # Сохраняем только обучающие данные
    output_path = "research/data/boundary_train_data.npz"
    np.savez_compressed(output_path,
                        X_train=X_train,
                        y_train=y_train,
                        doc_ids_train=doc_ids_train,
                        pair_idx_train=pair_idx_train)
    print(f"Обучающие данные сохранены в {output_path}")

if __name__ == "__main__":
    main()