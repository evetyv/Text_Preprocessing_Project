import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_boundary_dataset_with_ids(processed_docs_path, test_size=0.2, random_seed=42):
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
    
    # Разбиваем на train/test, сохраняя doc_ids и pair_indices
    X_train, X_test, y_train, y_test, doc_ids_train, doc_ids_test, pair_idx_train, pair_idx_test = train_test_split(
        X, y, doc_ids, pair_indices,
        test_size=test_size, random_state=random_seed, stratify=y
    )
    
    print(f"Всего пар: {len(X)}")
    print(f"Обучающих: {len(X_train)}, границ: {y_train.sum()}")
    print(f"Тестовых: {len(X_test)}, границ: {y_test.sum()}")
    
    return (X_train, X_test, y_train, y_test,
            doc_ids_train, doc_ids_test, pair_idx_train, pair_idx_test)

def main():
    processed_path = Path("research/data/rutextseg_processed.json")
    if not processed_path.exists():
        print("Файл не найден.")
        return
    
    (X_train, X_test, y_train, y_test,
     doc_ids_train, doc_ids_test, pair_idx_train, pair_idx_test) = build_boundary_dataset_with_ids(processed_path)
    
    np.savez_compressed("research/data/boundary_data.npz",
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        doc_ids_train=doc_ids_train, doc_ids_test=doc_ids_test,
                        pair_idx_train=pair_idx_train, pair_idx_test=pair_idx_test)
    print("Данные с doc_id сохранены в data/boundary_data.npz")

if __name__ == "__main__":
    main()