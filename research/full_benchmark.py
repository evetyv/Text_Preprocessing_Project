import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
import nltk
from nltk.tokenize import TextTilingTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing_module'))

from src.text_splitter import (
    ParagraphSplitter, 
    SemanticSplitter, 
    FixedSizeSplitter, 
    SemanticEmbeddingSplitter,
    SplitterConfig
)

def build_sentence_positions(sentences, full_text):
    """Возвращает список (start, end) для каждого предложения в full_text."""
    positions = []
    search_start = 0
    for sent in sentences:
        start = full_text.find(sent, search_start)
        if start == -1:
            # Если не нашли (например, из-за различий в пунктуации), используем эвристику
            start = search_start
        end = start + len(sent)
        positions.append((start, end))
        search_start = end + 1  # +1 учитывает пробел между предложениями
    return positions

def chunks_to_indices(sent_positions, chunks_info):
    """Превращает список TextChunkInfo в список списков индексов предложений."""
    result = []
    for chunk in chunks_info:
        indices = []
        for i, (s_start, s_end) in enumerate(sent_positions):
            # Пересечение интервалов (учитываем, что граница чанка может совпадать с границей предложения)
            if max(chunk.start_pos, s_start) < min(chunk.end_pos, s_end):
                indices.append(i)
        result.append(indices)
    return result

def labels_to_chunks(labels):
    chunks = []
    current = [0]
    for i in range(1, len(labels)):
        if labels[i] == 1:
            chunks.append(current)
            current = [i]
        else:
            current.append(i)
    chunks.append(current)
    return chunks

def chunks_to_boundaries(chunks, n_sent):
    b = np.zeros(n_sent - 1, dtype=bool)
    for ch in chunks:
        if ch:
            end = ch[-1]
            if end < n_sent - 1:
                b[end] = True
    return b

def pk_metric(true_chunks, pred_chunks, n_sent, window_size=None):
    if window_size is None:
        window_size = max(1, n_sent // 20)
    true_b = chunks_to_boundaries(true_chunks, n_sent)
    pred_b = chunks_to_boundaries(pred_chunks, n_sent)
    total = 0
    disagreements = 0
    for i in range(n_sent - window_size):
        true_has = np.any(true_b[i:i+window_size])
        pred_has = np.any(pred_b[i:i+window_size])
        if true_has != pred_has:
            disagreements += 1
        total += 1
    return disagreements / total if total > 0 else 0

def windowdiff(true_chunks, pred_chunks, n_sent, window_size=None):
    if window_size is None:
        window_size = max(1, n_sent // 20)
    true_b = chunks_to_boundaries(true_chunks, n_sent)
    pred_b = chunks_to_boundaries(pred_chunks, n_sent)
    total = 0
    diff_sum = 0
    for i in range(n_sent - window_size):
        diff_sum += abs(np.sum(true_b[i:i+window_size]) - np.sum(pred_b[i:i+window_size]))
        total += 1
    return diff_sum / total if total > 0 else 0

def intra_chunk_similarity(chunks, embeddings):
    total_sim = 0.0
    count = 0
    for ch in chunks:
        if len(ch) < 2:
            continue
        chunk_emb = embeddings[ch]
        sim_mat = cosine_similarity(chunk_emb)
        triu = np.triu_indices_from(sim_mat, k=1)
        total_sim += np.mean(sim_mat[triu])
        count += 1
    return total_sim / count if count > 0 else 0

def boundary_f1(true_chunks, pred_chunks, n_sent, tolerance=1):
    """
    F1-мера для границ с толерантностью (по умолчанию +-1 предложение).
    Граница считается найденной, если есть предсказанная граница в радиусе tolerance.
    """
    true_b = chunks_to_boundaries(true_chunks, n_sent)
    pred_b = chunks_to_boundaries(pred_chunks, n_sent)
    true_positions = np.where(true_b)[0]
    pred_positions = np.where(pred_b)[0]

    if len(true_positions) == 0 and len(pred_positions) == 0:
        return 1.0
    if len(true_positions) == 0 or len(pred_positions) == 0:
        return 0.0

    matched_true = set()
    matched_pred = set()
    for ti, tp in enumerate(true_positions):
        for pi, pp in enumerate(pred_positions):
            if pi in matched_pred:
                continue
            if abs(tp - pp) <= tolerance:
                matched_true.add(ti)
                matched_pred.add(pi)
                break
    tp = len(matched_true)
    fp = len(pred_positions) - len(matched_pred)
    fn = len(true_positions) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ------------------------------------------------------------
# Методы чанкования
# ------------------------------------------------------------

config = SplitterConfig(min_chunk_size=100, max_chunk_size=2000)

splitters = {
    "Paragraph": ParagraphSplitter(config),
    "FixedSize": FixedSizeSplitter(config),
    "Semantic": SemanticSplitter(config),
    "SemanticEmbedding": SemanticEmbeddingSplitter(config)  # использует MiniLM, auto k
}

def greedy_chunking(embeddings, threshold=0.5):
    n = len(embeddings)
    if n < 2:
        return [list(range(n))]
    sims = []
    for i in range(n - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        sims.append(sim)
    chunks = []
    cur = [0]
    for i in range(1, n):
        if sims[i-1] < threshold:
            chunks.append(cur)
            cur = [i]
        else:
            cur.append(i)
    chunks.append(cur)
    return chunks

def hierarchical_chunking(embeddings, n_clusters):
    n = len(embeddings)
    if n < 2:
        return [list(range(n))]
    n_clusters = min(n_clusters, n)
    if n_clusters < 2:
        return [list(range(n))]
    connectivity = np.zeros((n, n))
    for i in range(n - 1):
        connectivity[i, i+1] = 1
        connectivity[i+1, i] = 1
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
        connectivity=connectivity
    )
    labels = clustering.fit_predict(embeddings)
    chunks = []
    cur_label = labels[0]
    cur = [0]
    for i in range(1, n):
        if labels[i] != cur_label:
            chunks.append(cur)
            cur = [i]
            cur_label = labels[i]
        else:
            cur.append(i)
    chunks.append(cur)
    return chunks

# ------------------------------------------------------------
# Основной блок
# ------------------------------------------------------------
def main():
    # Загрузка данных
    with open('research/data/rutextseg_processed.json', 'r', encoding='utf-8') as f:
        all_docs = json.load(f)

    data = np.load('research/data/boundary_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    doc_ids_test = data['doc_ids_test']
    pair_idx_test = data['pair_idx_test']

    # Обучение логистической регрессии с подбором параметров
    print("Обучение LogisticRegression с подбором параметров...")
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    param_grid = {'C': [0.01, 0.1, 1.0]}
    base_clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    grid = GridSearchCV(base_clf, param_grid, cv=3, scoring='f1')
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    print(f"Лучший C: {grid.best_params_['C']}")
    
    y_probs = best_clf.predict_proba(X_test)[:, 1]

    # Группировка предсказаний по документам
    test_docs_map = {}
    for i in range(len(doc_ids_test)):
        did = int(doc_ids_test[i])
        pid = int(pair_idx_test[i])
        prob = y_probs[i]
        if did not in test_docs_map:
            test_docs_map[did] = {'pairs': [], 'probs': []}
        test_docs_map[did]['pairs'].append(pid)
        test_docs_map[did]['probs'].append(prob)

    # Собираем тестовые документы из all_docs
    test_docs = []
    for doc in all_docs:
        if doc['doc_id'] in test_docs_map:
            test_docs.append(doc)
    
    # import random
    # random.seed(58)
    # test_docs = random.sample(test_docs, min(10, len(test_docs)))  # только 10 документов
    
    config = SplitterConfig(min_chunk_size=100, max_chunk_size=2000)
    splitters = {
        "Paragraph": ParagraphSplitter(config),
        "FixedSize": FixedSizeSplitter(config),   
        "Semantic (keyword)": SemanticSplitter(config),
        "SemanticEmbedding (ours)": SemanticEmbeddingSplitter(config)
    }

    # Список методов и сбор результатов
    methods = [
        'Paragraph', 'FixedSize',
          'Semantic (keyword)',
        'Greedy (MiniLM)',
        'Hierarchical MiniLM', 'Hierarchical rubert',
        'SemanticEmbedding',
        'Supervised LogReg'
    ]
    results = []

    import time
    print("Вычисление метрик для всех методов...")
    for doc in test_docs:
        doc_id = doc['doc_id']
        print(f"\nОбрабатываю документ {doc_id}...")
        doc_start = time.time()
        sentences = doc['sentences']
        n_sent = len(sentences)
        gt_chunks = doc['chunks_gt']
        emb_minilm = np.array(doc['embeddings_minilm'])
        emb_rubert = np.array(doc['embeddings_rubert'])
        k = len(gt_chunks)

        full_text = ' '.join(sentences)
        sent_positions = build_sentence_positions(sentences, full_text)

        our_results = {}

        # Измеряем каждый сплиттер отдельно
        for name, splitter in splitters.items():
            t0 = time.time()
            if isinstance(splitter, SemanticEmbeddingSplitter):
                chunks_info = splitter.split(full_text, embeddings=emb_minilm)
            else:
                chunks_info = splitter.split(full_text)
            pred_chunks = chunks_to_indices(sent_positions, chunks_info)
            pk = pk_metric(gt_chunks, pred_chunks, n_sent)
            wd = windowdiff(gt_chunks, pred_chunks, n_sent)
            intra = intra_chunk_similarity(pred_chunks, emb_minilm)
            f1 = boundary_f1(gt_chunks, pred_chunks, n_sent)
            our_results[name] = [pk, wd, intra, f1]
            print(f"  {name:30s} {time.time() - t0:.2f} сек")

        # Greedy MiniLM
        gr_chunks = greedy_chunking(emb_minilm, threshold=0.5)
        gr_pk = pk_metric(gt_chunks, gr_chunks, n_sent)
        gr_wd = windowdiff(gt_chunks, gr_chunks, n_sent)
        gr_intra = intra_chunk_similarity(gr_chunks, emb_minilm)
        gr_f1 = boundary_f1(gt_chunks, gr_chunks, n_sent)

        # Hierarchical MiniLM (oracle)
        hier1_chunks = hierarchical_chunking(emb_minilm, n_clusters=k)
        hier1_pk = pk_metric(gt_chunks, hier1_chunks, n_sent)
        hier1_wd = windowdiff(gt_chunks, hier1_chunks, n_sent)
        hier1_intra = intra_chunk_similarity(hier1_chunks, emb_minilm)
        hier1_f1 = boundary_f1(gt_chunks, hier1_chunks, n_sent)

        # Hierarchical rubert (oracle)
        hier2_chunks = hierarchical_chunking(emb_rubert, n_clusters=k)
        hier2_pk = pk_metric(gt_chunks, hier2_chunks, n_sent)
        hier2_wd = windowdiff(gt_chunks, hier2_chunks, n_sent)
        hier2_intra = intra_chunk_similarity(hier2_chunks, emb_rubert)
        hier2_f1 = boundary_f1(gt_chunks, hier2_chunks, n_sent)

        # Supervised LogReg
        did = doc['doc_id']
        pairs = test_docs_map[did]['pairs']
        probs = test_docs_map[did]['probs']
        sorted_idx = np.argsort(pairs)
        probs_sorted = np.array(probs)[sorted_idx]
        labels_pred = [0]
        for p in probs_sorted:
            labels_pred.append(1 if p >= 0.5 else 0)
        sup_chunks = labels_to_chunks(labels_pred)
        sup_pk = pk_metric(gt_chunks, sup_chunks, n_sent)
        sup_wd = windowdiff(gt_chunks, sup_chunks, n_sent)
        sup_intra = intra_chunk_similarity(sup_chunks, emb_minilm)
        sup_f1 = boundary_f1(gt_chunks, sup_chunks, n_sent)

        fix_pk, fix_wd, fix_intra, fix_f1 = our_results.get('FixedSize', [0, 0, 0, 0])
        row = {
            'doc_id': doc_id, 'n_sentences': n_sent,
            'par_pk': our_results['Paragraph'][0],
            'par_wd': our_results['Paragraph'][1],
            'par_intra': our_results['Paragraph'][2],
            'par_f1': our_results['Paragraph'][3],
            'fix_pk': fix_pk, 'fix_wd': fix_wd, 'fix_intra': fix_intra, 'fix_f1': fix_f1,
            'sem_key_pk': our_results['Semantic (keyword)'][0],
            'sem_key_wd': our_results['Semantic (keyword)'][1],
            'sem_key_intra': our_results['Semantic (keyword)'][2],
            'sem_key_f1': our_results['Semantic (keyword)'][3],
            'sememb_pk': our_results['SemanticEmbedding (ours)'][0],
            'sememb_wd': our_results['SemanticEmbedding (ours)'][1],
            'sememb_intra': our_results['SemanticEmbedding (ours)'][2],
            'sememb_f1': our_results['SemanticEmbedding (ours)'][3],
            'gr_pk': gr_pk, 'gr_wd': gr_wd, 'gr_intra': gr_intra, 'gr_f1': gr_f1,
            'hier_oracle_minilm_pk': hier1_pk, 'hier_oracle_minilm_wd': hier1_wd,
            'hier_oracle_minilm_intra': hier1_intra, 'hier_oracle_minilm_f1': hier1_f1,
            'hier_oracle_rubert_pk': hier2_pk, 'hier_oracle_rubert_wd': hier2_wd,
            'hier_oracle_rubert_intra': hier2_intra, 'hier_oracle_rubert_f1': hier2_f1,
            'sup_pk': sup_pk, 'sup_wd': sup_wd, 'sup_intra': sup_intra, 'sup_f1': sup_f1
        }
        results.append(row)
        print(f"  Документ {doc_id} завершён за {time.time() - doc_start:.2f} сек")
        
    df = pd.DataFrame(results)
    df.to_csv('research/data/full_benchmark_detailed.csv', index=False)
    print("Детальные метрики сохранены в data/full_benchmark_detailed.csv")
    print("Для визуализации запустите visualization.py")


if __name__ == "__main__":
    main()