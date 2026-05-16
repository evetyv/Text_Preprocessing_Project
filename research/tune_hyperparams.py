import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def chunks_to_boundaries(chunks, n_sent):
    b = np.zeros(n_sent - 1, dtype=bool)
    for ch in chunks:
        if ch:
            end = ch[-1]
            if end < n_sent - 1:
                b[end] = True
    return b

def boundary_f1(true_chunks, pred_chunks, n_sent, tolerance=1):
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

# Кластеризация с заданным параметром alpha
def hierarchical_chunking_alpha(embeddings, n_total, alpha):
    n = len(embeddings)
    if n < 2:
        return [list(range(n))]
    n_clusters = max(2, n // alpha)
    n_clusters = min(n_clusters, n)
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
    # Группировка по непрерывным меткам
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

def main():
    data_path = Path("research/data/rutextseg_train_factor.json")
    if not data_path.exists():
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print(f"Загружено {len(docs)} документов для подбора гиперпараметров")

    models = {
        "MiniLM": "embeddings_minilm",
        "RuBERT": "embeddings_rubert"
    }
    alphas = list(range(2, 9))      # 2..8
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metric_names = ['f1', 'pk', 'wd', 'intra', 'rel_chunk_size']
    results = {model: {a: {m: [] for m in metric_names} for a in alphas} for model in models}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(docs)):
        val_docs = [docs[i] for i in val_idx]
        for model_name, emb_key in models.items():
            for alpha in alphas:
                f1_scores, pk_scores, wd_scores, intra_scores, rel_sizes = [], [], [], [], []
                for doc in val_docs:
                    emb = np.array(doc[emb_key])
                    gt_chunks = doc['chunks_gt']
                    n_sent = len(doc['sentences'])
                    pred_chunks = hierarchical_chunking_alpha(emb, n_sent, alpha)

                    f1 = boundary_f1(gt_chunks, pred_chunks, n_sent)
                    pk = pk_metric(gt_chunks, pred_chunks, n_sent)
                    wd = windowdiff(gt_chunks, pred_chunks, n_sent)
                    intra = intra_chunk_similarity(pred_chunks, emb)

                    chunk_sizes = [len(ch) for ch in pred_chunks]
                    if n_sent > 0:
                        avg_rel_size = np.mean(chunk_sizes) / n_sent
                    else:
                        avg_rel_size = 0

                    f1_scores.append(f1)
                    pk_scores.append(pk)
                    wd_scores.append(wd)
                    intra_scores.append(intra)
                    rel_sizes.append(avg_rel_size)

                results[model_name][alpha]['f1'].append(np.mean(f1_scores))
                results[model_name][alpha]['pk'].append(np.mean(pk_scores))
                results[model_name][alpha]['wd'].append(np.mean(wd_scores))
                results[model_name][alpha]['intra'].append(np.mean(intra_scores))
                results[model_name][alpha]['rel_chunk_size'].append(np.mean(rel_sizes))

        print(f"Фолд {fold_idx+1} завершён")

    # Усреднение и поиск лучшей комбинации по F1
    best_f1 = -1
    best_model = None
    best_alpha = None
    summary = {}

    for model_name in models:
        summary[model_name] = {}
        for alpha in alphas:
            means = {}
            stds = {}
            for m in metric_names:
                arr = results[model_name][alpha][m]
                means[m] = np.mean(arr)
                stds[m] = np.std(arr)
            summary[model_name][alpha] = {'means': means, 'stds': stds}
            if means['f1'] > best_f1:
                best_f1 = means['f1']
                best_model = model_name
                best_alpha = alpha

    print(f"\nЛучшая комбинация по F1: модель {best_model}, α = {best_alpha}")
    print(f"Средняя Boundary F1 = {best_f1:.4f}")

    # Сохранение таблицы
    rows = []
    for model_name in models:
        for alpha in alphas:
            m = summary[model_name][alpha]['means']
            s = summary[model_name][alpha]['stds']
            rows.append({
                'model': model_name,
                'alpha': alpha,
                'f1_mean': m['f1'], 'f1_std': s['f1'],
                'pk_mean': m['pk'], 'pk_std': s['pk'],
                'wd_mean': m['wd'], 'wd_std': s['wd'],
                'intra_mean': m['intra'], 'intra_std': s['intra'],
                'rel_chunk_size_mean': m['rel_chunk_size'], 'rel_chunk_size_std': s['rel_chunk_size']
            })
    df_summary = pd.DataFrame(rows)
    csv_path = Path("research/data/hyperparams_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Сводная таблица сохранена: {csv_path}")

    output_dir = Path("research/data")
    output_dir.mkdir(exist_ok=True)

    metrics_to_plot = ['f1', 'pk', 'wd', 'intra', 'rel_chunk_size']
    ylabels = ['Boundary F1', 'Pk', 'WindowDiff', 'Intra Similarity', 'Относительный размер блока']

    plt.style.use('seaborn-v0_8-whitegrid')
    for metric, ylab in zip(metrics_to_plot, ylabels):
        plt.figure(figsize=(10, 6))
        for model_name, color in zip(models.keys(), ['#1f77b4', '#d62728']):
            means = [summary[model_name][a]['means'][metric] for a in alphas]
            stds  = [summary[model_name][a]['stds'][metric] for a in alphas]
            plt.errorbar(alphas, means, yerr=stds, marker='o', capsize=5,
                         label=model_name, color=color)
        plt.xlabel('Параметр α (n // α)', fontsize=12)
        plt.ylabel(ylab, fontsize=12)
        plt.title(f'Зависимость {ylab} от параметра α', fontsize=14)
        plt.legend()
        plt.xticks(alphas)
        plt.tight_layout()
        save_path = output_dir / f"tuning_{metric}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"График сохранён: {save_path}")

if __name__ == "__main__":
    main()