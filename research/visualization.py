# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# INPUT_CSV = 'research/data/full_benchmark_detailed.csv'  
# OUTPUT_DIR = 'research/data'                               

# # Какие метрики анализируем
# METRICS = ['pk', 'wd', 'intra', 'f1']             # базовые имена столбцов

# # Словарь методов и соответствующих префиксов в столбцах CSV
# # Например, для метода 'Paragraph' столбцы: par_pk, par_wd, ...
# METHODS = {
#     'Paragraph': 'par',
#     'FixedSize': 'fix',
#     'Semantic (keyword)': 'sem_key',
#     'TextTiling': 'tt',
#     'Greedy (MiniLM)': 'gr',
#     'Hierarchical MiniLM (oracle)': 'hier_oracle_minilm',
#     'Hierarchical rubert (oracle)': 'hier_oracle_rubert',
#     'SemanticEmbedding (ours)': 'sememb',
#     'Supervised LogReg': 'sup'
# }


# def load_data():
#     df = pd.read_csv(INPUT_CSV)
#     print(f"Загружено {len(df)} документов")
#     return df

# def get_metric_columns(df, metric):
#     """Возвращает словарь {method_label: column_name} для заданной метрики (pk, wd, intra, f1)."""
#     return {label: f"{prefix}_{metric}" for label, prefix in METHODS.items() if f"{prefix}_{metric}" in df.columns}

# def plot_boxplot(df, metric='pk', output_file=None):
#     """Строит boxplot для всех методов по одной метрике."""
#     cols = get_metric_columns(df, metric)
#     plot_df = df[list(cols.values())].copy()
#     plot_df.columns = list(cols.keys())
    
#     # Melt для seaborn
#     plot_melt = plot_df.melt(var_name='Метод', value_name=metric.upper())
    
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=plot_melt, x='Метод', y=metric.upper())
#     plt.xticks(rotation=45, ha='right')
#     plt.title(f'Сравнение методов по метрике {metric.upper()}')
#     plt.tight_layout()
    
#     if output_file:
#         plt.savefig(output_file, dpi=150, bbox_inches='tight')
#         print(f"Boxplot сохранён: {output_file}")
#     plt.show()

# def create_summary_table(df, output_csv=None):
#     """Создаёт сводную таблицу (среднее ± std) и сохраняет в CSV."""
#     rows = []
#     for label, prefix in METHODS.items():
#         row = {'Метод': label}
#         for metric in METRICS:
#             col = f"{prefix}_{metric}"
#             if col in df.columns:
#                 mean = df[col].mean()
#                 std = df[col].std()
#                 row[metric.upper()] = f"{mean:.3f} ± {std:.3f}"
#             else:
#                 row[metric.upper()] = "—"
#         rows.append(row)
    
#     summary = pd.DataFrame(rows)
#     print("\nСводная таблица:\n")
#     print(summary.to_string(index=False))
    
#     if output_csv:
#         summary.to_csv(output_csv, index=False)
#         print(f"Таблица сохранена: {output_csv}")
#     return summary

# def bootstrap_ci(data, n_boot=1000, alpha=0.95):
#     """Вычисляет доверительный интервал для среднего методом бутстрепа."""
#     np.random.seed(42)
#     boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
#     lower = np.percentile(boot_means, (1-alpha)/2*100)
#     upper = np.percentile(boot_means, (1+alpha)/2*100)
#     return np.mean(data), lower, upper

# def print_bootstrap_ci(df, metric='pk', methods_of_interest=None):
#     """Выводит доверительные интервалы для выбранных методов."""
#     if methods_of_interest is None:
#         methods_of_interest = ['SemanticEmbedding (ours)', 'Hierarchical MiniLM (oracle)', 'Supervised LogReg']
    
#     cols = get_metric_columns(df, metric)
#     print(f"\n95% доверительные интервалы для метрики {metric.upper()}:\n")
#     for label in methods_of_interest:
#         col = cols.get(label)
#         if col and col in df.columns:
#             mean, low, high = bootstrap_ci(df[col].dropna().values)
#             print(f"{label:35s} mean={mean:.4f}  CI=[{low:.4f}, {high:.4f}]")

# def main():
#     Path(OUTPUT_DIR).mkdir(exist_ok=True)
#     df = load_data()
    
#     # 1. Boxplot для Pk (основной)
#     plot_boxplot(df, metric='pk', output_file=f'{OUTPUT_DIR}/boxplot_pk.png')
    
#     # 2. Boxplot для Boundary F1 или других метрик (при желании)
#     plot_boxplot(df, metric='f1', output_file=f'{OUTPUT_DIR}/boxplot_f1.png')
    
#     # 3. Сводная таблица (сохраняем в CSV и печатаем)
#     create_summary_table(df, output_csv=f'{OUTPUT_DIR}/summary_table.csv')
    
#     # 4. Доверительные интервалы для Pk (три главных метода)
#     print_bootstrap_ci(df, metric='pk')
    
#     print("\nГотово! Графики и таблицы в папке", OUTPUT_DIR)

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_CSV = 'research/data/full_benchmark_detailed.csv'
OUTPUT_DIR = 'research/data'

# Метрики для анализа
METRICS = ['pk', 'wd', 'intra', 'f1']

METHODS = {
    'По абзацам': 'par',
    'Фиксированный размер': 'fix',
    'Ключевые слова': 'sem_key',
    'Жадное наращивание (MiniLM)': 'gr',
    'Иерарх. класт. (MiniLM, истинное число блоков)': 'hier_oracle_minilm',
    'Иерарх. класт. (RuBERT, истинное число блоков)': 'hier_oracle_rubert',
    'Предлагаемый метод': 'sememb',
    'Логистическая регрессия': 'sup',
}

def load_data():
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено {len(df)} документов")
    return df

def get_metric_columns(df, metric):
    return {label: f"{prefix}_{metric}" for label, prefix in METHODS.items()
            if f"{prefix}_{metric}" in df.columns}

def bootstrap_ci(data, n_boot=1000, alpha=0.95):
    """95% доверительный интервал для среднего (бутстреп)."""
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper = np.percentile(boot_means, (1 + alpha) / 2 * 100)
    return np.mean(data), lower, upper

def plot_boxplot(df, metric='pk', output_file=None):
    cols = get_metric_columns(df, metric)
    # Оставляем только методы с ненулевой средней 
    active_cols = {label: col for label, col in cols.items() if df[col].mean() != 0}
    plot_df = df[list(active_cols.values())].copy()
    plot_df.columns = list(active_cols.keys())

    plot_melt = plot_df.melt(var_name='Метод', value_name=metric.upper())

    plt.figure(figsize=(12, 6))
    # Выделяем предлагаемый метод красным
    palette = {m: '#d62728' if 'Предлагаемый' in m else '#1f77b4' for m in active_cols}
    sns.boxplot(data=plot_melt, x='Метод', y=metric.upper(), hue='Метод',
            palette=palette, dodge=False, legend=False)
    plt.xticks(rotation=30, ha='right')
    plt.title(f'Сравнение методов по метрике {metric.upper()}')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Boxplot сохранён: {output_file}")
    plt.show()

def plot_bar_with_ci(df, metric='pk', output_file=None):
    """Столбчатая диаграмма средних значений с 95% доверительными интервалами."""
    cols = get_metric_columns(df, metric)
    active_cols = {label: col for label, col in cols.items() if df[col].mean() != 0}
    methods = list(active_cols.keys())
    means = []
    low_err = []
    high_err = []
    colors = []

    for m in methods:
        data = df[active_cols[m]].dropna().values
        m_mean, low, high = bootstrap_ci(data)
        means.append(m_mean)
        low_err.append(m_mean - low)
        high_err.append(high - m_mean)
        colors.append('#d62728' if 'Предлагаемый' in m else '#1f77b4')

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, means, yerr=[low_err, high_err], capsize=5,
                   color=colors, edgecolor='black', alpha=0.85)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'Средние значения метрики {metric.upper()} с 95% доверительными интервалами', fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, mean_val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Столбчатая диаграмма сохранена: {output_file}")
    plt.show()

def create_summary_table(df, output_csv=None):
    """Сводная таблица (среднее ± std) на русском."""
    rows = []
    for label, prefix in METHODS.items():
        row = {'Метод': label}
        for metric in METRICS:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                row[metric.upper()] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[metric.upper()] = "—"
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\nСводная таблица:\n")
    print(summary.to_string(index=False))

    if output_csv:
        summary.to_csv(output_csv, index=False)
        print(f"Таблица сохранена: {output_csv}")
    return summary

def print_bootstrap_ci(df, metric='pk', methods_of_interest=None):
    """Выводит 95% доверительные интервалы для нескольких методов."""
    if methods_of_interest is None:
        methods_of_interest = ['Предлагаемый метод', 'Иерарх. класт. (MiniLM, истинное число блоков)',
                               'Логистическая регрессия']

    cols = get_metric_columns(df, metric)
    print(f"\n95% доверительные интервалы для метрики {metric.upper()}:\n")
    for label in methods_of_interest:
        col = cols.get(label)
        if col and col in df.columns:
            mean, low, high = bootstrap_ci(df[col].dropna().values)
            print(f"{label:40s} mean={mean:.4f}  CI=[{low:.4f}, {high:.4f}]")

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    df = load_data()

    # 1. Boxplot для Pk
    plot_boxplot(df, metric='pk', output_file=f'{OUTPUT_DIR}/boxplot_pk.png')

    # 2. Boxplot для Boundary F1
    plot_boxplot(df, metric='f1', output_file=f'{OUTPUT_DIR}/boxplot_f1.png')

    # 3. Столбчатая диаграмма с доверительными интервалами (Pk)
    plot_bar_with_ci(df, metric='pk', output_file=f'{OUTPUT_DIR}/barplot_pk_ci.png')

    # 4. Столбчатая диаграмма для F1 (опционально)
    plot_bar_with_ci(df, metric='f1', output_file=f'{OUTPUT_DIR}/barplot_f1_ci.png')

    # 5. Сводная таблица
    create_summary_table(df, output_csv=f'{OUTPUT_DIR}/summary_table.csv')

    # 6. Доверительные интервалы для главных метрик
    print_bootstrap_ci(df, metric='pk')
    print_bootstrap_ci(df, metric='f1')

    print("\nГрафики и таблицы в папке", OUTPUT_DIR)

if __name__ == "__main__":
    main()