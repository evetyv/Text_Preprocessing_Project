import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_CSV = 'research/data/full_benchmark_detailed.csv'
OUTPUT_DIR = 'research/data'

# Метрики (ключи в CSV)
METRICS = ['pk', 'wd', 'intra', 'f1']

# Русские названия метрик для подписей
METRIC_NAMES_RU = {
    'pk': 'Pk',
    'wd': 'WindowDiff',
    'intra': 'Внутренняя однородность',
    'f1': 'Boundary F1'
}

# Полный список методов (ключ: русское название, значение: префикс в колонках CSV)
ALL_METHODS = {
    'По абзацам': 'par',
    'Фиксированный размер': 'fix',
    'Ключевые слова': 'sem_key',
    'Жадное наращивание\n(MiniLM)': 'gr',
    'Иерарх. класт.\n(MiniLM, oracle)': 'hier_oracle_minilm',
    'Иерарх. класт.\n(RuBERT, oracle)': 'hier_oracle_rubert',
    'Предлагаемый метод': 'sememb',
    'Логистическая регрессия': 'sup',
}

# Группы методов для отдельных рисунков
GROUPS = {
    'Базовые методы': ['По абзацам', 'Фиксированный размер', 'Ключевые слова', 'Предлагаемый метод'],
    'Иерархические и жадный': [
        'Жадное наращивание\n(MiniLM)',
        'Иерарх. класт.\n(MiniLM, oracle)',
        'Иерарх. класт.\n(RuBERT, oracle)',
        'Предлагаемый метод'
    ],
    'Обученный (supervised)': ['Логистическая регрессия', 'Предлагаемый метод'],
}

def load_data():
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено {len(df)} документов")
    return df

def bootstrap_ci(data, n_boot=1000, alpha=0.95):
    """95% доверительный интервал для среднего (бутстреп)."""
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper = np.percentile(boot_means, (1 + alpha) / 2 * 100)
    return np.mean(data), lower, upper

def get_method_column(method_name):
    """Возвращает имя колонки в DataFrame для данного метода и метрики (будет дополнено позже)."""
    return ALL_METHODS[method_name]

def plot_group_comparison(df, group_name, method_names, output_file=None):
    """
    Рисует один рисунок с 2x2 подграфиками (Pk, WindowDiff, Intra, F1)
    для заданной группы методов.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        metric_ru = METRIC_NAMES_RU[metric]

        # Собираем данные только для методов этой группы
        means = []
        low_err = []
        high_err = []
        labels = []
        colors = []

        for method_name in method_names:
            prefix = ALL_METHODS[method_name]
            col_name = f"{prefix}_{metric}"
            if col_name not in df.columns:
                continue
            data = df[col_name].dropna().values
            if len(data) == 0:
                continue
            m_mean, low, high = bootstrap_ci(data)
            means.append(m_mean)
            low_err.append(m_mean - low)
            high_err.append(high - m_mean)
            # Для переносов строк в подписях оставим короткое имя
            short_name = method_name.replace('\n', ' ')
            labels.append(short_name)
            colors.append('#d62728' if 'Предлагаемый' in method_name else '#1f77b4')

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=[low_err, high_err], capsize=5,
                      color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(metric_ru, fontsize=11)
        ax.set_title(f'{metric_ru}', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Подписываем значения над столбцами
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'Сравнение методов: {group_name}', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Групповой график сохранён: {output_file}")
    plt.show()

def create_summary_table(df, output_csv=None):
    """Сводная таблица (среднее ± std) с русскими названиями метрик."""
    rows = []
    for label, prefix in ALL_METHODS.items():
        # Убираем переносы строк для печати в консоль
        clean_label = label.replace('\n', ' ')
        row = {'Метод': clean_label}
        for metric in METRICS:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                row[METRIC_NAMES_RU[metric]] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[METRIC_NAMES_RU[metric]] = "—"
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\nСводная таблица:\n")
    print(summary.to_string(index=False))

    if output_csv:
        summary.to_csv(output_csv, index=False)
        print(f"Таблица сохранена: {output_csv}")
    return summary

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    df = load_data()

    # 1. Групповые сравнения
    for group_name, method_list in GROUPS.items():
        safe_name = group_name.replace(' ', '_').lower()
        plot_group_comparison(
            df,
            group_name,
            method_list,
            output_file=f'{OUTPUT_DIR}/comparison_{safe_name}.png'
        )

    # 2. Сводная таблица
    create_summary_table(df, output_csv=f'{OUTPUT_DIR}/summary_table.csv')

    print("\nВсе графики и таблицы сохранены в папке", OUTPUT_DIR)

if __name__ == "__main__":
    main()