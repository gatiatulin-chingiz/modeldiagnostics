class CategoricalFeatureProcessor:
    def __init__(self, threshold_strategy=('percentile', 70), manual_mappings=None,
                 rare_label='OTHER', missing_label='MISSING'):
        """
        Инициализация процессора категориальных признаков.

        Параметры:
        threshold_strategy (str or tuple or dict): 
            - если str: общая стратегия для всех признаков
            - если tuple: (стратегия, процентиль), например ('percentile', 10)
            - если dict: {признак: стратегия или (стратегия, процентиль)}
            Допустимые стратегии: 'median', 'mean', 'percentile', 'iqr', 'elbow'
        manual_mappings (dict): {колонка: [список категорий], ...} — категории, которые оставляем как "частые"
        rare_label (str): значение для редких категорий
        missing_label (str): значение для пропущенных значений
        """
        self.threshold_strategy = threshold_strategy
        self.manual_mappings = manual_mappings or {}
        self.rare_label = rare_label
        self.missing_label = missing_label
        self.mappings_ = {}

    def transform(self, df, cat_cols):
        """
        Применяет преобразования к данным: заполняет пропуски и группирует редкие категории.

        Параметры:
        df (pd.DataFrame): входной датафрейм
        cat_cols (list): список категориальных колонок

        Возвращает:
        pd.DataFrame: преобразованный датафрейм
        """
        self.cat_cols = cat_cols
        self._df = df.copy()

        for col in self.cat_cols:
            if col not in self._df.columns:
                raise ValueError(f"Column {col} not found in input data")

        # Парсим threshold_strategy
        if isinstance(self.threshold_strategy, str):
            self._threshold_per_col = {col: (self.threshold_strategy, None) for col in cat_cols}
        elif isinstance(self.threshold_strategy, tuple) and self.threshold_strategy[0] == 'percentile':
            self._threshold_per_col = {col: self.threshold_strategy for col in cat_cols}
        elif isinstance(self.threshold_strategy, dict):
            self._threshold_per_col = {}
            for col in cat_cols:
                strategy = self.threshold_strategy.get(col, 'median')  # по умолчанию median
                if isinstance(strategy, str):
                    self._threshold_per_col[col] = (strategy, None)
                elif isinstance(strategy, tuple) and strategy[0] == 'percentile':
                    self._threshold_per_col[col] = strategy
                else:
                    raise ValueError(
                        f"Unknown threshold strategy: {col}: {strategy}. Acceptable strategies: 'median', 'mean', 'percentile', 'iqr', 'elbow'")
        else:
            raise TypeError("threshold_strategy должен быть str, tuple или dict")

        mappings = {}

        for col in cat_cols:
            # Если есть ручное маппинг — используем его
            if col in self.manual_mappings:
                frequent_cats = list(map(str, self.manual_mappings[col]))
                mappings[col] = frequent_cats
                continue

            strategy, percentile = self._threshold_per_col[col]

            # Преобразуем 'None' в np.nan
            df_col = self._df[col].replace('None', np.nan)

            # Заполнение пропусков и приведение к строке
            df_col = df_col.fillna(self.missing_label).astype(str)

            # Подсчёт частоты для каждой категории
            counts = df_col.value_counts()

            # Выбор порога на основе выбранной стратегии
            if strategy == 'median':
                thr = counts.median()
            elif strategy == 'mean':
                thr = counts.mean()
            elif strategy == 'percentile':
                thr = np.percentile(counts, percentile)
            elif strategy == 'iqr':
                q1, q3 = np.percentile(counts, [25, 75])
                iqr = q3 - q1
                thr = q1 - 1.5 * iqr
            elif strategy == 'elbow':
                sorted_counts = counts.sort_values(ascending=False).values
                diffs = np.diff(sorted_counts)
                elbow_idx = np.argmax(diffs)  # индекс наибольшего скачка
                thr = sorted_counts[elbow_idx + 1]  # порог после "локтя"
            else:
                raise ValueError(f"Unknown threshold strategy: {strategy}")

            # Определяем частые категории
            frequent_cats = counts[counts >= thr].index.tolist()

            # Убираем из маппинга значения, соответствующие missing или rare
            if self.missing_label in frequent_cats:
                frequent_cats.remove(self.missing_label)

            # Сохраняем маппинг
            mappings[col] = frequent_cats

        self.mappings_ = mappings

        # Применяем маппинг ко всем колонкам
        for col in self.cat_cols:
            self._df[col] = self._df[col].apply(lambda x: x if x in self.mappings_[col] else self.rare_label)

        return self._df
