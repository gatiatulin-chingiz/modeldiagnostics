import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, max_error, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score, matthews_corrcoef, average_precision_score,
    precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from shap import TreeExplainer, summary_plot
from scipy.stats import spearmanr
from sklearn.inspection import PartialDependenceDisplay
import statsmodels.api as sm
from scipy import stats



class ModelDiagnostics:
    def __init__(self, X_train, y_train, X_test, y_test, model, features=None, cat_features=None,
                 target_transform=None, fairness=None, task_type=None):
        """
        Инициализация диагностики модели.
        Args:
            X_train: Тренировочные признаки
            y_train: Тренировочная целевая переменная
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная
            model: Обученная модель
            features: Список признаков для использования (по умолчанию все)
            cat_features: Список категориальных признаков
            target_transform (str or None): Преобразование целевой переменной. Поддерживается: 'log1p' или None.
            fairness: Признак для анализа справедливости
            task_type: Тип задачи - 'regression' или 'classification'
        """
        # Проверка task_type
        if task_type is None:
            raise ValueError(
                "Необходимо указать task_type. "
                "Доступные значения:\n"
                "- 'regression' - для задач регрессии\n"
                "- 'classification' - для задач классификации"
            )
        
        if task_type not in ['regression', 'classification']:
            raise ValueError(
                f"Неподдерживаемый task_type: '{task_type}'. "
                "Доступные значения: 'regression' или 'classification'"
            )
        
        # Установка features по умолчанию
        if features is None:
            self.features = list(X_train.columns)
        else:
            self.features = features

        # Установка категориальных признаков
        self.cat_features = cat_features
        self.X_train = X_train
        self.X_test = X_test

        # Сохраняем исходные (непреобразованные) значения таргета
#        self.y_train_raw = y_train.copy()
#        self.y_test_raw = y_test.copy()

        # Используем переданные y_train и y_test как уже преобразованные
        self.y_train = y_train
        self.y_test = y_test

        self.model = model
        self.task_type = task_type
        self.fairness = fairness
        self.RANDOM_STATE = 2025

        # Поддержка обратного преобразования
        self.target_transform = target_transform

    def _inverse_transform(self, values):
        """Обратное преобразование для предсказаний"""
        if self.target_transform == 'log1p':
            return np.expm1(values)
        else:
            return values
    
    def _calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """
        Вычисление Expected Calibration Error (ECE)
        
        Args:
            y_true: истинные метки классов
            y_pred_proba: предсказанные вероятности
            n_bins: количество бинов для разбиения
            
        Returns:
            float: значение ECE
        """
        # Разбиваем предсказания на бины
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Находим индексы предсказаний в текущем бине
            in_bin = np.logical_and(y_pred_proba > bin_lower, y_pred_proba <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Средняя предсказанная вероятность в бине
                mean_pred_prob = np.mean(y_pred_proba[in_bin])
                # Доля истинных положительных результатов в бине
                accuracy_in_bin = np.mean(y_true[in_bin])
                # Количество образцов в бине
                bin_size = np.sum(in_bin)
                
                # Добавляем вклад бина в ECE
                ece += (bin_size / len(y_true)) * np.abs(mean_pred_prob - accuracy_in_bin)
        
                return ece
    
    def _calculate_hosmer_lemeshow_data(self, y_true, y_pred_proba, n_bins=10):
        """
        Вычисление данных для кривых Hosmer-Lemeshow (Gain Chart)
        
        Args:
            y_true: истинные метки классов
            y_pred_proba: предсказанные вероятности
            n_bins: количество бинов для разбиения
            
        Returns:
            dict: данные для построения графиков
        """
        # Преобразуем в numpy arrays для безопасного индексирования
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Сортируем объекты по возрастанию вероятности
        sorted_indices = np.argsort(y_pred_proba)
        sorted_proba = y_pred_proba[sorted_indices]
        sorted_true = y_true[sorted_indices]
        
        # Разбиваем на равные бины
        bin_size = len(sorted_proba) // n_bins
        bin_boundaries = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_proba)
            bin_boundaries.append((start_idx, end_idx))
        
        # Вычисляем метрики для каждого бина
        bin_numbers = []
        mean_pred_proba = []
        empirical_proba = []
        bin_sizes = []
        
        for i, (start_idx, end_idx) in enumerate(bin_boundaries):
            bin_proba = sorted_proba[start_idx:end_idx]
            bin_true = sorted_true[start_idx:end_idx]
            
            # Средняя предсказанная вероятность
            p_g = np.mean(bin_proba)
            # Эмпирическая вероятность (доля положительных)
            e_g = np.mean(bin_true)
            # Размер бина
            size = len(bin_proba)
            
            bin_numbers.append(i + 1)
            mean_pred_proba.append(p_g)
            empirical_proba.append(e_g)
            bin_sizes.append(size)
        
        return {
            'bin_numbers': bin_numbers,
            'mean_pred_proba': mean_pred_proba,
            'empirical_proba': empirical_proba,
            'bin_sizes': bin_sizes
        }
    
    def compute_regression_metrics(self, real_values, predicted_values, ):
            metrics = {}
            metrics['mse'] = mean_squared_error(real_values, predicted_values)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(real_values, predicted_values)

            try:
                metrics['mape'] = mean_absolute_percentage_error(real_values, predicted_values)
            except:
                metrics['mape'] = float('nan')

            metrics['r2'] = r2_score(real_values, predicted_values)
            metrics['max_error'] = max_error(real_values, predicted_values)
            metrics['bias'] = np.mean(real_values - predicted_values)
            metrics['median_error'] = np.median(real_values - predicted_values)

            mask = real_values != 0
            if np.any(mask):
                mpe = np.mean((real_values[mask] - predicted_values[mask]) / real_values[mask])
                metrics['mpe'] = mpe
            else:
                metrics['mpe'] = float('nan')

            return metrics
    
    def compute_classification_metrics(self, real_values, predicted_proba):
            predicted_labels = (predicted_proba >= 0.5).astype(int)
            # Подсчёт TP, TN, FP, FN
            TP = ((real_values == 1) & (predicted_labels == 1)).sum()
            TN = ((real_values == 0) & (predicted_labels == 0)).sum()
            FP = ((real_values == 0) & (predicted_labels == 1)).sum()
            FN = ((real_values == 1) & (predicted_labels == 0)).sum()

            # Вычисление метрик
            try:
                sensitivity = TP / (TP + FN)
            except ZeroDivisionError:
                sensitivity = float('nan')

            try:
                specificity = TN / (TN + FP)
            except ZeroDivisionError:
                specificity = float('nan')

            youden_index = sensitivity + specificity - 1

            metrics = {
                'f1_score': f1_score(real_values, predicted_labels),
                'precision_score': precision_score(real_values, predicted_labels),
                'recall_score': recall_score(real_values, predicted_labels),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'youden_index': youden_index,
                'mcc': matthews_corrcoef(real_values, predicted_labels),
                'shift': predicted_labels.sum() / real_values.sum() if real_values.sum() > 0 else float('nan')
            }

            # Добавляем AUC-ROC, PR-AUC, Gini и ECE, если есть вероятности
            metrics['roc_auc'] = roc_auc_score(real_values, predicted_proba)
            metrics['pr_auc'] = average_precision_score(real_values, predicted_proba)
            metrics['gini'] = 2 * metrics['roc_auc'] - 1
            metrics['ece'] = self._calculate_ece(real_values, predicted_proba)

            return metrics
    
    def compute_metrics(self):
        if self.task_type == 'regression':
            pred_train = self.model.predict(self.X_train[self.features])
            pred_test = self.model.predict(self.X_test[self.features])

            # Обратное преобразование предсказаний
            pred_train = self._inverse_transform(pred_train)
            pred_test = self._inverse_transform(pred_test)

            self.metrics_train = self.compute_regression_metrics(self._inverse_transform(self.y_train), pred_train)
            self.metrics_test = self.compute_regression_metrics(self.y_test, pred_test)
        elif self.task_type == 'classification':
            pred_train_proba = self.model.predict_proba(self.X_train[self.features])[:, 1]
            pred_test_proba = self.model.predict_proba(self.X_test[self.features])[:, 1]
            self.metrics_train = self.compute_classification_metrics(self.y_train, pred_train_proba)
            self.metrics_test = self.compute_classification_metrics(self.y_test, pred_test_proba)
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

        # === Вывод метрик поочередно train/test ===
        for metric_name in self.metrics_train:
            print(f"{metric_name}_train: {self.metrics_train[metric_name]}")
            print(f"{metric_name}_test: {self.metrics_test[metric_name]}")
            print("-" * 50)
        return self.metrics_train, self.metrics_test
    
    def diagnostics_plots(self, real_values, predicted_values, title_prefix=""):
        # Обратное преобразование предсказаний перед построением графиков
        predicted_values = self._inverse_transform(predicted_values)
        if self.task_type == 'regression':
            # --- старая сетка для регрессии ---
            residuals = real_values - predicted_values
            fitted = predicted_values
            title_prefix = title_prefix or "Regression"
            fig, axs = plt.subplots(6, 2, figsize=(16, 26))

            # Графики остатков только для регрессии
            if self.task_type == 'regression':
                n = len(real_values)
                X_fitted = sm.add_constant(fitted)
                hat_matrix = X_fitted @ np.linalg.inv(X_fitted.T @ X_fitted) @ X_fitted.T
                leverage = np.diag(hat_matrix)
                mse = np.mean(residuals ** 2)
                std_residuals = residuals / np.sqrt(mse * (1 - leverage))
                cooks_d = (std_residuals ** 2) / 2 * (leverage / (1 - leverage))

                # Residuals vs Fitted
                sns.scatterplot(x=fitted, y=residuals, ax=axs[0, 0], alpha=0.6)
                axs[0, 0].axhline(y=0, color='r', linestyle='--')
                axs[0, 0].set_title(f'{title_prefix}: Residuals vs Fitted')
                axs[0, 0].set_xlabel('Fitted values')
                axs[0, 0].set_ylabel('Residuals')

                # Normal Q-Q
                sm.qqplot(residuals, line='s', ax=axs[0, 1])
                axs[0, 1].set_title(f'{title_prefix}: Normal Q-Q')

                # Scale-Location
                standardized_residuals = np.sqrt(np.abs(residuals))
                sns.scatterplot(x=fitted, y=standardized_residuals, ax=axs[1, 0], alpha=0.6)
                axs[1, 0].set_title(f'{title_prefix}: Scale-Location')
                axs[1, 0].set_xlabel('Fitted values')
                axs[1, 0].set_ylabel(r'$\sqrt{|\text{Standardized Residuals}|}$')

                # Residuals vs Leverage — обновлённый блок
                threshold_cooks_d = 4 / n  # Динамический порог: 4 / количество наблюдений
                outlier_mask = cooks_d > threshold_cooks_d
                non_outlier_mask = ~outlier_mask

                # Преобразуем в numpy.array для безопасного доступа по индексу
                residuals = np.array(residuals)
                leverage = np.array(leverage)
                cooks_d = np.array(cooks_d)

                sns.scatterplot(
                    x=leverage[non_outlier_mask], 
                    y=residuals[non_outlier_mask],
                    hue=cooks_d[non_outlier_mask],
                    size=cooks_d[non_outlier_mask],
                    sizes=(20, 200),
                    alpha=0.7,
                    palette='viridis',
                    legend='brief',
                    ax=axs[1, 1]
                )

                if outlier_mask.any():
                    sns.scatterplot(
                        x=leverage[outlier_mask],
                        y=residuals[outlier_mask],
                        color='red',
                        edgecolor='black',
                        size=cooks_d[outlier_mask],
                        sizes=(100, 300),
                        alpha=0.9,
                        legend=False,
                        ax=axs[1, 1]
                    )

                axs[1, 1].set_title(f'{title_prefix}: Residuals vs Leverage\n(Size ~ Cook\'s Distance, outliers in red)\n(Threshold: Cook\'s D > 4/n = {threshold_cooks_d:.4f})')
                axs[1, 1].set_xlabel('Leverage')
                axs[1, 1].set_ylabel('Residuals')
                axs[1, 1].axhline(y=0, color='r', linestyle='--')

                if outlier_mask.any():
                    sc = axs[1, 1].scatter([], [], c='red', s=100, label=f'Outliers (Cook\'s D > {threshold_cooks_d:.2f})')
                sc = axs[1, 1].scatter([], [], c='green', s=100, alpha=0.5, label='Normal')
                axs[1, 1].legend()

                # Distribution of residuals
                sns.histplot(residuals, ax=axs[2, 0], kde=True, bins=30, stat='density')
                mean_res = np.mean(residuals)
                std_res = np.std(residuals)
                x = np.linspace(*axs[2, 0].get_xlim(), 100)
                axs[2, 0].plot(x, stats.norm.pdf(x, mean_res, std_res), 'b-', lw=2)
                axs[2, 0].set_title(f'{title_prefix}: Distribution of Residuals\n(Histogram + KDE)')
                axs[2, 0].set_xlabel('Residual Value')
                axs[2, 0].set_ylabel('Density')
            else:
                # Отключаем графики остатков для классификации
                axs[0, 0].axis('off')
                axs[0, 1].axis('off')
                axs[1, 0].axis('off')
                axs[1, 1].axis('off')
                axs[2, 0].axis('off')

            # ROC vs PR Curves Comparison (график №1)
            if self.task_type == 'classification':
                # Вычисляем ROC curve
                fpr, tpr, _ = roc_curve(real_values, predicted_values)
                roc_auc = roc_auc_score(real_values, predicted_values)
                
                # График сравнения ROC и PR кривых
                axs[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
                
                # Добавляем PR curve для сравнения
                precision, recall, _ = precision_recall_curve(real_values, predicted_values)
                pr_auc = average_precision_score(real_values, predicted_values)
                axs[0, 0].plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
                
                axs[0, 0].set_xlabel('Recall / True Positive Rate')
                axs[0, 0].set_ylabel('Precision / True Positive Rate')
                axs[0, 0].set_title(f'{title_prefix}: ROC vs PR Curves Comparison')
                axs[0, 0].legend()
                axs[0, 0].grid(True, alpha=0.3)
                axs[0, 0].set_xlim([0, 1])
                axs[0, 0].set_ylim([0, 1])
            else:
                axs[0, 0].axis('off')

            # Calibration Curve (график №2)
            if self.task_type == 'classification':
                # Используем sklearn для построения калибровочной кривой
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    real_values, predicted_values, n_bins=10, strategy='uniform'
                )
                
                # Идеальная калибровка (диагональ)
                axs[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)
                
                # Калибровочная кривая модели
                axs[0, 1].plot(mean_predicted_value, fraction_of_positives, 'bo-', 
                              linewidth=2, markersize=8, label='Model Calibration')
                
                axs[0, 1].set_xlabel('Mean Predicted Probability')
                axs[0, 1].set_ylabel('Fraction of Positives')
                axs[0, 1].set_title(f'{title_prefix}: Calibration Curve (Reliability Diagram)')
                axs[0, 1].legend()
                axs[0, 1].grid(True, alpha=0.3)
                axs[0, 1].set_xlim([0, 1])
                axs[0, 1].set_ylim([0, 1])
            else:
                axs[0, 1].axis('off')

            # Объединенный график: Distribution + Hosmer-Lemeshow (график №3)
            if self.task_type == 'classification':
                # Получаем данные для кривых Hosmer-Lemeshow
                hl_data = self._calculate_hosmer_lemeshow_data(real_values, predicted_values, n_bins=10)
                
                # Создаем график как на картинке
                ax = axs[0, 2]
                
                # График 1: Столбцы - доля наблюдаемых событий (эмпирическая вероятность)
                bars = ax.bar(hl_data['bin_numbers'], hl_data['empirical_proba'], 
                             alpha=0.7, color='skyblue', edgecolor='black', 
                             label='Доля наблюдаемых событий')
                
                # График 2: Линия - средние предсказанные вероятности
                ax.plot(hl_data['bin_numbers'], hl_data['mean_pred_proba'], 
                       'o-', color='orange', linewidth=2, markersize=8, 
                       label='Средние предсказанные вероятности')
                
                ax.set_xlabel('Бин (группа)')
                ax.set_ylabel('Доля / Средняя вероятность')
                ax.set_title(f'{title_prefix}: Доля наблюдаемых событий и средние предсказанные вероятности по бинам')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks(hl_data['bin_numbers'])
                ax.set_ylim([0, 1])
            else:
                axs[0, 2].axis('off')
                axs[0, 3].axis('off')

            # Отключаем строку 4 для классификации
            if self.task_type == 'classification':
                axs[1, 0].axis('off')
                axs[1, 1].axis('off')
            else:
                axs[1, 0].axis('off')
                axs[1, 1].axis('off')

            # Отключаем строку 5 для классификации
            if self.task_type == 'classification':
                axs[1, 2].axis('off')
                axs[1, 3].axis('off')
            else:
                axs[1, 2].axis('off')
                axs[1, 3].axis('off')

            plt.tight_layout()
            plt.suptitle(f'{title_prefix} Diagnostic Plots', y=1.02)
            plt.subplots_adjust(bottom=0.2)  # Добавляем место для таблицы
            plt.show()

            # === Вывод таблицы с аномальными точками (только для регрессии) ===
            if self.task_type == 'regression':
                if outlier_mask.any():
                    anomaly_indices = np.where(outlier_mask)[0]
                    anomaly_data = pd.DataFrame({
                        'Index': anomaly_indices,
                        'Leverage': leverage[outlier_mask],
                        'Residual': residuals[outlier_mask],
                        'Cook\'s D': cooks_d[outlier_mask]
                    }).sort_values(by='Cook\'s D', ascending=False).head(10)
                    print(f"\n=== Таблица аномальных точек (Cook's D > {threshold_cooks_d:.4f}), топ-10 ===")
                    print(anomaly_data.to_string(index=False))
                    return anomaly_data
                else:
                    print(f"\n=== Аномальных точек не найдено (Cook's D > {threshold_cooks_d:.4f}) ===")
                    return pd.DataFrame()
            else:
                # Для классификации возвращаем пустой DataFrame
                return pd.DataFrame()
        elif self.task_type == 'classification':
            title_prefix = title_prefix or "Classification"
            # --- компактная сетка для классификации ---
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            # axs[0]: ROC vs PR Curves Comparison
            # axs[1]: Calibration Curve
            # axs[2]: Distribution + Hosmer-Lemeshow

            # 1. ROC vs PR Curves Comparison
            fpr, tpr, _ = roc_curve(real_values, predicted_values)
            roc_auc = roc_auc_score(real_values, predicted_values)
            precision, recall, _ = precision_recall_curve(real_values, predicted_values)
            pr_auc = average_precision_score(real_values, predicted_values)
            axs[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axs[0].plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
            axs[0].set_xlabel('Recall / True Positive Rate')
            axs[0].set_ylabel('Precision / True Positive Rate')
            axs[0].set_title(f'{title_prefix}: ROC vs PR Curves')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            axs[0].set_xlim([0, 1])
            axs[0].set_ylim([0, 1])

            # 2. Calibration Curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                real_values, predicted_values, n_bins=10, strategy='uniform'
            )
            axs[1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)
            axs[1].plot(mean_predicted_value, fraction_of_positives, 'bo-',
                        linewidth=2, markersize=8, label='Model Calibration')
            axs[1].set_xlabel('Mean Predicted Probability')
            axs[1].set_ylabel('Fraction of Positives')
            axs[1].set_title(f'{title_prefix}: Calibration Curve')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            axs[1].set_xlim([0, 1])
            axs[1].set_ylim([0, 1])

            # 3. Distribution + Hosmer-Lemeshow
            hl_data = self._calculate_hosmer_lemeshow_data(real_values, predicted_values, n_bins=10)
            ax = axs[2]
            bars = ax.bar(hl_data['bin_numbers'], hl_data['empirical_proba'],
                         alpha=0.7, color='skyblue', edgecolor='black',
                         label='Доля наблюдаемых событий')
            ax.plot(hl_data['bin_numbers'], hl_data['mean_pred_proba'],
                    'o-', color='orange', linewidth=2, markersize=8,
                    label='Средние предсказанные вероятности')
            ax.set_xlabel('Бин (группа)')
            ax.set_ylabel('Доля / Средняя вероятность')
            ax.set_title(f'{title_prefix}: Доля наблюдаемых событий и средние предсказанные вероятности по бинам')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(hl_data['bin_numbers'])
            ax.set_ylim([0, 1])

            plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
            plt.subplots_adjust(top=0.85, bottom=0.18)
            plt.suptitle(f'{title_prefix} Diagnostic Plots', y=1.02)
            plt.show()
            return pd.DataFrame()  # Для классификации не возвращаем аномалии
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

    def test_normality_kolmogorov(self, real_values, predicted_values, verbose=True):
        """
        Проверка нормальности остатков с использованием критерия Колмогорова-Смирнова.

        Args:
            real_values (np.array or pd.Series): Реальные значения целевой переменной.
            predicted_values (np.array or pd.Series): Предсказанные моделью значения.
            verbose (bool): Если True — выводит результаты на экран.

        Returns:
            dict: {'statistic', 'p_value', 'is_normal'}
        """
        residuals = real_values - predicted_values

        # Нормируем остатки (стандартизация)
        mean = np.mean(residuals)
        std = np.std(residuals, ddof=1)
        normalized_residuals = (residuals - mean) / std

        # Тест Колмогорова-Смирнова против стандартного нормального распределения
        statistic, p_value = stats.kstest(normalized_residuals, 'norm')

        # Интерпретация результата
        is_normal = p_value > 0.05

        if verbose:
            print(f"Статистика теста: {statistic:.4f}")
            print(f"P-value: {p_value:.4f}")
            if is_normal:
                print("Остатки соответствуют нормальному распределению.")
            else:
                print("Остатки НЕ соответствуют нормальному распределению.")
            print()
        return {'statistic': statistic, 'p_value': p_value, 'is_normal': is_normal}
    
    def test_heteroscedasticity(self, real_values, predicted_values, verbose=True):
        """
        Тестирование гетероскедастичности остатков с использованием 5 бинов.

        Args:
            real_values (np.array or pd.Series): Реальные значения целевой переменной
            predicted_values (np.array or pd.Series): Предсказанные моделью значения
            verbose (bool): Если True — выводит результаты на экран

        Returns:
            dict: Результаты тестов между всеми бинами
        """
        residuals = real_values - predicted_values

        # Разделение на 5 квантильных бинов по предсказаниям
        bins = pd.qcut(predicted_values, q=5, duplicates='drop')
        binned_residuals = [residuals[bins == cat] for cat in np.unique(bins)]

        # Проверяем, что есть хотя бы два бина для тестирования
        if len(binned_residuals) < 2:
            raise ValueError("Недостаточно бинов для тестирования гетероскедастичности")

        # Bartlett's Test
        stat_bartlett, p_bartlett = stats.bartlett(*binned_residuals)

        # Levene's Test
        stat_levene, p_levene = stats.levene(*binned_residuals)

        # Fligner-Killeen Test
        stat_fligner, p_fligner = stats.fligner(*binned_residuals)

        results = {
            'bartlett_pvalue': p_bartlett,
            'levene_pvalue': p_levene,
            'fligner_pvalue': p_fligner,
            'is_heteroscedastic': any(p < 0.05 for p in [p_bartlett, p_levene, p_fligner])
        }

        if verbose:
            print("=== Тесты на гетероскедастичность (разбиение на 5 бинов) ===")
            print(f"Bartlett's test p-value: {p_bartlett:.4f} {'(Гетероскедастичность)' if p_bartlett < 0.05 else '(Гомоскедастичность)'}")
            print(f"Levene's test p-value:   {p_levene:.4f} {'(Гетероскедастичность)' if p_levene < 0.05 else '(Гомоскедастичность)'}")
            print(f"Fligner-Killeen test p-value: {p_fligner:.4f} {'(Гетероскедастичность)' if p_fligner < 0.05 else '(Гомоскедастичность)'}")
            print("=== Конец тестов ===\n")

        return results
    
    def adversarial_validation(self, plot=True, val_size=0.3, val_top_k_quantile=0.3):
            train_data = self.X_train.copy()
            test_data = self.X_test.copy()

            train_data = train_data[self.features]
            test_data = test_data[self.features]

            if self.cat_features:
                cat_indices = [i for i, f in enumerate(self.features) if f in self.cat_features]
            else:
                cat_indices = []

            X = pd.concat([train_data, test_data], axis=0)
            y = np.array([0] * len(train_data) + [1] * len(test_data))
#            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=self.RANDOM_STATE, stratify=y)

            model = CatBoostClassifier(iterations=200, verbose=0, random_seed=self.RANDOM_STATE, cat_features=cat_indices or None)
            model.fit(X, y, use_best_model=True)

            y_pred = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred)

            feature_importances = pd.DataFrame({'feature': self.features, 'importance': model.get_feature_importance()})
            feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)

            train_scores = model.predict_proba(train_data)[:, 1]
            train_with_scores = pd.DataFrame({'index': train_data.index, 'score': train_scores}).sort_values(by='score', ascending=False)
            n_val_samples = int(len(train_data) * val_top_k_quantile)
            val_indices = train_with_scores.head(n_val_samples)['index'].values

            if plot:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importances.head(20), x='importance', y='feature', palette='viridis')
                plt.title(f'Adversarial Validation Feature Importance\n(AUC = {auc:.4f})')
                plt.xlabel('Важность')
                plt.ylabel('Признак')
                plt.tight_layout()
                plt.show()

            return {
                'auc': auc,
                'feature_importances': feature_importances,
                'val_indices': val_indices
            }
    
    def analyze_top_errors(self, top_percent=0.1, error_type='abs', plot=True):
        if self.task_type == 'regression':
            predicted = self.model.predict(self.X_train[self.features])
            errors = self._inverse_transform(self.y_train) - predicted

            if error_type == 'abs':
                error_metric = np.abs(errors)
            elif error_type == 'positive':
                error_metric = np.where(errors < 0, np.abs(errors), 0)
            elif error_type == 'negative':
                error_metric = np.where(errors > 0, np.abs(errors), 0)
            else:
                raise ValueError("error_type должен быть 'abs', 'positive' или 'negative'")
        elif self.task_type == 'classification':
            # Для классификации используем log-loss как метрику ошибки
            predicted_proba = self.model.predict_proba(self.X_train[self.features])[:, 1]
            eps = 1e-15  # Защита от log(0)
            predicted_proba = np.clip(predicted_proba, eps, 1 - eps)
            
            # Log-loss для каждого образца
            log_loss_per_sample = -(self.y_train * np.log(predicted_proba) + 
                                   (1 - self.y_train) * np.log(1 - predicted_proba))
            
            if error_type == 'abs':
                error_metric = log_loss_per_sample
            elif error_type == 'positive':
                # Ошибки для положительных классов (где y_true = 1)
                error_metric = np.where(self.y_train == 1, log_loss_per_sample, 0)
            elif error_type == 'negative':
                # Ошибки для отрицательных классов (где y_true = 0)
                error_metric = np.where(self.y_train == 0, log_loss_per_sample, 0)
            else:
                raise ValueError("error_type должен быть 'abs', 'positive' или 'negative'")
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

        threshold = np.quantile(error_metric, 1 - top_percent)
        labels = (error_metric >= threshold).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train[self.features], labels, test_size=0.3, random_state=42, stratify=labels
        )

        cat_indices = [i for i, f in enumerate(self.features) if f in (self.cat_features or [])]
        model = CatBoostClassifier(iterations=200, verbose=0, random_seed=42, cat_features=cat_indices or None)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        feature_importances = pd.DataFrame({'feature': self.features, 'importance': model.get_feature_importance()})
        feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)

        explainer = TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train[self.features])
        shap_data = self.X_train[self.features].values

        # Получаем топ-1 важный признак
        top_feature = feature_importances.iloc[0]['feature']

        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importances.head(20), x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance (ошибки модели)\n(AUC = {auc:.4f})')
            plt.xlabel('Важность')
            plt.ylabel('Признак')
            plt.tight_layout()
            plt.show()

            summary_plot(shap_values, shap_data, feature_names=self.features, plot_type="bar", show=False)
            plt.suptitle('SHAP Feature Importance (ошибки модели)')
            plt.tight_layout()
            plt.show()

            # === Новые графики: распределение топ-признака на train и test ===
            is_numeric = pd.api.types.is_numeric_dtype(self.X_train[top_feature])

            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

            if is_numeric:
                # Числовой признак: гистограмма с 30 бинами
                bins = 30
                sns.histplot(self.X_train[top_feature], ax=axes[0], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
                sns.histplot(self.X_test[top_feature], ax=axes[1], bins=bins, color='salmon', edgecolor='black', alpha=0.7)
                axes[0].set_title(f'Train: Распределение "{top_feature}" (числовой, {bins} бинов)')
                axes[1].set_title(f'Test: Распределение "{top_feature}" (числовой, {bins} бинов)')
                axes[0].set_xlabel('Значение признака')
                axes[1].set_xlabel('Значение признака')
                axes[0].set_ylabel('Количество')
            else:
                # Категориальный признак: подсчёт количества по категориям
                train_counts = self.X_train[top_feature].value_counts()
                test_counts = self.X_test[top_feature].value_counts()
                train_counts.sort_index().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black', alpha=0.7)
                test_counts.sort_index().plot(kind='bar', ax=axes[1], color='salmon', edgecolor='black', alpha=0.7)
                axes[0].set_title(f'Train: Распределение "{top_feature}" (категориальный)')
                axes[1].set_title(f'Test: Распределение "{top_feature}" (категориальный)')
                axes[0].set_xlabel('Категория')
                axes[1].set_xlabel('Категория')
                axes[0].set_ylabel('Количество')

            plt.tight_layout()
            plt.suptitle(f'Распределение топ-признака "{top_feature}" в Train и Test', y=1.05)
            plt.show()

    def plot_fairness_by_group(self, X, y, group_col, plot=True):
        sns.set_theme(style="whitegrid", font_scale=1.15)
        df = X.copy()
        df['target'] = y
        
        if self.task_type == 'regression':
            df['prediction'] = self.model.predict(df[self.features])
            df['error'] = df['target'] - df['prediction']
            df['abs_error'] = abs(df['error'])
        elif self.task_type == 'classification':
            df['prediction_proba'] = self.model.predict_proba(df[self.features])[:, 1]
            df['prediction'] = (df['prediction_proba'] >= 0.5).astype(int)
            # Для классификации используем log-loss как метрику ошибки
            eps = 1e-15
            df['prediction_proba'] = np.clip(df['prediction_proba'], eps, 1 - eps)
            df['error'] = -(df['target'] * np.log(df['prediction_proba']) + 
                           (1 - df['target']) * np.log(1 - df['prediction_proba']))
            df['abs_error'] = abs(df['error'])
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

        is_numeric = pd.api.types.is_numeric_dtype(df[group_col])

        if is_numeric:
            min_val = df[group_col].min()
            max_val = df[group_col].max()
            # шаг 25 000, добавим небольшой запас к max
            bins = np.arange(min_val, max_val + 25001, 25000)
            df[group_col + '_bin'] = pd.cut(df[group_col], bins=bins, right=False, include_lowest=True)
            group_col_bin = group_col + '_bin'
            grouped = df.groupby(group_col_bin, observed=True).agg(
                mae=('abs_error', 'mean'),
                bias=('error', 'mean'),
                count=(group_col_bin, 'size')
            ).reset_index()
            x_col = group_col_bin
            xticks_labels = grouped[x_col].astype(str)
            title_suffix = '(численный признак, бины по 25 000)'
        else:
            x_col = group_col
            grouped = df.groupby(group_col).agg(
                mae=('abs_error', 'mean'),
                bias=('error', 'mean'),
                count=(group_col, 'size')
            ).reset_index()
            grouped = grouped.sort_values(by=group_col)
            xticks_labels = grouped[x_col].astype(str)
            title_suffix = '(категориальный признак)'

        if plot:
            palette = sns.color_palette()
            mae_color = palette[0]
            bias_pos_color = palette[2]
            bias_neg_color = palette[3]
            bias_line_color = palette[1]

            fig, ax1 = plt.subplots(figsize=(16, 9))
            bars = ax1.bar(xticks_labels, grouped['mae'],
                           color=mae_color, edgecolor='black',
                           alpha=0.85, label='MAE', width=0.68)

            ax1.set_xlabel(f'Группа "{group_col}"', fontsize=14, fontweight='bold')
            ax1.set_ylabel('MAE', color=mae_color, fontsize=13, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=mae_color, labelsize=12)
            ax1.tick_params(axis='x', labelsize=12)
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=10, fontweight="bold", color=mae_color)

            ax1.yaxis.grid(True, linestyle='--', alpha=0.32)
            ax1.xaxis.grid(False)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Bias (смещение)\n(положительное/отрицательное)', color=bias_line_color, fontsize=13, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=bias_line_color, labelsize=12)

            bias_values = grouped['bias']
            ax2.plot(xticks_labels, bias_values, color=bias_line_color, marker='o', linestyle='-', linewidth=2.2, label='Bias')

            # Точки Bias, БЕЗ ЦИФР!
            for x, y in zip(xticks_labels, bias_values):
                color = bias_pos_color if y > 0 else bias_neg_color
                ax2.scatter(x, y, color=color, s=135, zorder=5, alpha=0.98, edgecolor='black', linewidth=1.22)

            ax1.set_title(f'Анализ справедливости (Fairness): "{group_col}" {title_suffix}', fontsize=16, fontweight='bold', pad=22)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11)

            plt.tight_layout()
            plt.show()

        return grouped
    
    def run_full_diagnostics(self):
        print("=== Вычисление метрик ===")
        metrics_train, metrics_test = self.compute_metrics()
        print()
        print("=== Диагностические графики ===")
        if self.task_type == 'regression':
            self.diagnostics_plots(self.y_train, self.model.predict(self.X_train[self.features]), title_prefix="Train")
            self.diagnostics_plots(self.y_test, self.model.predict(self.X_test[self.features]), title_prefix="Test")
            print("=== Тест Колмогорова-Смирнова на нормальность остатков ===")
            self.test_normality_kolmogorov(self.y_test, self.model.predict(self.X_test[self.features]))
            print("=== Тест на гетероскедастичность ===")
            self.test_heteroscedasticity(self.y_test, self.model.predict(self.X_test[self.features]))
        elif self.task_type == 'classification':
            pred_proba_train = self.model.predict_proba(self.X_train[self.features])[:, 1]
            self.diagnostics_plots(self.y_train, pred_proba_train, title_prefix="Train")
            pred_proba_test = self.model.predict_proba(self.X_test[self.features])[:, 1]
            self.diagnostics_plots(self.y_test, pred_proba_test, title_prefix="Test")
        print("=== Adversarial Validation ===")
        self.adversarial_validation()
        print("=== Анализ топ-ошибок ===")
        self.analyze_top_errors()
        if self.fairness:
            print("=== Fairness анализ ===")
            self.plot_fairness_by_group(self.X_train, self.y_train, group_col=self.fairness)
            self.plot_fairness_by_group(self.X_test, self.y_test, group_col=self.fairness)