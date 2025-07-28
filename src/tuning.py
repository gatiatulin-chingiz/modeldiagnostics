import datetime
import math
import pandas as pd

import mlflow  # pip install mlflow
import numpy as np  # pip install numpy
import optuna  # pip install optuna
from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # pip install catboost
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, KFold

from modeldiagnostics.src.modeldiagnostics import ModelDiagnostics

class TuningHyperparameters:
    def __init__(self, df, features, mvp, experiment_name,
                 run_name="CatboostClassifier", n_trials=100, cv=5, random_seed=42, tags=None, comment=None,
                 split_type="kfold", sort_col=None, date_col=None, train_start=None, train_end=None, test_start=None, test_end=None, target_col="target", task_type="classification", optimize_metric=None):
        self.df = df.copy()
        self.features = features
        self.mvp = mvp
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.n_trials = n_trials
        self.cv = cv
        self.random_seed = random_seed
        self.comment = comment
        self.tags = tags or {}
        self.trials_info = {}
        self.split_type = split_type
        self.sort_col = sort_col
        self.date_col = date_col
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.target_col = target_col
        self.task_type = task_type
        self.optimize_metric = optimize_metric
        self._prepare_tags()
        self.experiment_id = self.get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(experiment_id=self.experiment_id)
        self._split_data()

    def _split_data(self):
        if self.split_type == "timeseries":
            assert self.sort_col is not None, "sort_col must be provided for timeseries split"
            df_sorted = self.df.sort_values(self.sort_col)
            self.X = df_sorted[self.features]
            self.y = df_sorted[self.target_col]
            self.X_test = None
            self.y_test = None
        elif self.split_type == "kfold":
            df_shuffled = self.df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            test_size = int(0.2 * len(df_shuffled))
            df_test = df_shuffled.iloc[:test_size]
            df_train = df_shuffled.iloc[test_size:]
            self.X_train = df_train[self.features]
            self.y_train = df_train[self.target_col]
            self.X_test = df_test[self.features]
            self.y_test = df_test[self.target_col]
        elif self.split_type == "custom_dates":
            assert self.date_col and self.train_start and self.train_end and self.test_start and self.test_end, "date_col, train_start, train_end, test_start, test_end must be provided for custom_dates split"
            df = self.df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            train_mask = (df[self.date_col] >= pd.to_datetime(self.train_start)) & (df[self.date_col] <= pd.to_datetime(self.train_end))
            test_mask = (df[self.date_col] >= pd.to_datetime(self.test_start)) & (df[self.date_col] <= pd.to_datetime(self.test_end))
            df_train = df[train_mask]
            df_test = df[test_mask]
            self.X_train = df_train[self.features]
            self.y_train = df_train[self.target_col]
            self.X_test = df_test[self.features]
            self.y_test = df_test[self.target_col]
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

    def _prepare_tags(self):
        self.tags.update({
            'datetime': str(datetime.datetime.now()),
            'comment': self.comment,
            'model': 'catboost',
            'features': str(self.features),
        })

    @staticmethod
    def get_or_create_experiment(experiment_name):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    @staticmethod
    def champion_callback(study, frozen_trial):
        winner = study.user_attrs.get("winner", None)
        if study.best_value and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner:
                improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
                print(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent: .4f}% improvement"
                )
            else:
                print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

    def _get_cat_features(self):
        return [i for i in (self.mvp.types_dict['BINARY'] + self.mvp.types_dict['CATEGORIAL']) if i in self.features]

    def objective(self, trial):
        print(f'Trial № {str(trial.number)}')
        with mlflow.start_run(run_name=f'Trial № {trial.number}', nested=True):
            if self.task_type == "classification":
                params = {
                            # Основные параметры обучения
                            'iterations': trial.suggest_int('iterations', 100, 2000, step=100),
                            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                            'depth': trial.suggest_int('depth', 3, 12),
                            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                            
                            # Регуляризация
                            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, step=0.5),
                            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No']),
                            
                            # Веса классов и балансировка
                            'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced', None]),
                            
                            # Параметры для разных типов бутстрэпа
                            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0) 
                            if trial.params.get('bootstrap_type') == 'Bayesian' else None,
                            
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0) 
                            if trial.params.get('bootstrap_type') in ['Bernoulli', 'MVS'] else None,
                            
                            # Дополнительные параметры регуляризации
                            'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                            'rsm': trial.suggest_float('rsm', 0.5, 1.0),  # Случайный сэмплинг признаков
                            
                            # Параметры для Lossguide grow policy
                            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20) 
                            if trial.params.get('grow_policy') == 'Lossguide' else None,
                            
                            'max_leaves': trial.suggest_int('max_leaves', 2, 64) 
                            if trial.params.get('grow_policy') == 'Lossguide' else None,
                            
                            # Дополнительные параметры
                            'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
                            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                            'feature_border_type': trial.suggest_categorical('feature_border_type', 
                                                                            ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']),
                            
                            # Скорость обучения и адаптация
                            'learning_rate_decay': trial.suggest_float('learning_rate_decay', 0.8, 1.0),
                            
                            # Работа с категориальными признаками
                            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 255),
                            
                            # Параметры для уменьшения переобучения
                            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
                            
                            # Фиксированные параметры
                            'random_seed': self.random_seed,
                            'verbose': False,
                            'allow_writing_files': False,
                        }
                ModelClass = CatBoostClassifier
            else:
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
                    'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10, step=1),
                    'depth': trial.suggest_int('depth', 2, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                    'random_state': self.random_seed,
                }
                ModelClass = CatBoostRegressor
            cat_features = self._get_cat_features()
            metrics_list = []
            if self.split_type == "timeseries":
                splitter = TimeSeriesSplit(n_splits=self.cv)
                for fold, (train_index, valid_index) in enumerate(splitter.split(self.X)):
                    with mlflow.start_run(run_name=f'Fold № {fold}', nested=True):
                        _X_train = self.X.iloc[train_index].copy()
                        _y_train = self.y.iloc[train_index]
                        _X_valid = self.X.iloc[valid_index].copy()
                        _y_valid = self.y.iloc[valid_index]
                        for col in cat_features:
                            _X_train[col] = _X_train[col].astype(str)
                            _X_valid[col] = _X_valid[col].astype(str)
                        pool = Pool(_X_train, _y_train, cat_features=cat_features, feature_names=list(_X_train.columns))
                        model = ModelClass(**params, verbose=0)
                        model.fit(pool)
                        # Используем ModelDiagnostics
                        diag = ModelDiagnostics(_X_train, _y_train, _X_valid, _y_valid, model, features=self.features, cat_features=cat_features, task_type=self.task_type)
                        train_metrics, valid_metrics = diag.compute_metrics(print_metrics=False)
                        metrics_list.append((train_metrics, valid_metrics))
                # Среднее по фолдам
                avg_valid_metric = np.mean([m[1][self.optimize_metric] for m in metrics_list])
                trial_metrics = metrics_list[-1][1]  # метрики последнего фолда для логирования
            elif self.split_type == "kfold":
                splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_seed)
                for fold, (train_index, valid_index) in enumerate(splitter.split(self.X_train)):
                    with mlflow.start_run(run_name=f'Fold № {fold}', nested=True):
                        _X_train = self.X_train.iloc[train_index].copy()
                        _y_train = self.y_train.iloc[train_index]
                        _X_valid = self.X_train.iloc[valid_index].copy()
                        _y_valid = self.y_train.iloc[valid_index]
                        for col in cat_features:
                            _X_train[col] = _X_train[col].astype(str)
                            _X_valid[col] = _X_valid[col].astype(str)
                        self.X_test[cat_features] = self.X_test[cat_features].astype(str)
                        pool = Pool(_X_train, _y_train, cat_features=cat_features, feature_names=list(_X_train.columns))
                        model = ModelClass(**params, verbose=0)
                        model.fit(pool)
                        diag = ModelDiagnostics(_X_train, _y_train, _X_valid, _y_valid, model, features=self.features, cat_features=cat_features, task_type=self.task_type)
                        train_metrics, valid_metrics = diag.compute_metrics(print_metrics=False)
                        # Тест
                        diag_test = ModelDiagnostics(_X_train, _y_train, self.X_test, self.y_test, model, features=self.features, cat_features=cat_features, task_type=self.task_type)
                        _, test_metrics = diag_test.compute_metrics(print_metrics=False)
                        metrics_list.append((train_metrics, valid_metrics, test_metrics))
                avg_valid_metric = np.mean([m[1][self.optimize_metric] for m in metrics_list])
                trial_metrics = metrics_list[-1][1]  # метрики последнего фолда для логирования
            elif self.split_type == "custom_dates":
                _X_train = self.X_train.copy()
                _y_train = self.y_train
                _X_test = self.X_test.copy()
                _y_test = self.y_test
                for col in cat_features:
                    _X_train[col] = _X_train[col].astype(str)
                    _X_test[col] = _X_test[col].astype(str)
                pool = Pool(_X_train, _y_train, cat_features=cat_features, feature_names=list(_X_train.columns))
                model = ModelClass(**params, verbose=0)
                model.fit(pool)
                diag = ModelDiagnostics(_X_train, _y_train, _X_test, _y_test, model, features=self.features, cat_features=cat_features, task_type=self.task_type)
                train_metrics, test_metrics = diag.compute_metrics(print_metrics=False)
                avg_valid_metric = test_metrics[self.optimize_metric]
                trial_metrics = test_metrics
            else:
                raise ValueError(f"Unknown split_type: {self.split_type}")
            mlflow.log_params(params)
            # Для train/valid
            if 'train_metrics' in locals():
                for k, v in train_metrics.items():
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        mlflow.log_metric(f"{k}_train", v)
            if 'valid_metrics' in locals():
                for k, v in valid_metrics.items():
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        mlflow.log_metric(f"{k}_valid", v)
            # Для test (если есть)
            if 'test_metrics' in locals():
                for k, v in test_metrics.items():
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        mlflow.log_metric(f"{k}_test", v)
            self.trials_info[trial.number] = {
                'train': train_metrics if 'train_metrics' in locals() else None,
                'valid': valid_metrics if 'valid_metrics' in locals() else None,
                'test': test_metrics if 'test_metrics' in locals() else None
            }
            self.tags['datetime'] = str(datetime.datetime.now())
            mlflow.set_tags(tags=self.tags)
            return avg_valid_metric

    def optimize_and_log(self):
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name, nested=True):
            # Логируем информацию о датасете
            dataset_info = {
                'features': self.features,
                'target_col': self.target_col,
                'split_type': self.split_type,
                'n_train': len(self.X_train) if hasattr(self, 'X_train') else (len(self.X) if hasattr(self, 'X') else None),
                'n_valid': len(self.X_valid) if hasattr(self, 'X_valid') else None,
                'n_test': len(self.X_test) if hasattr(self, 'X_test') else None,
                'train_start': str(self.train_start) if hasattr(self, 'train_start') else None,
                'train_end': str(self.train_end) if hasattr(self, 'train_end') else None,
                'test_start': str(self.test_start) if hasattr(self, 'test_start') else None,
                'test_end': str(self.test_end) if hasattr(self, 'test_end') else None,
                'cat_features': self._get_cat_features(),
                'target_unique': list(self.y_train.unique()) if hasattr(self, 'y_train') else None,
                'target_counts': self.y_train.value_counts().to_dict() if hasattr(self, 'y_train') else None,
            }
            mlflow.log_dict(dataset_info, 'dataset_info.json')
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.champion_callback])
            mlflow.set_tags(tags=self.tags)
            mlflow.log_params(study.best_params)
            best_trial = study.best_trial.number
            if self.trials_info:
                # Найти трейл с максимальной метрикой (например, roc_auc/r2)
                def get_metric(trial_metrics):
                    return trial_metrics.get(self.optimize_metric, float('-inf')) if isinstance(trial_metrics, dict) else float('-inf')
                best_key = max(self.trials_info, key=lambda k: get_metric(self.trials_info[k]))
                trial_metrics = self.trials_info[best_key]
                # Логировать метрики строго в порядке train, valid, test
                for stage in ['train', 'valid', 'test']:
                    metrics = trial_metrics.get(stage)
                    if metrics:
                        for key, value in metrics.items():
                            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                                mlflow.log_metric(f'{key}_{stage}', value)
            else:
                print('No trials_info to log!')
            print(f"Best trial: {best_trial}, metrics: {self.trials_info[best_trial]}")
            print(f"Best params: {study.best_params}")

# ===== Пример вызова класса =====
# from tuning import TuningHyperparameters
# # 1. TimeSeriesSplit
# tuner = TuningHyperparameters(
#     df=df,
#     features=features,
#     mvp=mvp,
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     cv=5,
#     split_type="timeseries",
#     sort_col="date_col",
#     target_col="target",
#     task_type="classification",
#     optimize_metric="roc_auc"
# )
# # 2. KFold
# tuner = TuningHyperparameters(
#     df=df,
#     features=features,
#     mvp=mvp,
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     cv=5,
#     split_type="kfold",
#     target_col="target",
#     task_type="regression",
#     optimize_metric="r2"
# )
# # 3. Custom dates
# tuner = TuningHyperparameters(
#     df=df,
#     features=features,
#     mvp=mvp,
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     split_type="custom_dates",
#     date_col="date_col",
#     train_start="2020-01-01",
#     train_end="2021-01-01",
#     test_start="2021-01-02",
#     test_end="2022-01-01",
#     target_col="target",
#     task_type="classification",  # или "regression"
#     optimize_metric="roc_auc"    # или "r2", "mae" и т.д.
# )
# tuner.optimize_and_log()