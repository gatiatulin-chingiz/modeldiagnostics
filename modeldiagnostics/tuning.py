# Требуются библиотеки: mlflow, optuna, numpy, catboost, scikit-learn
import mlflow  # pip install mlflow
import optuna  # pip install optuna
import numpy as np  # pip install numpy
import datetime
from catboost import CatBoostClassifier, Pool  # pip install catboost
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, KFold
import pandas as pd

class CatBoostTuner:
    def __init__(self, df, features, mvp, experiment_name,
                 run_name="CatboostClassifier", n_trials=100, cv=5, random_seed=42, tags=None, comment=None,
                 split_type="kfold", sort_col=None, date_col=None, train_start=None, train_end=None, test_start=None, test_end=None, target_col="target"):
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
    def Gini(y_true, y_pred):
        return 2 * roc_auc_score(y_true, y_pred) - 1

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
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10, step=1),
                'depth': trial.suggest_int('depth', 2, 6),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', None]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'random_state': self.random_seed,
            }
            metrics = {k: [] for k in [
                'precision_train', 'recall_train', 'f1_train', 'roc_auc_train', 'gini_train',
                'precision_valid', 'recall_valid', 'f1_valid', 'roc_auc_valid', 'gini_valid',
                'precision_test', 'recall_test', 'f1_test', 'roc_auc_test', 'gini_test',
            ]}
            cat_features = self._get_cat_features()
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
                        cb_model = CatBoostClassifier(**params, verbose=0)
                        cb_model.fit(pool)
                        # Train
                        predict_train = cb_model.predict(_X_train)
                        proba_train = cb_model.predict_proba(_X_train)[:, 1]
                        metrics['precision_train'].append(precision_score(_y_train, predict_train))
                        metrics['recall_train'].append(recall_score(_y_train, predict_train))
                        metrics['f1_train'].append(f1_score(_y_train, predict_train))
                        metrics['roc_auc_train'].append(roc_auc_score(_y_train, proba_train))
                        metrics['gini_train'].append(self.Gini(_y_train, proba_train))
                        # Valid
                        predict_valid = cb_model.predict(_X_valid)
                        proba_valid = cb_model.predict_proba(_X_valid)[:, 1]
                        metrics['precision_valid'].append(precision_score(_y_valid, predict_valid))
                        metrics['recall_valid'].append(recall_score(_y_valid, predict_valid))
                        metrics['f1_valid'].append(f1_score(_y_valid, predict_valid))
                        metrics['roc_auc_valid'].append(roc_auc_score(_y_valid, proba_valid))
                        metrics['gini_valid'].append(self.Gini(_y_valid, proba_valid))
                # Тест не используется
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
                        cb_model = CatBoostClassifier(**params, verbose=0)
                        cb_model.fit(pool)
                        # Train
                        predict_train = cb_model.predict(_X_train)
                        proba_train = cb_model.predict_proba(_X_train)[:, 1]
                        metrics['precision_train'].append(precision_score(_y_train, predict_train))
                        metrics['recall_train'].append(recall_score(_y_train, predict_train))
                        metrics['f1_train'].append(f1_score(_y_train, predict_train))
                        metrics['roc_auc_train'].append(roc_auc_score(_y_train, proba_train))
                        metrics['gini_train'].append(self.Gini(_y_train, proba_train))
                        # Valid
                        predict_valid = cb_model.predict(_X_valid)
                        proba_valid = cb_model.predict_proba(_X_valid)[:, 1]
                        metrics['precision_valid'].append(precision_score(_y_valid, predict_valid))
                        metrics['recall_valid'].append(recall_score(_y_valid, predict_valid))
                        metrics['f1_valid'].append(f1_score(_y_valid, predict_valid))
                        metrics['roc_auc_valid'].append(roc_auc_score(_y_valid, proba_valid))
                        metrics['gini_valid'].append(self.Gini(_y_valid, proba_valid))
                        # Test
                        predict_test = cb_model.predict(self.X_test)
                        proba_test = cb_model.predict_proba(self.X_test)[:, 1]
                        metrics['precision_test'].append(precision_score(self.y_test, predict_test))
                        metrics['recall_test'].append(recall_score(self.y_test, predict_test))
                        metrics['f1_test'].append(f1_score(self.y_test, predict_test))
                        metrics['roc_auc_test'].append(roc_auc_score(self.y_test, proba_test))
                        metrics['gini_test'].append(self.Gini(self.y_test, proba_test))
            elif self.split_type == "custom_dates":
                # Только train и test, без кроссвалидации
                _X_train = self.X_train.copy()
                _y_train = self.y_train
                _X_test = self.X_test.copy()
                _y_test = self.y_test
                for col in cat_features:
                    _X_train[col] = _X_train[col].astype(str)
                    _X_test[col] = _X_test[col].astype(str)
                pool = Pool(_X_train, _y_train, cat_features=cat_features, feature_names=list(_X_train.columns))
                cb_model = CatBoostClassifier(**params, verbose=0)
                cb_model.fit(pool)
                # Train
                predict_train = cb_model.predict(_X_train)
                proba_train = cb_model.predict_proba(_X_train)[:, 1]
                metrics['precision_train'].append(precision_score(_y_train, predict_train))
                metrics['recall_train'].append(recall_score(_y_train, predict_train))
                metrics['f1_train'].append(f1_score(_y_train, predict_train))
                metrics['roc_auc_train'].append(roc_auc_score(_y_train, proba_train))
                metrics['gini_train'].append(self.Gini(_y_train, proba_train))
                # Test
                predict_test = cb_model.predict(_X_test)
                proba_test = cb_model.predict_proba(_X_test)[:, 1]
                metrics['precision_test'].append(precision_score(_y_test, predict_test))
                metrics['recall_test'].append(recall_score(_y_test, predict_test))
                metrics['f1_test'].append(f1_score(_y_test, predict_test))
                metrics['roc_auc_test'].append(roc_auc_score(_y_test, proba_test))
                metrics['gini_test'].append(self.Gini(_y_test, proba_test))
            else:
                raise ValueError(f"Unknown split_type: {self.split_type}")
            # Средние значения по фолдам (или просто значения для custom_dates)
            trial_metrics = {k: float(np.mean(v)) if len(v) > 0 else None for k, v in metrics.items()}
            self.trials_info[trial.number] = trial_metrics
            mlflow.log_params(params)
            for k, v in trial_metrics.items():
                mlflow.log_metric(k, v)
            self.tags['datetime'] = str(datetime.datetime.now())
            mlflow.set_tags(tags=self.tags)
            # Для timeseries и kfold возвращаем валидационную метрику, для custom_dates — тестовую
            if self.split_type == "custom_dates":
                return trial_metrics['roc_auc_test']
            else:
                return trial_metrics['roc_auc_valid']

    def optimize_and_log(self):
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name, nested=True):
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.champion_callback])
            mlflow.set_tags(tags=self.tags)
            mlflow.log_params(study.best_params)
            best_trial = study.best_trial.number
            for key, value in self.trials_info[best_trial].items():
                mlflow.log_metric(key, value)
            print(f"Best trial: {best_trial}, metrics: {self.trials_info[best_trial]}")
            print(f"Best params: {study.best_params}")

# ===== Пример вызова класса =====
# from tuning import CatBoostTuner
# # 1. TimeSeriesSplit
# tuner = CatBoostTuner(
#     df=df,
#     features=features,
#     mvp=mvp,
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     cv=5,
#     split_type="timeseries",
#     sort_col="date_col",
#     target_col="target"
# )
# # 2. KFold
# tuner = CatBoostTuner(
#     df=df,
#     features=features,
#     mvp=mvp,
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     cv=5,
#     split_type="kfold",
#     target_col="target"
# )
# # 3. Custom dates
# tuner = CatBoostTuner(
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
#     target_col="target"
# )
# tuner.optimize_and_log()