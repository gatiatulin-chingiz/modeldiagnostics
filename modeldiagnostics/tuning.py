# Требуются библиотеки: mlflow, optuna, numpy, catboost, scikit-learn
import mlflow  # pip install mlflow
import optuna  # pip install optuna
import numpy as np  # pip install numpy
import datetime
from catboost import CatBoostClassifier, Pool  # pip install catboost
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, KFold

class CatBoostTuner:
    def __init__(self, X_train, y_train, X_test, y_test, summary, mvp, experiment_name,
                 run_name="CatboostClassifier", n_trials=100, cv=5, random_seed=42, tags=None, comment=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.summary = summary  # список признаков
        self.mvp = mvp          # список категориальных/бинарных признаков
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.n_trials = n_trials
        self.cv = cv
        self.random_seed = random_seed
        self.comment = comment
        self.tags = tags or {}
        self.trials_info = {}
        self._prepare_tags()
        self.experiment_id = self.get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(experiment_id=self.experiment_id)

    def _prepare_tags(self):
        self.tags.update({
            'datetime': str(datetime.datetime.now()),
            'comment': self.comment,
            'model': 'catboost',
            'features': str(self.summary),
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
        # mvp должен содержать словарь с ключами 'BINARY' и 'CATEGORIAL', каждый из которых - список признаков
        if isinstance(self.mvp, dict):
            return [i for i in (self.mvp.get('BINARY', []) + self.mvp.get('CATEGORIAL', [])) if i in self.summary]
        # если mvp - просто список, возвращаем пересечение
        return [i for i in self.mvp if i in self.summary]

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
            tscv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_seed)
            cat_features = self._get_cat_features()
            for fold, (train_index, valid_index) in enumerate(tscv.split(self.X_train)):
                with mlflow.start_run(run_name=f'KFold № {fold}', nested=True):
                    _X_train = self.X_train.iloc[train_index]
                    _y_train = self.y_train.iloc[train_index]
                    _X_valid = self.X_train.iloc[valid_index]
                    _y_valid = self.y_train.iloc[valid_index]
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
                    # Логируем метрики для каждого фолда
                    for k in metrics:
                        mlflow.log_metric(k, metrics[k][-1])
                    mlflow.log_params(params)
            # Средние значения по фолдам
            trial_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
            self.trials_info[trial.number] = trial_metrics
            mlflow.log_params(params)
            for k, v in trial_metrics.items():
                mlflow.log_metric(k, v)
            self.tags['datetime'] = str(datetime.datetime.now())
            mlflow.set_tags(tags=self.tags)
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
# tuner = CatBoostTuner(
#     X_train, y_train, X_test, y_test,
#     summary=summary,
#     mvp=mvp,
#     comment='claim_probability',te
#     experiment_name="claim_probability_2",
#     run_name="CatboostClassifier",
#     n_trials=100,
#     cv=5
# )
# tuner.optimize_and_log()