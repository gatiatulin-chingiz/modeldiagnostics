import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from scr.modeldiagnostics import ModelDiagnostics

def test_classification_gini():
    """Тест метрики Gini для классификации"""
    print("Тестирование метрики Gini для классификации...")
    
    # Создаем синтетические данные для классификации
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Создаем объект диагностики
    diagnostics = ModelDiagnostics(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model=model, task_type='classification'
    )
    
    # Вычисляем метрики
    metrics_train, metrics_test = diagnostics.compute_metrics(print_metrics=True)
    
    print(f"Gini для train: {metrics_train['gini']:.4f}")
    print(f"Gini для test: {metrics_test['gini']:.4f}")
    print(f"ROC-AUC для train: {metrics_train['roc_auc']:.4f}")
    print(f"ROC-AUC для test: {metrics_test['roc_auc']:.4f}")
    print("-" * 50)

def test_regression_gini():
    """Тест метрики Gini для регрессии"""
    print("Тестирование метрики Gini для регрессии...")
    
    # Создаем синтетические данные для регрессии
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, 
                          random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Создаем объект диагностики
    diagnostics = ModelDiagnostics(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model=model, task_type='regression'
    )
    
    # Вычисляем метрики
    metrics_train, metrics_test = diagnostics.compute_metrics(print_metrics=True)
    
    print(f"Gini для train: {metrics_train['gini']:.4f}")
    print(f"Gini для test: {metrics_test['gini']:.4f}")
    print(f"R2 для train: {metrics_train['r2']:.4f}")
    print(f"R2 для test: {metrics_test['r2']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    test_classification_gini()
    test_regression_gini()
    print("Тестирование завершено!") 