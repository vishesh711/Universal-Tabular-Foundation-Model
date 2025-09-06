"""Baseline models for comparison with TabGPT."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
import warnings

import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type
        self.model = None
        self.preprocessor = None
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying model."""
        pass
    
    @abstractmethod
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create data preprocessor."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the model."""
        # Create preprocessor if not exists
        if self.preprocessor is None:
            self.preprocessor = self._create_preprocessor(X)
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model(**kwargs)
        
        # Preprocess data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Fit model
        self.model.fit(X_processed, y)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame):
        """Get prediction probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline model."""
    
    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__("RandomForest", task_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create Random Forest model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Random Forest baseline")
        
        # Merge default and provided kwargs
        model_kwargs = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        if self.task_type == "classification":
            return RandomForestClassifier(**model_kwargs)
        else:
            return RandomForestRegressor(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for Random Forest."""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create transformers
        transformers = []
        
        if numeric_features:
            # Random Forest doesn't need scaling, but handle missing values
            from sklearn.impute import SimpleImputer
            numeric_transformer = SimpleImputer(strategy='median')
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            # Use ordinal encoding for categorical features (RF can handle it)
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OrdinalEncoder
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            # No preprocessing needed
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')


class XGBoostBaseline(BaselineModel):
    """XGBoost baseline model."""
    
    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__("XGBoost", task_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost required for XGBoost baseline")
        
        # Merge default and provided kwargs
        model_kwargs = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        if self.task_type == "classification":
            return xgb.XGBClassifier(**model_kwargs)
        else:
            return xgb.XGBRegressor(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for XGBoost."""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            from sklearn.impute import SimpleImputer
            numeric_transformer = SimpleImputer(strategy='median')
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            # Use ordinal encoding for XGBoost
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OrdinalEncoder
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')


class LogisticRegressionBaseline(BaselineModel):
    """Logistic Regression baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression", "classification")
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create Logistic Regression model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Logistic Regression baseline")
        
        model_kwargs = {
            'random_state': 42,
            'max_iter': 1000
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        return LogisticRegression(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for Logistic Regression."""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            # Scale numeric features and handle missing values
            from sklearn.impute import SimpleImputer
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            # One-hot encode categorical features
            from sklearn.impute import SimpleImputer
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')


class LinearRegressionBaseline(BaselineModel):
    """Linear Regression baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("LinearRegression", "regression")
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create Linear Regression model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Linear Regression baseline")
        
        model_kwargs = {}
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        return LinearRegression(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for Linear Regression."""
        # Same as Logistic Regression
        return LogisticRegressionBaseline()._create_preprocessor(X)


class GradientBoostingBaseline(BaselineModel):
    """Gradient Boosting baseline model."""
    
    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__("GradientBoosting", task_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create Gradient Boosting model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Gradient Boosting baseline")
        
        model_kwargs = {
            'n_estimators': 100,
            'random_state': 42
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        if self.task_type == "classification":
            return GradientBoostingClassifier(**model_kwargs)
        else:
            return GradientBoostingRegressor(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for Gradient Boosting."""
        # Same as Random Forest
        return RandomForestBaseline(self.task_type)._create_preprocessor(X)


class LightGBMBaseline(BaselineModel):
    """LightGBM baseline model."""
    
    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__("LightGBM", task_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required for LightGBM baseline")
        
        model_kwargs = {
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        if self.task_type == "classification":
            return lgb.LGBMClassifier(**model_kwargs)
        else:
            return lgb.LGBMRegressor(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for LightGBM."""
        # Same as XGBoost
        return XGBoostBaseline(self.task_type)._create_preprocessor(X)


class CatBoostBaseline(BaselineModel):
    """CatBoost baseline model."""
    
    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__("CatBoost", task_type)
        self.model_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create CatBoost model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost required for CatBoost baseline")
        
        model_kwargs = {
            'iterations': 100,
            'random_state': 42,
            'verbose': False
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs.update(kwargs)
        
        if self.task_type == "classification":
            return cb.CatBoostClassifier(**model_kwargs)
        else:
            return cb.CatBoostRegressor(**model_kwargs)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for CatBoost."""
        # CatBoost can handle categorical features natively
        # Just handle missing values
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            from sklearn.impute import SimpleImputer
            numeric_transformer = SimpleImputer(strategy='median')
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            from sklearn.impute import SimpleImputer
            categorical_transformer = SimpleImputer(strategy='constant', fill_value='missing')
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')


def create_baseline_models(
    task_type: str = "classification",
    include_models: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, BaselineModel]:
    """Create a collection of baseline models."""
    
    available_models = {
        'RandomForest': RandomForestBaseline,
        'LogisticRegression': LogisticRegressionBaseline,
        'LinearRegression': LinearRegressionBaseline,
        'GradientBoosting': GradientBoostingBaseline,
    }
    
    # Add optional models if available
    if XGBOOST_AVAILABLE:
        available_models['XGBoost'] = XGBoostBaseline
    
    if LIGHTGBM_AVAILABLE:
        available_models['LightGBM'] = LightGBMBaseline
    
    if CATBOOST_AVAILABLE:
        available_models['CatBoost'] = CatBoostBaseline
    
    # Filter models based on task type
    if task_type == "classification":
        # Remove regression-only models
        available_models.pop('LinearRegression', None)
    elif task_type == "regression":
        # Remove classification-only models
        available_models.pop('LogisticRegression', None)
    
    # Filter by include_models if specified
    if include_models is not None:
        available_models = {
            name: cls for name, cls in available_models.items()
            if name in include_models
        }
    
    # Create model instances
    models = {}
    for name, model_class in available_models.items():
        try:
            if name in ['LogisticRegression', 'LinearRegression']:
                model = model_class(**kwargs)
            else:
                model = model_class(task_type=task_type, **kwargs)
            models[name] = model
        except Exception as e:
            logger.warning(f"Could not create {name} baseline: {e}")
            continue
    
    return models


def get_default_baselines(task_type: str = "classification") -> Dict[str, BaselineModel]:
    """Get default baseline models for a task type."""
    
    if task_type == "classification":
        default_models = ['RandomForest', 'LogisticRegression', 'XGBoost']
    elif task_type == "regression":
        default_models = ['RandomForest', 'LinearRegression', 'XGBoost']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return create_baseline_models(
        task_type=task_type,
        include_models=default_models
    )


class EnsembleBaseline(BaselineModel):
    """Ensemble of multiple baseline models."""
    
    def __init__(self, models: List[BaselineModel], task_type: str, voting: str = 'soft'):
        super().__init__("Ensemble", task_type)
        self.base_models = models
        self.voting = voting
    
    def _create_model(self, **kwargs):
        """Create ensemble model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for ensemble baseline")
        
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        # Prepare estimators
        estimators = [(model.name, model) for model in self.base_models]
        
        if self.task_type == "classification":
            return VotingClassifier(estimators=estimators, voting=self.voting)
        else:
            return VotingRegressor(estimators=estimators)
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessor for ensemble."""
        # Use the preprocessor from the first model
        if self.base_models:
            return self.base_models[0]._create_preprocessor(X)
        else:
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit ensemble model."""
        # Fit each base model individually first
        for model in self.base_models:
            model.fit(X, y, **kwargs)
        
        # Then fit the ensemble
        super().fit(X, y, **kwargs)