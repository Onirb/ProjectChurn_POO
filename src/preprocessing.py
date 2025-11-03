

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    """
    Handles all preprocessing tasks:
    - Cleans columns and fills missing values
    - Encodes categorical variables
    - Applies log transformation to skewed features
    - Performs scaling (Min-Max) on usage metrics
    - Splits dataset into train and test, preventing data leakage
    """

    def __init__(self, target_col='churn', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.numeric_cols = None
        self.categorical_cols = None

    def encode_categorical(self, df):
        """Encodes categorical columns (Yes/No -> 1/0 and get_dummies)."""
        df = df.copy()

        # Normalize capitalization and encode binary values
        if self.target_col in df.columns:
            df[self.target_col] = (
                df[self.target_col]
                .astype(str)
                .str.lower()
                .replace({'yes': 1, 'no': 0})
            )

        df = df.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

        # Convert object columns to dummy variables
        self.categorical_cols = df.select_dtypes(include='object').columns
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        return df

    def preprocess(self, df):
        """
        Full preprocessing pipeline with data leakage prevention.
        Applies:
        - column cleaning
        - missing value imputation (median)
        - log transformation (numbervmailmessages)
        - Min-Max scaling on usage-related columns
        """

        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # Fill missing numeric values with median
        df = df.fillna(df.median(numeric_only=True))

        # Apply log transform to 'numbervmailmessages' (if it exists)
        if 'numbervmailmessages' in df.columns:
            df['numbervmailmessages'] = np.log1p(df['numbervmailmessages'])

        # Encode categorical variables
        df = self.encode_categorical(df)

        # Split before scaling to prevent data leakage
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Define usage-related columns for scaling
        usage_columns = [
            'totaldayminutes', 'totaldaycalls', 'totaldaycharge',
            'totaleveminutes', 'totalevecalls', 'totalevecharge',
            'totalnightminutes', 'totalnightcalls', 'totalnightcharge',
            'totalintlminutes', 'totalintlcalls', 'totalintlcharge',
            'accountlength'
        ]
        cols_to_scale = [c for c in usage_columns if c in X_train.columns]

        # Apply Min-Max scaling only to usage columns
        X_train[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])

        return X_train, X_test, y_train, y_test
