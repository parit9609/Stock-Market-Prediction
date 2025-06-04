"""
Data processing module for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the data processor
        
        Args:
            sequence_length (int): Length of sequence for LSTM input
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load stock data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and preprocessed DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for model training
        """
        try:
            # Scale the data
            data = self.scaler.fit_transform(df[[target_column]])
            
            # Create sequences
            X = []
            y = []
            
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length])
                
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            train_ratio (float): Ratio of training data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and testing sets
        """
        try:
            train_size = int(len(X) * train_ratio)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data
        
        Args:
            data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        try:
            return self.scaler.inverse_transform(data)
        except Exception as e:
            logger.error(f"Error in inverse transform: {str(e)}")
            raise
            
    def prepare_prediction_data(self, df: pd.DataFrame, target_column: str = 'Close') -> np.ndarray:
        """
        Prepare data for making predictions
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column to predict
            
        Returns:
            np.ndarray: Prepared data for prediction
        """
        try:
            data = self.scaler.fit_transform(df[[target_column]])
            X = []
            X.append(data[-self.sequence_length:])
            return np.array(X)
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise 