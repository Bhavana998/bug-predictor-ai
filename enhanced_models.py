"""
Enhanced Models with Deep Learning and Advanced ML
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertModel, BertTokenizer
import joblib
import os

class LSTMModel:
    """LSTM Deep Learning Model for Bug Classification"""
    
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture"""
        model = models.Sequential([
            layers.Embedding(self.max_words, 128, input_length=self.max_len),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            layers.LSTM(32, dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, texts):
        """Convert texts to sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=10):
        """Train the LSTM model"""
        # Fit tokenizer
        self.tokenizer.fit_on_texts(X_train)
        
        # Prepare sequences
        X_train_seq = self.prepare_sequences(X_train)
        X_val_seq = self.prepare_sequences(X_val)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Train
        self.history = self.model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return self.history
    
    def predict(self, texts):
        """Make predictions"""
        sequences = self.prepare_sequences(texts)
        return self.model.predict(sequences)
    
    def save(self, path):
        """Save model and tokenizer"""
        self.model.save(f"{path}_model.h5")
        joblib.dump(self.tokenizer, f"{path}_tokenizer.pkl")
    
    def load(self, path):
        """Load model and tokenizer"""
        self.model = tf.keras.models.load_model(f"{path}_model.h5")
        self.tokenizer = joblib.load(f"{path}_tokenizer.pkl")

class BERTModel:
    """BERT Transformer Model for Bug Classification"""
    
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = TFBertModel.from_pretrained(self.model_name)
        self.classifier = None
        self.history = None
        
    def build_model(self):
        """Build BERT-based classifier"""
        input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
        
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[1]
        
        x = layers.Dense(256, activation='relu')(bert_output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(2, activation='softmax')(x)
        
        model = models.Model(inputs=[input_ids, attention_mask], outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def encode_texts(self, texts, max_len=128):
        """Encode texts for BERT"""
        return self.tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

class ModelOptimizer:
    """Hyperparameter Optimization using PSO"""
    
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = {}
        self.best_score = 0
        
    def pso_optimize(self, n_particles=20, n_iterations=50):
        """Particle Swarm Optimization for hyperparameters"""
        import pyswarm
        
        def objective_function(params):
            n_estimators = int(params[0])
            max_depth = int(params[1])
            learning_rate = params[2]
            
            # Update model parameters
            self.model.set_params(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate
            )
            
            # Train and evaluate
            self.model.fit(self.X_train, self.y_train)
            score = self.model.score(self.X_val, self.y_val)
            
            return -score  # Minimize negative score
        
        # Parameter bounds
        lb = [50, 5, 0.01]   # lower bounds
        ub = [500, 50, 0.3]  # upper bounds
        
        # Run PSO
        xopt, fopt = pyswarm.pso(
            objective_function,
            lb, ub,
            swarmsize=n_particles,
            maxiter=n_iterations
        )
        
        self.best_params = {
            'n_estimators': int(xopt[0]),
            'max_depth': int(xopt[1]),
            'learning_rate': xopt[2]
        }
        self.best_score = -fopt
        
        return self.best_params