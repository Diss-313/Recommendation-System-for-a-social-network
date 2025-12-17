import os
import pandas as pd
import re
import string
import nltk
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

# Загрузка переменных окружения
load_dotenv()

# Создание подключения к БД
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL не найден.")

engine = create_engine(DATABASE_URL)

# ==================== Загрузка данных ====================
print("Загрузка данных...")
posts_info = pd.read_sql('SELECT * FROM post_text_df', con=engine)

# ==================== Предобработка текста ====================
nltk.download("wordnet")
wnl = nltk.stem.WordNetLemmatizer()

def preprocess_text(text, lemmatizer=wnl):
    """
    Предобработка текста: приведение к нижнему регистру, удаление пунктуации,
    нормализация пробелов и лемматизация.
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [lemmatizer.lemmatize(token) for token in text.split()]
    return ' '.join(tokens)

# ==================== Нейросетевые эмбеддинги ====================
print("Генерация нейросетевых эмбеддингов...")

class TextEmbedder:
    """
    Класс для генерации эмбеддингов текстов с использованием предобученных моделей
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Инициализация модели и токенизатора
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Переводим в режим inference
        
    def get_embeddings(self, texts, batch_size=32, max_length=256):
        """
        Генерация эмбеддингов для списка текстов
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Токенизация
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(self.device)
            
            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем mean pooling для получения векторного представления
                batch_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - берем среднее значение hidden states с учетом attention mask
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# Инициализация эмбеддера
embedder = TextEmbedder()

# Предобработка текстов
print("Предобработка текстов...")
processed_texts = posts_info['text'].fillna('').apply(preprocess_text).tolist()

# Генерация эмбеддингов
print("Генерация эмбеддингов...")
text_embeddings = embedder.get_embeddings(processed_texts)

# Создание DataFrame с эмбеддингами
embedding_columns = [f'embedding_{i}' for i in range(text_embeddings.shape[1])]
embeddings_df = pd.DataFrame(
    text_embeddings,
    index=posts_info.post_id,
    columns=embedding_columns
)

# ==================== Статистики и фичи из эмбеддингов ====================
print("Генерация признаков из эмбеддингов...")

# Базовые статистики по эмбеддингам
posts_info['EmbeddingNorm'] = np.linalg.norm(text_embeddings, axis=1)  # Норма вектора
posts_info['EmbeddingMean'] = text_embeddings.mean(axis=1)  # Среднее значение
posts_info['EmbeddingStd'] = text_embeddings.std(axis=1)   # Стандартное отклонение

# ==================== Кластеризация в пространстве эмбеддингов ====================
print("Кластеризация в пространстве эмбеддингов...")

# PCA для уменьшения размерности
pca = PCA(n_components=20)
pca_features = pca.fit_transform(text_embeddings)

# K-means кластеризация
N_CLUSTERS = 15
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(pca_features)

posts_info['TextCluster'] = cluster_labels

# Добавление расстояний до центроидов кластеров
distance_columns = [f'DistanceToCluster_{i}' for i in range(1, N_CLUSTERS + 1)]
distances_df = pd.DataFrame(
    data=kmeans.transform(pca_features),
    columns=distance_columns,
    index=posts_info.index
)

# Добавляем PCA компоненты как отдельные фичи
pca_columns = [f'PC_{i+1}' for i in range(pca_features.shape[1])]
pca_df = pd.DataFrame(
    pca_features,
    columns=pca_columns,
    index=posts_info.index
)

posts_info = pd.concat([posts_info, distances_df, pca_df], axis=1)

# Сохранение признаков в БД
print("Сохранение признаков в БД...")
posts_info.to_sql("k_m_exp_neural_38_post_features_lesson_22", con=engine)

# ==================== Подготовка данных для модели ====================
print("Подготовка данных для модели...")

# Загрузка данных о взаимодействиях
feed_query = """
    SELECT
        EXTRACT(HOUR FROM timestamp)::INTEGER as hour,
        EXTRACT(MONTH FROM timestamp)::INTEGER as month,
        timestamp,
        feed_data.user_id,
        post_id,
        gender,
        age,
        country,
        city,
        exp_group,
        os,
        source,
        target
    FROM feed_data
    JOIN user_data ON feed_data.user_id = user_data.user_id
    WHERE action = 'view'
    LIMIT 5000000
"""

feed_data = pd.read_sql(feed_query, con=engine)

# Объединение признаков постов с данными о взаимодействиях
model_data = pd.merge(
    feed_data,
    posts_info.drop(columns=['text']),
    on='post_id',
    how='left'
)

# Разделение на train/test
train_data = model_data[model_data['timestamp'] < '2021-12-15'].drop('timestamp', axis=1)
test_data = model_data[model_data['timestamp'] >= '2021-12-15'].drop('timestamp', axis=1)

# Подготовка фич и целевой переменной
feature_columns = [
    'topic', 'EmbeddingNorm', 'EmbeddingMean', 'EmbeddingStd', 'TextCluster',
    *distance_columns,
    *pca_columns,
    'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source', 'hour', 'month'
]

X_train = train_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['target']
y_test = test_data['target']

# Категориальные признаки для CatBoost
categorical_features = [
    'topic', 'TextCluster', 'gender', 'country', 
    'city', 'exp_group', 'hour', 'month', 'os', 'source'
]

# ==================== Обучение модели ====================
print("Обучение модели...")

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    auto_class_weights='Balanced',
    task_type='GPU',
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features
)

# ==================== Оценка модели ====================
train_predictions = model.predict_proba(X_train)[:, 1]
test_predictions = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC на трейне: {roc_auc_score(y_train, train_predictions):.4f}")
print(f"ROC-AUC на тесте: {roc_auc_score(y_test, test_predictions):.4f}")

# ==================== Визуализация важности признаков ====================
def plot_feature_importance(importance, names, model_type):
    """Визуализация важности признаков модели"""
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    importance_df = pd.DataFrame({
        'feature_names': feature_names,
        'feature_importance': feature_importance
    }).sort_values('feature_importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance_df['feature_importance'], y=importance_df['feature_names'])
    plt.title(f'{model_type} Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.tight_layout()
    plt.show()

plot_feature_importance(model.feature_importances_, X_train.columns, 'CatBoost')

# ==================== Сохранение модели ====================
model.save_model('recommender_model_v_exp_neural.cbm')
print("Модель сохранена как 'recommender_model_v_exp_neural.cbm'")