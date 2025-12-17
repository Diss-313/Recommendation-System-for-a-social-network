import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
from schema import PostGet

# Загрузка переменных окружения
load_dotenv()

# Получение URL базы данных
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL не найден.")

# ==================== Инициализация приложения ====================
app = FastAPI()

def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Загрузка больших объемов данных из PostgreSQL с потоковой обработкой
    """
    engine = create_engine(DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
    
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str) -> str:
    """
    Определение пути к модели в зависимости от окружения
    """
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    else:
        return path


def load_models() -> CatBoostClassifier:
    """Загрузка обученной модели CatBoost из файла"""
    model_path = get_model_path("recommender_model_v_exp.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_features() -> List[pd.DataFrame]:
    """
    Загрузка всех необходимых данных для рекомендаций
    """
    # Загрузка информации о лайкнутых постах
    liked_posts_query = """
        SELECT DISTINCT post_id, user_id
        FROM feed_data
        WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query)

    # Загрузка признаков постов (теперь с нейросетевыми эмбеддингами)
    posts_features = pd.read_sql(
        "SELECT * FROM k_m_exp_neural_38_post_features_lesson_22",
        con=DATABASE_URL
    )

    # Загрузка данных пользователей
    user_features = pd.read_sql(
        "SELECT * FROM user_data",
        con=DATABASE_URL
    )

    return [liked_posts, posts_features, user_features]


# ==================== Загрузка моделей и данных при старте ====================
print("Загрузка модели...")
model = load_models()

print("Загрузка признаков...")
features = load_features()
liked_posts, posts_features, user_features = features

# Выведем доступные колонки для отладки
print("Доступные колонки в posts_features:", posts_features.columns.tolist())

# Определяем ожидаемые колонки
EXPECTED_FEATURES = [
    'topic', 'EmbeddingNorm', 'EmbeddingMean', 'EmbeddingStd', 'TextCluster',
    'DistanceToCluster_1', 'DistanceToCluster_2', 'DistanceToCluster_3',
    'DistanceToCluster_4', 'DistanceToCluster_5', 'DistanceToCluster_6', 
    'DistanceToCluster_7', 'DistanceToCluster_8', 'DistanceToCluster_9',
    'DistanceToCluster_10', 'DistanceToCluster_11', 'DistanceToCluster_12',
    'DistanceToCluster_13', 'DistanceToCluster_14', 'DistanceToCluster_15',
    'PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6', 'PC_7', 'PC_8', 'PC_9', 'PC_10',
    'PC_11', 'PC_12', 'PC_13', 'PC_14', 'PC_15', 'PC_16', 'PC_17', 'PC_18', 'PC_19', 'PC_20'
]

# Категориальные признаки
CATEGORICAL_FEATURES = [
    'topic', 'TextCluster', 'gender', 'country', 
    'city', 'exp_group', 'hour', 'month', 'os', 'source'
]


def get_recommended_feed(id: int, time: datetime, limit: int) -> List[PostGet]:
    """
    Основная функция для генерации рекомендаций
    """
    # Получение признаков конкретного пользователя
    user_data = user_features.loc[user_features.user_id == id]
    if user_data.empty:
        print(f"Пользователь {id} не найден")
        return []
    
    user_data = user_data.drop('user_id', axis=1)

    # Проверяем, есть ли колонка post_id в posts_features
    # Если нет, ищем альтернативные названия
    post_id_column = None
    possible_post_id_names = ['post_id', 'id', 'post_id', 'postid']
    
    for col_name in possible_post_id_names:
        if col_name in posts_features.columns:
            post_id_column = col_name
            break
    
    if post_id_column is None:
        print("Ошибка: не найдена колонка с идентификатором поста")
        print("Доступные колонки:", posts_features.columns.tolist())
        return []
    
    print(f"Используем колонку '{post_id_column}' как идентификатор поста")

    # Подготовка признаков постов
    available_features = [col for col in EXPECTED_FEATURES if col in posts_features.columns]
    
    # Создаем копию данных постов с нужными признаками
    posts_data = posts_features[available_features].copy()
    
    # Добавляем колонку с идентификатором поста в данные
    posts_data[post_id_column] = posts_features[post_id_column]
    
    # Данные для ответа (текст и тема поста)
    # Ищем колонки с текстом и темой
    text_column = None
    topic_column = None
    
    for col in posts_features.columns:
        if col.lower() == 'text':
            text_column = col
        elif col.lower() == 'topic':
            topic_column = col
    
    if text_column is None or topic_column is None:
        print("Ошибка: не найдены колонки с текстом или темой поста.")
        return []
    
    post_content = posts_features[[post_id_column, text_column, topic_column]].copy()

    # Создание объединенного датафрейма пользователь + посты
    user_features_dict = dict(zip(user_data.columns, user_data.values[0]))
    user_posts_data = posts_data.assign(**user_features_dict)
    user_posts_data = user_posts_data.set_index(post_id_column)

    # Добавление временных признаков
    user_posts_data['hour'] = time.hour
    user_posts_data['month'] = time.month

    # Предсказание вероятностей лайка для каждого поста
    try:
        predictions = model.predict_proba(user_posts_data)[:, 1]
        user_posts_data['predicts'] = predictions
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return []

    # Фильтрация уже лайкнутых постов
    user_liked_posts = liked_posts[liked_posts.user_id == id][post_id_column].values
    filtered_posts = user_posts_data[~user_posts_data.index.isin(user_liked_posts)]

    # Выбор топ-N постов с наибольшей вероятностью
    recommended_post_ids = filtered_posts.sort_values('predicts')[-limit:].index

    # Формирование ответа
    recommendations = []
    for post_id in recommended_post_ids:
        post_info = post_content[post_content[post_id_column] == post_id]
        if not post_info.empty:
            recommendations.append(
                PostGet(**{
                    "id": int(post_id),
                    "text": post_info[text_column].values[0],
                    "topic": post_info[topic_column].values[0]
                })
            )
    
    print(f"Сгенерировано {len(recommendations)} рекомендаций для пользователя {id}")
    return recommendations


# ==================== API endpoint ====================
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int, 
    time: datetime, 
    limit: int = 10
) -> List[PostGet]:
    """
    API endpoint для получения рекомендаций постов
    """
    
    return get_recommended_feed(id, time, limit)