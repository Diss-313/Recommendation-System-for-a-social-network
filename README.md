# Recommendation System for a social network

<br/> **Реализация рекомендательной системы для постов в социальной сети. Система возвращает персонализированные рекомендации постов для каждого пользователя на основе машинного обучения и нейросетевых эмбеддингов.**
<br/>
<br/> По ID пользователя и текущей дате сервис выдаёт 5 наиболее релевантных публикаций для данного пользователя.
<br/>
> &ensp; **Исходные данные:**
<br/> &ensp; Базовые сырые данные - таблицы PostgreSQL
<br/>&ensp; 1) **Users** - данные пользователей
<br/>&ensp; 2) **Posts** - данные постов
<br/>&ensp; 3) **Feeds** - данные активности пользователей
<br/>&ensp; - *Набор пользователей фиксирован*
<br/>&ensp; - *Модели не переобучаются при работе сервиса*
<br/>

## Структура проекта

```
├── model_exp_neural.py          # Скрипт обучения модели с нейросетевыми эмбеддингами
├── service_fastapi_nl.py        # FastAPI сервис для рекомендаций
├── schema.py                    # Pydantic схемы данных
├── requirements.txt             # Зависимости проекта
├── .gitignore                   # Игнорируемые файлы
└── README.md                    # Документация
```

## Особенности реализации

### Нейросетевые эмбеддинги
- Использование модели `sentence-transformers/all-MiniLM-L6-v2` для генерации эмбеддингов текстов
- Mean Pooling для получения векторного представления
- PCA для снижения размерности
- K-means кластеризация в пространстве эмбеддингов

### Признаки модели
- **Текстовые**: эмбеддинги, кластеры, расстояния до центроидов, PCA компоненты
- **Пользовательские**: пол, возраст, страна, город, группа эксперимента, ОС, источник
- **Временные**: час, месяц

## Основные библиотеки

![PyTorch](https://img.shields.io/badge/PYTORCH-1.11.0-090909?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/TRANSFORMERS-4.20.0-090909?style=flat-square&logo=huggingface)
![CatBoost](https://img.shields.io/badge/CATBOOST-1.0.6-090909?style=flat-square&logo=catboost)
![Scikit--learn](https://img.shields.io/badge/SCIKITLEARN-1.1.1-090909?style=flat-square&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/PANDAS-1.4.2-090909?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NUMPY-1.22.4-090909?style=flat-square&logo=numpy)
![FastAPI](https://img.shields.io/badge/FASTAPI-0.75.1-090909?style=flat-square&logo=fastapi)
![Uvicorn](https://img.shields.io/badge/UVICORN-0.16.0-090909?style=flat-square&logo=uvicorn)
![Pydantic](https://img.shields.io/badge/PYDANTIC-1.9.1-090909?style=flat-square&logo=pydantic)
![SQLAlchemy](https://img.shields.io/badge/SQLALCHEMY-1.4.35-090909?style=flat-square&logo=sqlalchemy)

## Метрики качества

Модель обучена на метрике **HitRate@5** (метрика принимает значение = 1, если пользователь лайкнул 1 или более постов из 5 рекомендованных, 0 если ни один из 5 не был лайкнут)

## Итоговые показатели

HitRate@5 > 0.65; скорость работы сервиса на FastAPI – менее 0.5 сек; стабильная работа сервиса и оптимизация памяти

## Запуск сервиса

### 1. Клонирование проекта

```bash
git clone https://github.com/Diss-313/Recommendation-System-for-a-social-network
cd Recommendation-System-for-a-social-network
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

Создайте файл `.env`, где будет переменная `DATABASE_URL` с данными подключения к PostgreSQL.

### 4. Обучение модели (опционально)

Если нужно переобучить модель:

```bash
python model_exp_neural.py
```

### 5. Запуск сервиса

```bash
uvicorn service_fastapi_nl:app --host 0.0.0.0 --port 8000
```

### 6. Тестирование

Откройте в браузере или используйте Postman:

```
http://localhost:8000/post/recommendations/?id=200&time=2023-11-12%2022:57:45&limit=5
```

Параметры:
- `id` - ID пользователя (199 < id < 163206)
- `time` - временная метка для рекомендаций
- `limit` - количество рекомендаций (по умолчанию 10)

## API Документация

После запуска сервиса документация доступна по адресам:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`