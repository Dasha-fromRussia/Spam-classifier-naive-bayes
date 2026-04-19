import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Скачивание необходимых ресурсов NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Загрузка датасета
print("КЛАССИФИКАЦИЯ СПАМ-СООБЩЕНИЙ С ИСПОЛЬЗОВАНИЕМ НАИВНОГО БАЙЕСОВСКОГО КЛАССИФИКАТОРА")
# Загрузка данных (SMS Spam Collection Dataset)
# Можно было самостоятельно скачать файл, но я сделала через ссылку в коде
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except:
    # Загрузка через pandas
    url = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    print("Данные загружены из указанного источника")

# Выбираем нужные колонки
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

print(f"\nРазмер датасета: {df.shape[0]} сообщений")
print(f"Распределение классов:\n{df['label'].value_counts()}")

# Преобразование меток в бинарные значения
df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1, 'ham': 0, 'spam': 1})

# 1. ПРЕДОБРАБОТКА ТЕКСТА
print("1. ПРЕДОБРАБОТКА ТЕКСТОВЫХ ДАННЫХ")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(text):
    """Функция очистки и предобработки текста"""
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации и цифр
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_message(message):
    """Полная предобработка сообщения"""
    # Очистка текста
    cleaned = clean_text(message)
    # Токенизация и удаление стоп-слов
    tokens = cleaned.split()
    tokens = [token for token in tokens if token not in stop_words]
    # Стемминг (процесс приведения слова к его корневой форме путем отсечения окончаний и суффиксов)
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)


# Применяем предобработку
df['cleaned_message'] = df['message'].apply(preprocess_message)

print("Пример предобработки:")
print(f"Оригинал: {df['message'].iloc[0]}")
print(f"После обработки: {df['cleaned_message'].iloc[0]}")

# Создание дополнительных признаков
df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
df['capital_letters'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['exclamation_marks'] = df['message'].apply(lambda x: x.count('!'))
df['question_marks'] = df['message'].apply(lambda x: x.count('?'))
df['url_count'] = df['message'].apply(lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', x)))

print("\nСозданы дополнительные признаки:")
print(f"- Длина сообщения (символов): мин={df['message_length'].min()}, макс={df['message_length'].max()}")
print(f"- Количество слов: мин={df['word_count'].min()}, макс={df['word_count'].max()}")
print(f"- Заглавные буквы: мин={df['capital_letters'].min()}, макс={df['capital_letters'].max()}")

# 2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
print("2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")

# Статистика по классам
print("\nОписательная статистика по классам:")
print(df.groupby('label')[['message_length', 'word_count', 'capital_letters', 'exclamation_marks']].describe())
# Подробная статистика по каждому признаку отдельно
print("\n" + "="*60)
print("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ПРИЗНАКАМ")
print("="*60)

print("\n--- ЗАГЛАВНЫЕ БУКВЫ (capital_letters) ---")
print(df.groupby('label')['capital_letters'].describe())

print("\n--- ВОСКЛИЦАТЕЛЬНЫЕ ЗНАКИ (exclamation_marks) ---")
print(df.groupby('label')['exclamation_marks'].describe())

print("\n--- ВОПРОСИТЕЛЬНЫЕ ЗНАКИ (question_marks) ---")
print(df.groupby('label')['question_marks'].describe())

print("\n--- ДЛИНА СООБЩЕНИЯ (message_length) ---")
print(df.groupby('label')['message_length'].describe())

print("\n--- КОЛИЧЕСТВО СЛОВ (word_count) ---")
print(df.groupby('label')['word_count'].describe())

# Визуализация 1: Распределение классов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График распределения классов
df['label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Распределение классов сообщений', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Класс')
axes[0, 0].set_ylabel('Количество сообщений')
axes[0, 0].tick_params(axis='x', rotation=0)

# Гистограмма длины сообщений
axes[0, 1].hist([df[df['label'] == 'ham']['message_length'], df[df['label'] == 'spam']['message_length']],
                bins=30, label=['Ham (не спам)', 'Spam'], alpha=0.7, color=['green', 'red'])
axes[0, 1].set_title('Распределение длины сообщений', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Длина сообщения (символов)')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].legend()

# Boxplot длины сообщений
df.boxplot(column='message_length', by='label', ax=axes[1, 0])
axes[1, 0].set_title('Распределение длины сообщений по классам (Boxplot)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Класс')
axes[1, 0].set_ylabel('Длина сообщения')

# Количество заглавных букв
axes[1, 1].bar(['Ham', 'Spam'], [df[df['label'] == 'ham']['capital_letters'].mean(),
                                 df[df['label'] == 'spam']['capital_letters'].mean()],
               color=['green', 'red'])
axes[1, 1].set_title('Среднее количество заглавных букв', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Среднее количество')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# WordCloud для спама и не-спама
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

spam_words = ' '.join(df[df['label'] == 'spam']['cleaned_message'])
ham_words = ' '.join(df[df['label'] == 'ham']['cleaned_message'])

wordcloud_spam = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(spam_words)
wordcloud_ham = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_words)

ax1.imshow(wordcloud_spam, interpolation='bilinear')
ax1.set_title('Облако слов: СПАМ', fontsize=16, fontweight='bold')
ax1.axis('off')

ax2.imshow(wordcloud_ham, interpolation='bilinear')
ax2.set_title('Облако слов: НЕ СПАМ', fontsize=16, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
plt.show()

# Корреляционная матрица
correlation_features = ['message_length', 'word_count', 'capital_letters', 'exclamation_marks', 'question_marks']
corr_matrix = df[correlation_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Корреляционная матрица признаков', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ПОСТРОЕНИЕ МОДЕЛИ NAIVE BAYES
print("3. ПОСТРОЕНИЕ МОДЕЛИ НАИВНОГО БАЙЕСОВСКОГО КЛАССИФИКАТОРА")

# Подготовка данных для моделирования
X = df['cleaned_message']
y = df['label_binary']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {len(X_train)} сообщений")
print(f"Размер тестовой выборки: {len(X_test)} сообщений")

# Векторизация текста (Bag of Words)
vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"\nРазмер матрицы признаков: {X_train_vectorized.shape}")

# Обучение модели Multinomial Naive Bayes
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_vectorized, y_train)

# Предсказания
y_pred = nb_model.predict(X_test_vectorized)
y_pred_proba = nb_model.predict_proba(X_test_vectorized)[:, 1]

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nРезультаты классификации:")
print(f"Точность (Accuracy): {accuracy:.4f}")
print(f"Точность (Precision): {precision:.4f}")
print(f"Полнота (Recall): {recall:.4f}")
print(f"F1-мера: {f1:.4f}")

print("\nДетальный отчет классификации:")
print(classification_report(y_test, y_pred, target_names=['Ham (не спам)', 'Spam']))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Матрица ошибок классификации', fontsize=16, fontweight='bold')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. СТАТИСТИЧЕСКАЯ ПРОВЕРКА ГИПОТЕЗ
print("4. СТАТИСТИЧЕСКАЯ ПРОВЕРКА ГИПОТЕЗ")

# Используем хи-квадрат тест для проверки значимости признаков
from sklearn.feature_selection import chi2

chi2_scores, p_values = chi2(X_train_vectorized, y_train)

# Получаем топ-10 наиболее значимых слов
feature_names = vectorizer.get_feature_names_out()
significant_features = [(feature_names[i], chi2_scores[i], p_values[i])
                        for i in range(len(feature_names)) if p_values[i] < 0.05]
significant_features.sort(key=lambda x: x[1], reverse=True)

print("\nТоп-10 наиболее значимых признаков (слов) для классификации:")
for i, (word, score, p_val) in enumerate(significant_features[:10]):
    print(f"{i + 1}. '{word}' - Хи-квадрат: {score:.2f}, p-value: {p_val:.6f}")

# Формулировка гипотез
print("\nФормулировка статистических гипотез:")
print("H0 (нулевая гипотеза): Частота встречаемости слова не связана с классом сообщения (спам/не спам)")
print("H1 (альтернативная гипотеза): Существует статистически значимая связь между частотой слова и классом сообщения")
print("\nРезультат: Для всех топ-10 слов p-value < 0.05, следовательно, H0 отвергается.")
print("Это подтверждает, что выбранные слова являются статистически значимыми предикторами спама.")

# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

# Топ-10 наиболее важных слов для спама
feature_log_prob = nb_model.feature_log_prob_
spam_word_prob = np.exp(feature_log_prob[1, :])
ham_word_prob = np.exp(feature_log_prob[0, :])
word_importance = spam_word_prob / (ham_word_prob + 1e-10)

top_indices = np.argsort(word_importance)[-10:][::-1]
top_words = [feature_names[i] for i in top_indices]
top_scores = [word_importance[i] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_words, top_scores, color='red')
plt.xlabel('Важность (отношение вероятностей)', fontsize=12)
plt.title('Топ-10 слов, наиболее характерных для СПАМА', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('important_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC-кривая
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительные срабатывания (FPR)', fontsize=12)
plt.ylabel('Истинноположительные срабатывания (TPR)', fontsize=12)
plt.title('ROC-кривая классификатора Naive Bayes', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. ТЕСТИРОВАНИЕ НА НОВЫХ ПРИМЕРАХ
print("6. ТЕСТИРОВАНИЕ НА НОВЫХ ПРИМЕРАХ")

test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
    "Hi, are we still meeting for coffee tomorrow at 3pm?",
    "URGENT: Your account has been suspended. Verify your information immediately.",
    "Don't forget to buy milk on your way home.",
    "FREE entry to the lottery! Text WIN to 12345 to claim $1000!"
]

print("\nРезультаты классификации новых сообщений:")
for msg in test_messages:
    cleaned = preprocess_message(msg)
    vectorized = vectorizer.transform([cleaned])
    prediction = nb_model.predict(vectorized)[0]
    probability = nb_model.predict_proba(vectorized)[0]

    result = "СПАМ" if prediction == 1 else "НЕ СПАМ"
    confidence = probability[1] if prediction == 1 else probability[0]

    print(f"\nСообщение: {msg[:50]}...")
    print(f"Результат: {result} (уверенность: {confidence:.2%})")

# Сохранение модели для дашборда
import joblib

joblib.dump(nb_model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nМодель и векторизатор сохранены для использования в дашборде.")

# Сохранение результатов в CSV для дашборда
df['prediction'] = nb_model.predict(vectorizer.transform(df['cleaned_message']))
df['confidence'] = nb_model.predict_proba(vectorizer.transform(df['cleaned_message']))[:, 1]
df[['message', 'cleaned_message', 'label', 'prediction', 'confidence', 'message_length', 'capital_letters']].to_csv('spam_predictions.csv', index=False)

print("Результаты сохранены в файл 'spam_predictions.csv' для создания дашборда.")
