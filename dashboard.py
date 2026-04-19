import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Загрузка данных
df = pd.read_csv('spam_predictions.csv')

# Подготовка данных
df['label_ru'] = df['label'].map({'ham': 'Не спам', 'spam': 'Спам'})
df['prediction_ru'] = df['prediction'].map({0: 'Не спам', 1: 'Спам'})
df['correct'] = df['label'] == df['prediction'].map({0: 'ham', 1: 'spam'})
df['confidence_percent'] = df['confidence'] * 100

# Расчет метрик
total_messages = len(df)
spam_count = len(df[df['label'] == 'spam'])
spam_share = spam_count / total_messages
accuracy = df['correct'].mean()

# Создание приложения Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Классификация спам-сообщений - Naive Bayes",
            className="text-center mt-4 mb-4",
            style={'color': '#2c3e50'}),

    # Ряд карточек с метриками
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Точность модели", className="card-title text-center"),
                html.H2(f"{accuracy:.2%}", className="text-center text-success")
            ])
        ], color="light"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Всего сообщений", className="card-title text-center"),
                html.H2(f"{total_messages}", className="text-center text-primary")
            ])
        ], color="light"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Спам-сообщений", className="card-title text-center"),
                html.H2(f"{spam_count}", className="text-center text-danger")
            ])
        ], color="light"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Доля спама", className="card-title text-center"),
                html.H2(f"{spam_share:.2%}", className="text-center text-warning")
            ])
        ], color="light"), width=3),
    ], className="mb-4"),

    # Фильтры
    dbc.Row([
        dbc.Col([
            html.Label("Фильтр по классу:"),
            dcc.Dropdown(
                id='class-filter',
                options=[
                    {'label': 'Все', 'value': 'all'},
                    {'label': 'Спам', 'value': 'spam'},
                    {'label': 'Не спам', 'value': 'ham'}
                ],
                value='all',
                clearable=False
            )
        ], width=4),

        dbc.Col([
            html.Label("Минимальная уверенность модели:"),
            dcc.Slider(
                id='confidence-slider',
                min=0, max=100, step=5,
                value=0,
                marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'}
            )
        ], width=8),
    ], className="mb-4"),

    # Графики в два столбца
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='length-hist')
        ], width=6),

        dbc.Col([
            dcc.Graph(id='capital-hist')
        ], width=6),
    ], className="mb-4"),

    # Таблица с сообщениями
    dbc.Row([
        dbc.Col([
            html.H4("Сообщения", className="mt-2"),
            html.P("Нажмите на заголовок столбца для сортировки, используйте поиск для фильтрации"),
            dcc.Input(
                id='search-input',
                type='text',
                placeholder='🔍 Поиск по тексту сообщения...',
                className='form-control mb-2'
            ),
            html.Div(id='messages-table')
        ], width=12),
    ], className="mb-4"),

    # Облака слов (через альтернативную визуализацию - гистограммы частотности)
    dbc.Row([
        dbc.Col([
            html.H4("Топ-15 слов в СПАМ-сообщениях", className="text-center text-danger"),
            dcc.Graph(id='spam-wordcloud')
        ], width=6),

        dbc.Col([
            html.H4("Топ-15 слов в НЕ СПАМ-сообщениях", className="text-center text-success"),
            dcc.Graph(id='ham-wordcloud')
        ], width=6),
    ]),
], fluid=True)


@app.callback(
    [Output('length-hist', 'figure'),
     Output('capital-hist', 'figure'),
     Output('spam-wordcloud', 'figure'),
     Output('ham-wordcloud', 'figure'),
     Output('messages-table', 'children')],
    [Input('class-filter', 'value'),
     Input('confidence-slider', 'value'),
     Input('search-input', 'value')]
)
def update_dashboard(class_filter, confidence_threshold, search_text):
    # Фильтрация данных
    df_filtered = df[df['confidence_percent'] >= confidence_threshold].copy()

    if class_filter == 'spam':
        df_filtered = df_filtered[df_filtered['label'] == 'spam']
    elif class_filter == 'ham':
        df_filtered = df_filtered[df_filtered['label'] == 'ham']

    if search_text:
        df_filtered = df_filtered[df_filtered['message'].str.contains(search_text, case=False, na=False)]

    # 1. Гистограмма длины сообщений
    fig_length = px.histogram(
        df_filtered, x='message_length', color='label_ru',
        title='Распределение длины сообщений',
        labels={'message_length': 'Длина сообщения (символов)', 'count': 'Количество сообщений'},
        barmode='overlay', nbins=30,
        color_discrete_map={'Спам': '#ef553b', 'Не спам': '#2ca02c'}
    )
    fig_length.update_layout(legend_title_text='Класс', template='plotly_white')

    # 2. Гистограмма заглавных букв
    fig_capital = px.histogram(
        df_filtered, x='capital_letters', color='label_ru',
        title='Распределение заглавных букв',
        labels={'capital_letters': 'Количество заглавных букв', 'count': 'Количество сообщений'},
        barmode='overlay', nbins=30,
        color_discrete_map={'Спам': '#ef553b', 'Не спам': '#2ca02c'}
    )
    fig_capital.update_layout(legend_title_text='Класс', template='plotly_white')

    # 3. Топ-15 слов в спаме (вместо облака слов)
    from collections import Counter
    import re

    def get_top_words(texts, n=15):
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
            all_words.extend(words)
        return Counter(all_words).most_common(n)

    spam_texts = df[df['label'] == 'spam']['cleaned_message'].tolist() if 'cleaned_message' in df.columns else \
    df[df['label'] == 'spam']['message'].tolist()
    ham_texts = df[df['label'] == 'ham']['cleaned_message'].tolist() if 'cleaned_message' in df.columns else \
    df[df['label'] == 'ham']['message'].tolist()

    top_spam_words = get_top_words(spam_texts, 15)
    top_ham_words = get_top_words(ham_texts, 15)

    fig_spam = px.bar(
        x=[w[1] for w in top_spam_words[::-1]],
        y=[w[0] for w in top_spam_words[::-1]],
        orientation='h',
        title='Самые частотные слова в СПАМЕ',
        labels={'x': 'Частота встречаемости', 'y': 'Слово'},
        color_discrete_sequence=['#ef553b']
    )
    fig_spam.update_layout(template='plotly_white', showlegend=False)

    fig_ham = px.bar(
        x=[w[1] for w in top_ham_words[::-1]],
        y=[w[0] for w in top_ham_words[::-1]],
        orientation='h',
        title='Самые частотные слова в НЕ СПАМЕ',
        labels={'x': 'Частота встречаемости', 'y': 'Слово'},
        color_discrete_sequence=['#2ca02c']
    )
    fig_ham.update_layout(template='plotly_white', showlegend=False)

    # 4. Таблица сообщений
    table_df = df_filtered[['message', 'label_ru', 'prediction_ru', 'confidence_percent', 'correct']].copy()
    table_df.columns = ['Сообщение', 'Истинный класс', 'Предсказанный класс', 'Уверенность (%)', 'Верно?']
    table_df['Уверенность (%)'] = table_df['Уверенность (%)'].round(1)
    table_df['Верно?'] = table_df['Верно?'].map({True: '✅', False: '❌'})

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in table_df.columns])),
            html.Tbody([
                html.Tr([html.Td(row[col]) for col in table_df.columns])
                for _, row in table_df.head(50).iterrows()
            ])
        ],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="mt-2"
    )

    return fig_length, fig_capital, fig_spam, fig_ham, table


if __name__ == '__main__':
    print("ЗАПУСК ИНТЕРАКТИВНОГО ДАШБОРДА")
    print("\nДашборд будет доступен по адресу: http://127.0.0.1:8050")
    print("Нажмите Ctrl+C в терминале для остановки\n")
    app.run(debug=True)