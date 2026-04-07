import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
import os
import datetime

# --- DATABASE CONNECTION ---
DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")
engine = create_engine(DB_PATH)

def fetch_data():
    """
    Автоматически подключается к базе данных и обновляет данные.
    """
    try:
        # Чтение таблицы из PostgreSQL
        df = pd.read_sql("SELECT * FROM track_points", engine)
        
        if df.empty:
            return df
            
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Инженерия признаков: Сезон
        def get_season(month):
            if pd.isna(month): return 'Неизвестно'
            if month in [12, 1, 2]: return 'Зима'
            if month in [3, 4, 5]: return 'Весна'
            if month in [6, 7, 8]: return 'Лето'
            if month in [9, 10, 11]: return 'Осень'
            return 'Неизвестно'
            
        df['month'] = df['datetime'].dt.month
        df['Season'] = df['month'].apply(get_season)
        
        # Инженерия признаков: Время суток
        def get_time_of_day(hour):
            if pd.isna(hour): return 'Неизвестно'
            if 5 <= hour < 12: return 'Утро'
            if 12 <= hour < 17: return 'День'
            if 17 <= hour < 22: return 'Вечер'
            return 'Ночь'
            
        df['hour'] = df['datetime'].dt.hour
        df['TimeOfDay'] = df['hour'].apply(get_time_of_day)
        
        return df
    except Exception as e:
        print(f"Ошибка загрузки данных из БД: {e}")
        return pd.DataFrame()

# Инициализация приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Модуль Б: Аналитика"

# --- ЭКРАН АВТОРИЗАЦИИ ---
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Вход в Систему Аналитики", className="text-center mt-5"),
            html.P("Демо: admin/admin или user/user", className="text-center text-muted"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Input(id="login-username", placeholder="Логин", type="text", className="mb-3"),
                    dbc.Input(id="login-password", placeholder="Пароль", type="password", className="mb-3"),
                    dbc.Button("Войти", id="login-button", color="primary", className="w-100"),
                    html.Div(id="login-alert", className="mt-2 text-danger")
                ])
            ], className="mt-4 shadow-sm"),
            html.Button(id="logout-button", style={"display": "none"})
        ], width=12, md=6, lg=4)
    ], className="justify-content-center")
], fluid=True, style={'height': '100vh', 'backgroundColor': '#f4f6f9', 'paddingTop': '10%'})

# --- ОСНОВНОЙ ДАШБОРД ---
def dashboard_layout(role):
    # Загружаем данные для инициализации фильтров
    df = fetch_data()
    
    # 2.1 Настраиваем уровни доступа (admin видит всё, user видит меньше регионов)
    all_regions = df['region'].unique().tolist() if not df.empty and 'region' in df else []
    if role == 'user' and len(all_regions) > 0:
        allowed_regions = all_regions[:max(1, len(all_regions)//2)]
    else:
        allowed_regions = all_regions
        
    seasons = df['Season'].unique().tolist() if not df.empty else []
    times_of_day = df['TimeOfDay'].unique().tolist() if not df.empty else []
    
    menu_panel = html.Div([
        html.H4("Фильтры", className="mt-3 text-white"),
        html.Hr(style={"borderColor": "white"}),
        html.Label("Регион:", className="text-white"),
        dcc.Dropdown(
            id='filter-region',
            options=[{'label': r, 'value': r} for r in allowed_regions],
            multi=True,
            value=allowed_regions,
            className="mb-3 text-dark"
        ),
        html.Label("Сезон:", className="text-white"),
        dcc.Dropdown(
            id='filter-season',
            options=[{'label': s, 'value': s} for s in seasons],
            multi=True,
            value=seasons,
            className="mb-3 text-dark"
        ),
        html.Label("Время суток:", className="text-white"),
        dcc.Dropdown(
            id='filter-time',
            options=[{'label': t, 'value': t} for t in times_of_day],
            multi=True,
            value=times_of_day,
            className="mb-3 text-dark"
        ),
        html.Br(),
        html.P(f"Уровень доступа: {role.upper()}", className="text-warning font-weight-bold"),
        dbc.Button("Выйти", id="logout-button", color="light", className="mt-5 w-100 text-dark")
    ], style={"padding": "20px", "backgroundColor": "#2C3E50", "height": "100vh"})
    
    dashboard_content = html.Div([
        dcc.Interval(id='data-refresh-interval', interval=60*1000, n_intervals=0), # Регулярное обновление (раз в минуту)
        
        dbc.Row([
            dbc.Col(html.H2(f"Анализ туристических маршрутов ({role})"), className="mt-4 mb-4")
        ]),
        
        # Ключевые метрики
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Средняя частота шагов", className="text-muted"), 
                html.H2(id="kpi-steps", className="text-primary")
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Средняя температура", className="text-muted"), 
                html.H2(id="kpi-temp", className="text-danger")
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Записей маршрутов", className="text-muted"), 
                html.H2(id="kpi-count", className="text-success")
            ]), className="shadow-sm"), md=4)
        ], className="mb-4"),
        
        # Интерактивная карта с фильтрацией (задание 2.2)
        dbc.Row([
             dbc.Col(dbc.Card([
                 dbc.CardHeader("Интерактивная карта активности туристов"),
                 dbc.CardBody(dcc.Graph(id="map-routes"))
             ], className="shadow-sm"), md=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Тренды активности (Частота шагов по сезонам)"),
                dbc.CardBody(dcc.Graph(id="chart-season-steps"))
            ], className="shadow-sm"), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Распределение активности по регионам"),
                dbc.CardBody(dcc.Graph(id="chart-region-activity"))
            ], className="shadow-sm"), md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Зависимость температуры от времени суток"),
                dbc.CardBody(dcc.Graph(id="chart-temp-time"))
            ], className="shadow-sm"), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Влияние типа местности на активность"),
                dbc.CardBody(dcc.Graph(id="chart-env-activity"))
            ], className="shadow-sm"), md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Взаимосвязь высоты маршрута с частотой шагов"),
                dbc.CardBody(dcc.Graph(id="chart-elev-steps"))
            ], className="shadow-sm"), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Популярные маршруты (Топ треков)"),
                dbc.CardBody(dcc.Graph(id="chart-popular-routes"))
            ], className="shadow-sm"), md=6),
        ], className="mb-5")
    ], style={"padding": "20px", "height": "100vh", "overflowY": "scroll", "backgroundColor": "#F4F6F9"})
    
    return dbc.Row([
        dbc.Col(menu_panel, width=2, style={'padding':'0', 'position': 'fixed', 'zIndex': 100}),
        dbc.Col(dashboard_content, width={"size": 10, "offset": 2}, style={'padding':'0'})
    ], className="m-0")

app.layout = html.Div([
    dcc.Store(id='session-role', storage_type='session'),
    html.Div(id='page-content')
])

# --- ЛОГИКА ПЕРЕКЛЮЧЕНИЯ СТРАНИЦ ---
@app.callback(
    Output('page-content', 'children'),
    Input('session-role', 'data')
)
def render_page(role):
    if role: return dashboard_layout(role)
    return login_layout


# --- ЛОГИКА АВТОРИЗАЦИИ ---
@app.callback(
    [Output('session-role', 'data', allow_duplicate=True), Output('login-alert', 'children')],
    Input('login-button', 'n_clicks'),
    [State('login-username', 'value'), State('login-password', 'value')],
    prevent_initial_call=True
)
def login(n_clicks, username, password):
    if not username or not password:
        return dash.no_update, "Введите логин и пароль"
    if username == "admin" and password == "admin":
        return "admin", ""
    if username == "user" and password == "user":
        return "user", ""
    return dash.no_update, "Неверные учетные данные. Используйте admin/admin"

@app.callback(
    Output('session-role', 'data', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    return None

# --- ЛОГИКА ОБНОВЛЕНИЯ ДАШБОРДА (ЗАДАЧА 2.2) ---
@app.callback(
    [
        Output("kpi-steps", "children"),
        Output("kpi-temp", "children"),
        Output("kpi-count", "children"),
        Output("map-routes", "figure"),
        Output("chart-season-steps", "figure"),
        Output("chart-region-activity", "figure"),
        Output("chart-temp-time", "figure"),
        Output("chart-env-activity", "figure"),
        Output("chart-elev-steps", "figure"),
        Output("chart-popular-routes", "figure")
    ],
    [
        Input('filter-region', 'value'),
        Input('filter-season', 'value'),
        Input('filter-time', 'value'),
        Input('data-refresh-interval', 'n_intervals') # Авто-периодичность
    ]
)
def update_dashboard(regions, seasons, times, n_intervals):
    df = fetch_data()
    empty_fig = px.scatter(title="Нет данных")
    
    if df.empty:
        return "0", "0", "0", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
    # Применение фильтров, заданных пользователем
    if regions: df = df[df['region'].isin(regions)]
    if seasons: df = df[df['Season'].isin(seasons)]
    if times: df = df[df['TimeOfDay'].isin(times)]
    
    if df.empty:
        return "0", "0", "0", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Основные KPI
    kpi_steps = f"{df['step_frequency'].mean():.2f}" if 'step_frequency' in df.columns else "0"
    kpi_temp = f"{df['temp'].mean():.1f} °C" if 'temp' in df.columns else "0 °C"
    kpi_count = f"{len(df)}"
    
    color_discrete_sequence = px.colors.qualitative.Pastel
    
    # 1. Интерактивная карта
    map_fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude", color="step_frequency", size="step_frequency",
        hover_name="track_id", zoom=5, height=400, 
        color_continuous_scale=px.colors.sequential.Plasma
    )
    map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    
    # 2. Средняя частота шагов по сезонам (Trend analysis)
    season_steps = df.groupby('Season')['step_frequency'].mean().reset_index()
    chart1 = px.bar(season_steps, x='Season', y='step_frequency', color='Season', color_discrete_sequence=color_discrete_sequence)
    chart1.update_layout(margin={"r":10,"t":10,"l":10,"b":10}, showlegend=False)
    
    # 3. Распределение активности по регионам
    region_activity = df['region'].value_counts().reset_index()
    region_activity.columns = ['region', 'count']
    chart2 = px.pie(region_activity, values='count', names='region', color_discrete_sequence=px.colors.qualitative.Set3)
    chart2.update_layout(margin={"r":10,"t":10,"l":10,"b":10})
    
    # 4. Зависимость температуры от времени суток (Box plot for better distribution visual)
    chart3 = px.box(df, x="TimeOfDay", y="temp", color="TimeOfDay", color_discrete_sequence=color_discrete_sequence)
    chart3.update_layout(margin={"r":10,"t":10,"l":10,"b":10}, showlegend=False)
    
    # 5. Влияние типа местности на активность (Box plot)
    if 'environment_type' in df.columns and not df['environment_type'].isnull().all():
        chart4 = px.box(df, x="environment_type", y="step_frequency", color="environment_type")
        chart4.update_layout(margin={"r":10,"t":10,"l":10,"b":10}, showlegend=False)
    else:
        chart4 = px.scatter(title="Данные о местности отсутствуют (Нужно запустить Агент 1.2)")
        
    # 6. Высота маршрута vs Частота шагов (Взаимосвязь физических сил)
    chart5 = px.scatter(df, x="elevation", y="step_frequency", opacity=0.5, color="Season")
    chart5.update_layout(margin={"r":10,"t":10,"l":10,"b":10})
    
    # 7. Наиболее популярные маршруты (Гистограмма треков по периодам/сезонам)
    # Считаем количество записей по треку, берем топ 15 самых долгих/популярных
    popular = df.groupby(['Season', 'track_id']).size().reset_index(name='count')
    top_tracks = popular.groupby('track_id')['count'].sum().nlargest(15).index
    popular = popular[popular['track_id'].isin(top_tracks)]
    chart6 = px.bar(popular, x="track_id", y="count", color="Season", barmode='group')
    chart6.update_xaxes(type='category')
    chart6.update_layout(margin={"r":10,"t":10,"l":10,"b":10})

    return kpi_steps, kpi_temp, kpi_count, map_fig, chart1, chart2, chart3, chart4, chart5, chart6


if __name__ == '__main__':
    print("Запуск Дашборда на http://127.0.0.1:8050")
    app.run(debug=True, use_reloader=False, port=8050)
