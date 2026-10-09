# BTTS Multi-Model Predictor – TheStatsAPI

Predictor de **Both Teams To Score (BTTS)** con 5 modelos reales:

| Modelo              | Descripción                              |
|---------------------|------------------------------------------|
| Historical          | Tasa BTTS de los últimos 10 partidos     |
| Bivariate Poisson   | Poisson bivariada clásica                |
| LSTM Momentum       | LSTM sobre secuencia de forma reciente   |
| XGBoost             | Gradient Boosting con features de goles  |
| CatBoost            | CatBoost con las mismas features         |

**Ensemble** = promedio de los 5 modelos. Solo se muestran partidos con **ensemble > 80 %**.

## Uso local

```bash
cp .env.example .env
# edita .env y pon tu THESTATSAPI_KEY
pip install -r requirements.txt
python main.py
