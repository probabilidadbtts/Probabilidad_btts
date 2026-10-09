# BTTS Multi-Model Predictor – TheStatsAPI

Predictor de Both Teams To Score (BTTS) basado en 5 modelos reales y un ensemble combinado.

## Qué hace

- Consulta partidos futuros con `date_from` y `date_to` desde TheStatsAPI
- Obtiene historial reciente por equipo
- Evalúa 5 modelos de BTTS
- Calcula un ensemble y filtra por umbral mínimo
- Exporta el resultado a CSV

## Requisitos

- Python 3.11+
- Una clave API válida de TheStatsAPI en `THESTATSAPI_KEY`

## Configuración

```bash
cp env.example .env
# editar .env con tu THESTATSAPI_KEY
```

Variables soportadas:

- `THESTATSAPI_KEY`
- `DATE_FROM`
- `DATE_TO`
- `MIN_PCT` (default 80)
- `TOP_N` (default 20)
- `LOG_LEVEL` (default INFO)

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --date-from 2026-10-09 --date-to 2026-10-09 --min-pct 80 --top-n 20
```

## GitHub Actions

El workflow `.github/workflows/daily_btts.yml` ejecuta el proyecto cada día y también permite dispararlo manualmente con selector de fechas.
