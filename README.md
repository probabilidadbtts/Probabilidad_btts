```markdown
# BTTS Multi-Model Predictor – TheStatsAPI

Predictor de Both Teams To Score (BTTS) basado en 5 modelos reales, ensemble combinado y análisis dinámico de bajas.

## Qué hace

- Consulta partidos futuros con `date_from` y `date_to` desde TheStatsAPI
- Obtiene historial reciente por equipo (últimos 30 partidos)
- Obtiene bajas actuales por equipo desde `/football/teams/{id}/injuries`
- Evalúa 5 modelos de BTTS:
  - **Historical**: Tasa BTTS de últimos 10 partidos
  - **Bivariate Poisson**: Modelo clásico de Poisson bivariada
  - **LSTM Momentum**: Red neuronal sobre secuencias recientes
  - **XGBoost**: Gradient Boosting con features de goles
  - **CatBoost**: CatBoost con features de forma
- Calcula ensemble promediado de todos los modelos
- **Ajusta el ensemble según bajas críticas** (delanteros, extremos)
- Filtra por umbral mínimo configurado
- Reordena por score ajustado (descendente)
- Exporta a CSV incluyendo impacto de bajas

## Requisitos

- Python 3.11+
- Una clave API válida de TheStatsAPI en `THESTATSAPI_KEY`

## Configuración

```bash
cp env.example .env
# editar .env con tu THESTATSAPI_KEY
```

Variables soportadas:

- `THESTATSAPI_KEY` (requerido)
- `DATE_FROM` (YYYY-MM-DD, default: hoy UTC)
- `DATE_TO` (YYYY-MM-DD, default: DATE_FROM)
- `MIN_PCT` (default 80, umbral del ensemble ajustado)
- `TOP_N` (default 20, máximo de resultados)
- `LOG_LEVEL` (default INFO: DEBUG, INFO, WARNING, ERROR)

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --date-from 2026-10-09 --date-to 2026-10-09 --min-pct 80 --top-n 20
```

## Salida

```
#  Fecha UTC          Local         Visitante     Hist%  Poisson%  LSTM%  XGB%  Cat%  Ensemble%  Ajustado%  Chg  Bajas (L/V)
1  2026-10-09 15:00  Barcelona     Real Madrid   82.5   81.2      83.0   82.8  82.1  82.3       81.5       ↓    1/0
2  2026-10-09 17:30  Arsenal       Liverpool     79.2   80.1      81.5   79.8  80.3  80.2       80.2       =    0/0
```

**Columnas:**
- `Ensemble%`: Promedio puro de los 5 modelos
- `Ajustado%`: Ensemble ajustado por bajas (si hay delanteros o extremos lesionados)
- `Chg`: `↓` (reducido por bajas) o `=` (sin cambio)
- `Bajas (L/V)`: Número de bajas relevantes Local/Visitante

## GitHub Actions

El workflow `.github/workflows/daily_btts.yml` ejecuta automáticamente:

- **Diariamente** a las 06:00 UTC (solo partidos de hoy)
- **Manualmente** desde "Actions" con selector de fechas, min_pct y top_n

**Requisito:** Configurar el secret `THESTATSAPI_KEY` en:
`Settings → Secrets and variables → Actions → New repository secret`

Cada ejecución descarga un artefacto CSV por 14 días.

## Impacto de bajas

El análisis categoriza jugadores por posición:

| Posición | Impacto | Efecto |
|----------|--------|--------|
| Delantero (ST/CF) | 0.90 | Muy alto (reduce BTTS) |
| Extremo (LW/RW/AM) | 0.75 | Alto (reduce ataque) |
| Centrocampista (CM) | 0.50 | Medio (afecta creación) |
| Portero (GK) | 0.70 | Alto (favorece No-BTTS) |
| Defensa (CB/LB/RB) | 0.40 | Bajo (afecta defensa) |

Si un equipo pierde delanteros/extremos (impact > 0.5), el ensemble se reduce un ~20%.

## Tests

```bash
pytest tests/
```

## Estructura

```
src/
  cli.py              Entrada y flujo principal
  config.py           Configuración centralizada
  api_client.py       Cliente HTTP con reintentos
  data_fetcher.py     Obtención de partidos + historial
  analyzer.py         Orquestador de modelos + bajas
  injuries.py         Análisis de impacto de bajas
  exporter.py         CSV + tabla de consola
  models/             Modelos de predicción (5 implementaciones)
  utils.py            Funciones auxiliares
tests/                Tests unitarios
.github/workflows/    CI/CD (Daily + Manual)
```
```

**Instrucciones para actualizar el README:**

Copiar el contenido anterior y reemplazar el archivo `.github/workflows/daily_btts.yml` y `README.md` en el repo.

El workflow que está ahora **es el correcto y no necesita cambios**. La lógica de bajas se ejecuta automáticamente dentro del análisis.
