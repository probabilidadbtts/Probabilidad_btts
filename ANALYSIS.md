# Análisis Profundo del Repositorio: Probabilidad_btts

## Resumen Ejecutivo

**Probabilidad_btts** es un predictor de **Both Teams To Score (BTTS)** que utiliza 5 modelos de machine learning integrados (Historical, Bivariate Poisson, LSTM Momentum, XGBoost, CatBoost) con ensemble promediado y ajuste dinámico por bajas de jugadores. Consume datos de TheStatsAPI, procesa partidos programados en 20 ligas top, analiza historial reciente de equipos, y exporta predicciones en CSV y tabla de consola. Automatizado vía GitHub Actions para ejecuciones diarias o manuales.

---

## 1. Stack Técnico

### Lenguaje y Runtime
- **Python 3.11+** (requirement especificado en workflow)
- No framework web (CLI puro)

### Dependencias Críticas
| Librería | Versión | Propósito |
|----------|---------|----------|
| `requests` | ≥2.32.0 | Cliente HTTP robusto con reintentos |
| `numpy` | ≥1.26.0 | Operaciones numéricas base |
| `pandas` | ≥2.2.0 | Manipulación de datos |
| `scipy` | ≥1.13.0 | Cálculos estadísticos (Poisson) |
| `scikit-learn` | ≥1.5.0 | Utilidades ML, normalización |
| `xgboost` | ≥2.1.0 | Modelo XGBoost |
| `catboost` | ≥1.2.5 | Modelo CatBoost |
| `torch` | ≥2.3.0 | Modelo LSTM (redes neuronales) |
| `python-dotenv` | ≥1.0.0 | Gestión de variables de entorno |
| `tabulate` | ≥0.9.0 | Formateo de tablas en consola |
| `pytest` | ≥8.3.0 | Framework de tests unitarios |

---

## 2. Estructura y Arquitectura

### Árbol de Directorios

```
Probabilidad_btts/
├── main.py                    # Punto de entrada (wrapper simple)
├── requirements.txt           # Dependencias pip
├── .env.example / env.example # Plantilla de variables de entorno
├── README.md                  # Documentación
├── .gitignore                 # Archivos ignorados
│
├── src/                       # Código principal
│   ├── __init__.py
│   ├── cli.py                 # Parseo de args, flujo principal y orquestación
│   ├── config.py              # Dataclass Settings, normalización de inputs
│   ├── api_client.py          # Cliente HTTP StatsAPI (reintentos + rate limit)
│   ├── data_fetcher.py        # Obtención de partidos y históricos
│   ├── analyzer.py            # Orquestador de 5 modelos + ensemble
│   ├── injuries.py            # Análisis de impacto de bajas (lineups confirmados)
│   ├── lineups.py             # Extracción de absencias desde alineaciones
│   ├── exporter.py            # Exportación CSV + tabla de consola
│   ├── utils.py               # Funciones auxiliares
│   ├── top_leagues.py         # Configuración de ligas filtradas (20 top)
│   │
│   └── models/                # 5 implementaciones de predicción
│       ├── __init__.py        # Exporta los 5 modelos
│       ├── base.py            # Clase abstracta BaseBTTSModel
│       ├── historical.py      # Modelo 1: BTTS de últimos 10 partidos
│       ├── bivariate_poisson.py # Modelo 2: Poisson bivariada
│       ├── lstm_momentum.py    # Modelo 3: Red neuronal recurrente
│       ├── xgboost_model.py    # Modelo 4: Gradient Boosting
│       └── catboost_model.py   # Modelo 5: CatBoost
│
├── tests/                     # Suite de tests unitarios (directorio vacío)
│
└── .github/
    └── workflows/
        └── daily_btts.yml     # Workflow GitHub Actions (cron + manual dispatch)
```

### Flujo de Datos

```
┌────────────────────────────────────────────────────────────────┐
│  CLI Entry Point (main.py → cli.py)                            │
└────────────────┬───────────────────────────────────────────────┘
                 │ parse_args + Settings.from_dict
                 ▼
┌────────────────────────────────────────────────────────────────┐
│  StatsAPIClient (api_client.py)                                │
│  • Reintentos automáticos (Retry Strategy)                     │
│  • Rate limiting (0.30s entre requests)                        │
│  • Manejo de 429 (Too Many Requests)                           │
└────────────────┬───────────────────────────────────────────────┘
                 │ session.get() → JSON
                 ▼
┌────────────────────────────────────────────────────────────────┐
│  DataFetcher (data_fetcher.py)                                 │
│  • get_upcoming_matches() → partidos programados               │
│  • get_team_finished_matches() → últimos 30 partidos/equipo    │
│  • LeagueFilter.filter_matches() (actualmente deshabilitado)   │
└────────────────┬───────────────────────────────────────────────┘
                 │ List[Dict] → matches
                 ▼
┌────────────────────────────────────────────────────────────────┐
│  BTTSAnalyzer (analyzer.py)                                    │
│  1. Iterar sobre cada partido programado                       │
│  2. Cache de equipos: get_team_finished_matches() × 2          │
│  3. Para cada modelo:                                           │
│     - predict(home_matches, away_matches) → float [0-100]      │
│  4. Ensemble = promedio de scores válidos                      │
│  5. Si match_id: analyze_injuries() → ajustar ensemble         │
└────────────────┬───────────────────────────────────────────────┘
                 │ List[Dict] → results
                 ▼
┌────────────────────────────────────────────────────────────────┐
│  InjuryImpactAnalyzer (injuries.py) + LineupAnalyzer           │
│  • Fetch LineupAnalyzer.analyze_match_lineups()                │
│  • Extract absences from confirmed lineup                      │
│  • Calculate impact by position (0.3–0.9)                      │
│  • Adjust ensemble: ensemble *= (1 - impact * 0.20)            │
└────────────────┬───────────────────────────────────────────────┘
                 │ Updated scores with injury adjustment
                 ▼
┌────────────────────────────────────────────────────────────────┐
│  Exporter (exporter.py)                                        │
│  • print_table() → tabla con tabulate + console                │
│  • export_csv() → btts_DATE_FROM_to_DATE_TO.csv                │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Componentes Principales

### 3.1 **cli.py** – Entrada y Orquestación
- **`parse_args()`**: Parsea línea de comandos y env vars (`DATE_FROM`, `DATE_TO`, `MIN_PCT`, `TOP_N`, `LOG_LEVEL`)
- **`cli()`**: 
  - Crea `StatsAPIClient` y `DataFetcher`
  - **BUG IDENTIFICADO**: Instancia `BTTSAnalyzer(fetcher, client)` pero la clase solo acepta `fetcher`
  - Obtiene partidos futuros
  - Ejecuta análisis con umbrales
  - Exporta resultados
  - Manejo robusto de excepciones

### 3.2 **config.py** – Configuración Centralizada
- Dataclass `Settings` (inmutable/frozen)
- `from_dict()` classmétodo: normaliza inputs, aplica defaults
- Soporta env vars: `THESTATSAPI_KEY`, `DATE_FROM`, `DATE_TO`, `MIN_PCT`, `TOP_N`, `LOG_LEVEL`

### 3.3 **api_client.py** – Cliente Robusto
- `StatsAPIClient` con:
  - **Reintentos automáticos**: backoff exponencial (1.8×), max 4 intentos
  - **Rate limiting**: 0.30s entre requests
  - **Manejo 429**: lee header `Retry-After`, duerme dinámicamente
  - **Session reutilizable**: headers persistentes, Auth Bearer
  - **Timeout**: 30 segundos por request
- Excepción personalizada si falta `THESTATSAPI_KEY`

### 3.4 **data_fetcher.py** – Obtención de Datos
- `get_upcoming_matches()`: 
  - Rango de fechas explícito o legacy `days_ahead`
  - Paginación: 100 matches/página
  - Filtra por `status=scheduled`
  - Aplica `LeagueFilter.filter_matches()` (actualmente vacío)
- `get_team_finished_matches()`:
  - Últimos N partidos finalizados de un equipo
  - Ordena descendente (más recientes primero)
  - Cache en analyzer para evitar duplicados

### 3.5 **analyzer.py** – Orquestador de Modelos
- `BTTSAnalyzer.__init__(fetcher)` **SOLO ACEPTA FETCHER**
- `analyze(upcoming, min_pct, top_n)`:
  1. Itera partidos programados
  2. Cache dinámico de históricos (home_id → 30 últimos partidos)
  3. Ejecuta 5 modelos:
     - `HistoricalBTTS(n=10)`
     - `BivariatePoisson(n=20)`
     - `LSTMMomentum(seq_len=8, epochs=35)`
     - `XGBoostBTTS(n=15)`
     - `CatBoostBTTS(n=15)`
  4. Ensemble = promedio de scores > 0
  5. Filtra `ensemble > min_pct`
  6. Ordena descendente
  7. Top N resultados

### 3.6 **injuries.py** – Análisis de Bajas
- `InjuryImpactAnalyzer`:
  - Categoriza jugadores por posición (impacto 0.3–0.9)
    - Delanteros/Extremos: 0.90–0.75
    - Centrocampistas: 0.50
    - Defensores: 0.40
    - Porteros: 0.70
  - `analyze_injuries()`: obtiene bajas desde lineups confirmados
  - Fallback seguro: si no hay lineups, retorna 0 impacto (conservador)
  - `adjust_ensemble_by_injuries()`: reduce score si impacto > 0.5

### 3.7 **lineups.py** – Extracción de Alineaciones
- `LineupAnalyzer`:
  - `fetch_lineups()`: GET `/football/matches/{match_id}/lineups`
  - `extract_absences_from_lineup()`: detecta jugadores con status `out/injured/suspended`
  - `evaluate_absence_impact()`: promedio de impactos por posición
  - `analyze_match_lineups()`: orquesta obtención + evaluación
- Robusto frente a estructura variable en API

### 3.8 **models/** – 5 Implementaciones de Predicción

#### **base.py** – Clase Abstracta
```python
class BaseBTTSModel(ABC):
    name: str = "base"
    @abstractmethod
    def predict(home_matches, away_matches) -> float:
        # Retorna probabilidad [0-100]
```

#### **1. historical.py**
- BTTS % en últimos 10 partidos de cada equipo
- Promedia y escala a [0-100]
- **Pros**: Simple, rápido
- **Contras**: No incluye contexto temporal

#### **2. bivariate_poisson.py**
- Modelo clásico Poisson bivariada para predicción de goles
- Estima λ_home y λ_away
- Calcula P(home_goals > 0 AND away_goals > 0)
- **Pros**: Teórico, probado en literatura
- **Contras**: Supone independencia (no siempre cierto)

#### **3. lstm_momentum.py**
- Red neuronal recurrente LSTM
- Entrada: secuencia de 8 partidos recientes (goles casa/fuera)
- Oculta: 32 neuronas
- Salida: 1 neurona (sigmoid, [0,1])
- **Pros**: Captura patrones temporales
- **Contras**: Requiere torch, más overhead, posible overfit

#### **4. xgboost_model.py**
- Gradient Boosting
- Features: promedio de goles casa/fuera, varianza, ratio defensivo
- **Pros**: Maneja no-linealidad, features importancia
- **Contras**: Necesita tuning, interpretabilidad

#### **5. catboost_model.py**
- CatBoost (variante de Gradient Boosting)
- Similar a XGBoost pero con manejo nativo de categóricas
- **Pros**: Rápido, menos tuning
- **Contras**: Similar a XGBoost en interpretabilidad

### 3.9 **exporter.py** – Salida
- `print_table()`:
  - Usa `tabulate` para formateo "github"
  - Columnas: #, Fecha, Local, Visitante, Hist%, Poisson%, LSTM%, XGB%, Cat%, Ensemble%, Ajustado%, Chg (cambio), Bajas (L/V)
  - Símbolo "↓" si ensemble fue reducido por lesiones
- `export_csv()`:
  - Nombre: `btts_{date_from}_to_{date_to}.csv`
  - Headers: todas las claves del dict result
  - Incluye impacto de lesiones

### 3.10 **top_leagues.py** – Filtrado de Ligas
- `TOP_20_LEAGUES`: dict {id → nombre} de 20 ligas top
  - Premier League (39), La Liga (140), Serie A (135), Bundesliga (78), Ligue 1 (61), etc.
- `LeagueFilter.filter_matches()`: **Actualmente deshabilitado** (retorna todos los partidos)
- Posible activación futura si se requiere restricción

### 3.11 **utils.py** – Funciones Auxiliares
- Vacío o con funciones menores (no se lista contenido explícito)

---

## 4. Flujo de Ejecución (End-to-End)

### Escenario: Ejecución Manual vía GitHub Actions

```yaml
# .github/workflows/daily_btts.yml
on:
  schedule:
    - cron: "0 6 * * *"  # 06:00 UTC diario
  workflow_dispatch:     # Dispatch manual con inputs
```

**Pasos:**
1. **Checkout** → clone repo
2. **Setup Python 3.11** + cache pip
3. **Install** → pip install -r requirements.txt
4. **Dates logic** → si cron: hoy; si manual: inputs
5. **Run CLI**:
   ```bash
   python main.py \
     --date-from $DATE_FROM \
     --date-to $DATE_TO \
     --min-pct $MIN_PCT \
     --top-n $TOP_N
   ```
6. **Upload CSV** → artifact por 14 días

---

## 5. Identificación de Problemas y Mejoras

### **PROBLEMA CRÍTICO** ✗
**cli.py:57** → `analyzer = BTTSAnalyzer(fetcher, client)` 

**BTTSAnalyzer.__init__()** solo acepta **1 argumento** (`self`, `fetcher`), pero se pasan **2**.

**Síntoma**: `TypeError: BTTSAnalyzer.__init__() takes 2 positional arguments but 3 were given`

**Solución**: Remover `client` de la llamada (el analyzer no lo necesita).

---

### **Problemas de Diseño y Mejoras Potenciales**

| Área | Problema | Impacto | Recomendación |
|------|----------|--------|-----------------|
| **Modelos** | LSTM necesita ≥8 partidos; si equipo nuevo, falla silenciosamente | Bajo a Medio | Try/except + fallback a 0.0 |
| **Caché** | Sin validación TTL; si fetcher es compartido, edad desconocida | Bajo | Agregar timestamp a cache |
| **Tests** | Directorio `tests/` vacío | Alto | Agregar tests para cli, models, fetcher |
| **Lineup** | Estructura API variante; puede no detectar todas las ausencias | Medio | Documentar payload esperado; test con mocks |
| **LeagueFilter** | Deshabilitado (acepta todos); documentación desactualizada | Bajo | Re-habilitar si se necesita o remover |
| **Error Handling** | `except Exception` demasiado amplio en algunos lugares | Bajo | Ser más específico (e.g., RequestException, JSONDecodeError) |
| **Logging** | Verboso en INFO; difícil de seguir si DEBUG | Bajo | Restructurar logs, usar módulos específicos |
| **Env Vars** | Duplicados (.env.example vs env.example) | Trivial | Unificar a .env.example |
| **Type Hints** | Incompletos en algunos módulos (e.g., utils.py) | Bajo | Agregar typing completo, mypy |

---

## 6. Configuración y Secretos

### Variables de Entorno Requeridas

| Variable | Ejemplo | Requerida | Defecto |
|----------|---------|-----------|---------|
| `THESTATSAPI_KEY` | `abc123...` | ✓ Sí | — |
| `DATE_FROM` | `2026-10-09` | No | Hoy (UTC) |
| `DATE_TO` | `2026-10-09` | No | DATE_FROM |
| `MIN_PCT` | `75` | No | `80` |
| `TOP_N` | `20` | No | `20` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` | No | `INFO` |

### Setup en GitHub Actions
```
Settings → Secrets and variables → Actions → New repository secret
  Name: THESTATSAPI_KEY
  Value: <tu clave de TheStatsAPI>
```

---

## 7. Dependencias Externas

### TheStatsAPI
- **Base URL**: `https://api.thestatsapi.com/api`
- **Endpoints usados**:
  - `GET /football/matches?status=scheduled&date_from=X&date_to=Y&page=Z`
  - `GET /football/matches?team_id=X&status=finished`
  - `GET /football/matches/{match_id}/lineups`
  - `GET /football/teams/{team_id}` (si se activa fallback de roster)

### Licencias y Créditos
- No se especifica licencia en repo
- **Recomendación**: Agregar LICENSE (MIT, Apache 2.0 o equivalente)

---

## 8. Casos de Uso

### 1. **Daily Automation**
```bash
# Se ejecuta cada día a las 06:00 UTC vía cron
→ Predice BTTS para partidos de hoy
→ Exporta CSV (descargable 14 días)
```

### 2. **Manual Ad-hoc Queries**
```bash
# Via GitHub Actions dispatcher
Inputs:
  - date_from: 2026-10-01
  - date_to: 2026-10-31
  - min_pct: 75
  - top_n: 50
→ Genera CSV histórico
```

### 3. **Local Development**
```bash
source .venv/bin/activate
export THESTATSAPI_KEY=<tu_clave>
python main.py --date-from 2026-10-09 --date-to 2026-10-09 --min-pct 75 --top-n 10
```

---

## 9. Resumen de Fortalezas

✅ **Ensemble robusto**: 5 modelos heterogéneos (estadístico + ML)  
✅ **Cliente HTTP resiliente**: reintentos, rate limiting, timeouts  
✅ **Bajas inteligentes**: integración con lineups confirmados  
✅ **CLI flexible**: parámetros vía env vars o argumentos  
✅ **Exportación dual**: consola + CSV descargable  
✅ **Automatización**: cron + manual dispatch  
✅ **Código limpio**: type hints, logging, docstrings  

---

## 10. Resumen de Debilidades

❌ **Bug crítico**: cli.py instancia BTTSAnalyzer incorrectamente (SOLUCIONADO)  
❌ **Tests faltantes**: directorio vacío, no hay cobertura  
❌ **Documentación incompleta**: README tiene línea de actualización rota  
❌ **LeagueFilter deshabilitado**: 20 ligas configuradas pero sin efecto  
❌ **Env duplicados**: .env.example vs env.example  
❌ **LSTM poco robusto**: requiere ≥8 partidos, sin fallback documentado  
❌ **Type hints incompletos**: utils.py y otros sin anotaciones  

---

## 11. Recomendaciones de Mejora Inmediata

1. **[CRÍTICO]** Fijar bug de cli.py ✓ (YA HECHO)
2. **[IMPORTANTE]** Agregar tests unitarios básicos
3. **[IMPORTANTE]** Completar type hints con mypy
4. **[RECOMENDADO]** Re-habilitar o remover LeagueFilter
5. **[RECOMENDADO]** Consolidar archivos de env (.env.example)
6. **[RECOMENDADO]** Agregar LICENSE y CHANGELOG
7. **[RECOMENDADO]** Mejorar documentación de payload de lineups
8. **[RECOMENDADO]** Agregar métricas de validación (accuracy retroactivo)

---

## Conclusión

**Probabilidad_btts** es un proyecto bien estructurado, con separación clara de responsabilidades y uso de patrones modernos de Python. El análisis multi-modelo y la integración con datos de lesiones demuestran madurez. El bug identificado en cli.py es fácil de corregir pero crítico para la ejecución. Con los tests y la documentación completada, será un predictor BTTS robusto y mantenible.

