# Performance Model 8+4 · Streamlit

Aplicación bilingüe **Español / English** para pronosticar el desempeño financiero 2026 por **Planta + Cost Bucket + mes** y consolidarlo por **Planta + mes**. Usa un `RandomForestRegressor` global y reglas financieras posteriores para clasificar cada resultado como **Favorable**, **En riesgo** o **Desfavorable**.

> El archivo `2026 Performance Model Consolidation 8+4.1.xlsx` **no está incluido**. Debe cargarse manualmente desde la interfaz. `.gitignore` excluye los archivos Excel, modelos y exportaciones para evitar publicar datos financieros.

## Funcionalidad incluida

- Enero–agosto de 2026 como reales; septiembre–diciembre como meses pronosticados.
- Entrenamiento con Actuals 2026, Prior Year 2025 y Current Forecast 8+4 como predictor y referencia.
- Comparación del modelo mensual unificado frente a modelos independientes para Sep/Oct/Nov/Dec, con selección automática por menor RMSE de backtesting y sin predicción recursiva.
- Backtesting temporal 8+4 con MAE, RMSE, R² y error porcentual.
- Respaldo jerárquico: Planta/Cost Bucket → BU/Cost Bucket → modelo global → Forecast 8+4.
- Detección de categorías nuevas, advertencia, menor confianza y recomendación de reentrenamiento.
- Limpieza controlada: duplicados, exclusión de agregados FY/Q1–Q4, imputación configurable, extremos configurables y reporte de calidad.
- Modo ejecutivo: referencias, escenarios, materialidad, semáforos e intervalos por dispersión entre árboles.
- Modo avanzado: hiperparámetros, optimización limitada con semilla fija, bootstrap residual y diagnósticos.
- Escenarios what-if para Volume, Labor, Utilities, MUV, Faulty, Maintenance y ajustes de enero–agosto.
- Explicabilidad global e individual y resumen financiero.
- Dashboard Plotly con KPI, series, intervalos, ranking, heatmap, waterfall, comparaciones y tablas; diagnósticos Matplotlib/Seaborn.
- Descargas de Excel multipestaña, CSV individuales y modelo `.joblib` reutilizable.
- Carga del Excel original, CSV normalizado o plantilla CSV descargable.

## Estructura

```text
.
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Ejecución local

Requiere Python 3.11 o 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abra la URL local que muestra Streamlit. En la barra lateral:

1. Cargue manualmente `2026 Performance Model Consolidation 8+4.1.xlsx`.
2. Revise validación, calidad, imputación y tratamiento de extremos.
3. Defina filtros BU/POD/planta, referencia, umbrales y escenarios.
4. Pulse **Entrenar / actualizar modelo**.
5. Revise dashboard, explicabilidad y diagnósticos.
6. Descargue resultados, CSV y el modelo entrenado.

Los valores del libro original están expresados en **miles de USD**; la interfaz y las columnas monetarias exportadas se presentan en **USD**. La aplicación normaliza el impacto como `Prior Year 2025 – Actual 2026`, consistente con la lógica EBITDA del libro.

## Publicar en GitHub

1. Cree un repositorio vacío en GitHub.
2. Descomprima este proyecto y ejecute:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial Streamlit performance model"
git branch -M main
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```

3. Confirme antes del push que el Excel no está versionado:

```bash
git status
git check-ignore "2026 Performance Model Consolidation 8+4.1.xlsx"
```

No use `git add -f` con archivos financieros. Si el Excel ya se agregó por error, elimínelo del índice con:

```bash
git rm --cached "2026 Performance Model Consolidation 8+4.1.xlsx"
```

## Publicar en Streamlit Community Cloud

1. Suba el repositorio a GitHub sin el Excel.
2. Entre a [share.streamlit.io](https://share.streamlit.io/).
3. Seleccione **Create app** y conecte el repositorio.
4. Configure:
   - Branch: `main`
   - Main file path: `app.py`
5. Despliegue la aplicación.
6. En cada sesión, cargue el Excel manualmente desde la barra lateral.

La aplicación no requiere secretos. Streamlit Cloud usa almacenamiento efímero: descargue el modelo `.joblib` si desea reutilizarlo en otra sesión.

## Formato CSV normalizado

La plantilla descargable desde la app incluye estas columnas:

| Columna | Requerida | Descripción |
|---|---:|---|
| `BU` | No | Business Unit; si falta, se usa `UNKNOWN`. |
| `POD` | No | POD; si falta, se deriva de BU. |
| `Plant` | Sí | Planta. |
| `Cost_Bucket` | Sí | Labor, Utilities, MUV, etc. |
| `Month` | Sí | Número 1–12 o nombre de mes en español/inglés. |
| `Actual_2026` | Sí | Real 2026; Sep–Dec se ignora como real. |
| `Prior_Year_2025` | Sí | Valor mensual de 2025. |
| `Current_Forecast_8_4` | Sí | Referencia mensual 8+4. |
| `Unit_Scale` | No | Multiplicador a USD; predeterminado `1000`. |

Se aceptan alias comunes en español e inglés. La validación rechaza hojas o columnas incompatibles y meses inválidos.

## Modelo y criterios financieros

- **Objetivo numérico:** impacto EBITDA mensual, calculado como `Prior Year – Actual` para datos de costo. Un impacto mayor es favorable.
- **Clasificación:** exige simultáneamente el umbral porcentual y la materialidad monetaria. Valores predeterminados: **5%** y **USD 50,000**.
- **En riesgo:** desviaciones que no superan ambos umbrales.
- **Intervalos ejecutivos:** percentiles 10–90 de predicciones entre árboles.
- **Bootstrap avanzado:** amplía el intervalo con remuestreo de residuos del entrenamiento.
- **Backtesting:** orígenes temporales julio y agosto. El modelo unificado se compara con la estrategia independiente, que usa meses análogos por posición de ciclo de cuatro meses.
- **No recursivo:** Sep–Dec se predicen directamente con historia disponible hasta agosto; una predicción futura no alimenta otra.
- **Optimización:** conjunto deliberadamente limitado de configuraciones para controlar tiempo y riesgo de sobreajuste; semilla fija `42`.

## Reutilización y compatibilidad

El archivo `.joblib` guarda el pipeline, categorías, ajustes residuales jerárquicos, configuración y metadatos. La app verifica versión y huella del esquema antes de reutilizarlo. Si aparecen datos más recientes o categorías nuevas, muestra advertencias y recomienda reentrenar.

Los archivos `.joblib` pueden ejecutar objetos Python al cargarse. Cargue únicamente modelos generados por esta aplicación y provenientes de una fuente confiable.

## Prueba incluida en `app.py`

Para validar el pipeline sin abrir el navegador:

```bash
python app.py --self-test "/ruta/2026 Performance Model Consolidation 8+4.1.xlsx"
```

La prueba carga y valida el libro, normaliza los datos, entrena ambos enfoques, selecciona estrategia, genera Sep–Dec y verifica que todas las predicciones sean finitas. El Excel de prueba no se copia ni se modifica.

## English quick start

Create a virtual environment, install `requirements.txt`, and run `streamlit run app.py`. Upload the original workbook manually, review data quality, configure thresholds and scenarios, train, and download the outputs/model. Do not commit the source workbook; it is intentionally excluded by `.gitignore`.
