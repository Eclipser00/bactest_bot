# Flujo propuesto para v6.0 (con anti-overfitting y mesetas):

## Disclaimer

**Todo el código de este proyecto ha sido desarrollado utilizando vibe coding (codificación asistida por IA).**

Este proyecto es el resultado de desarrollo colaborativo con herramientas de inteligencia artificial. El código ha sido generado y refinado mediante asistencia de IA, siguiendo las mejores prácticas de programación y los estándares establecidos en el proyecto.

## Preparación inicial

Tú pones CSV en /data.

Configuras main.py (qué estrategias, optimización, parámetros, formato, tipo de chequeo de overfitting: CSCV, DSR, etc.).

## Carga de datos

Se leen los CSV con functions.csv_to_df_*.

Se validan con functions.validate_ohlcv().

Se guardan logs de: activo, nº de filas, rango temporal.

## Backtest base

Por cada activo y estrategia:

Se construye el grid de parámetros (build_param_grid).

Para cada combinación → strategies.run_strategy().

Se obtiene equity_curve, trades, métricas base.

Se guarda en /stats (cada combinación).

## Visualización preliminar

Se crean heatmaps de métricas (equity, DD, Sharpe).

Aquí ya puedes ver dónde están los “picos” y las “mesetas”.

## Anti-overfitting (overfitting.py)

Cada set de parámetros se pasa por:

CSCV + PBO → probabilidad de overfitting.

Stress de costes → sensibilidad a cambios en comisiones/slippage. solo OPTIMIZE = False, para un parametro concreto. (AUN SIN PROBAR FUNCIONAMIENTO)

Walk-Forward → estabilidad IS/OOS. (AUN SIN PROBAR FUNCIONAMIENTO)

## Selección de meseta robusta

Se filtra top X% de sets por equity/Sharpe OOS.

Se agrupan en clusters (vecindarios de parámetros).

Se elige un clúster estable (baja varianza, DD acotado, robustez alta).

Se selecciona un set representativo (mediana del clúster).

## Outputs finales

Solo el set robusto elegido se exporta a:

/plots: curva de equity y DD.

/trades: Excel con los trades.

/results: Excel con un resumen (1 fila por activo, con métricas + robustez).

Se loguea todo el proceso: desde parámetros hasta por qué se eligió ese set.

## 👉 En resumen:

Tu v4 buscaba el mejor resultado (maximize sobre un parámetro).

Tu v5 buscará una meseta robusta, donde los parámetros no sean frágiles y pasen filtros de sobreajuste.

Tu  v6 Cambio de libreria a Backtrader para mejorar estrategias.

![License: Personal Non-Commercial](https://img.shields.io/badge/license-personal_non_commercial-informational)

