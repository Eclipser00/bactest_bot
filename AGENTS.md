# AGENTS.md

## 0) Propósito

Este documento define **cómo debe trabajar un agente** (IA/coder) en este proyecto:

* Qué puede y no puede cambiar.
* Estándares de  **estilo, testing, logs y comentarios** .
* **Comandos de entorno** (instalación, ejecución, tests).
* **Reglas de seguridad** (secretos, datos).
* **Criterios de calidad** y checklist antes de abrir un PR.

## 1) Stack y alcance

* **Lenguaje** : Siempre **Python** (3.10.11). No usar otros lenguajes.
* **Trading** : Base obligatoria  **Backtrader** . Pandas/Numpy para data pipes.
* **Paradigma** : Priorizar  **Programación Orientada a Objetos (POO)** , clases limpias y módulos ordenados.
* **Estilo** :  **Código minimalista y autoexplicativo** . Donde no sea autoexplicativo, añadir **comentarios extensos** tras cada función/bloque.
* **Logging** : Incluir **muchos logs** (nivel DEBUG/INFO) a lo largo del flujo para trazar cada paso. Se podrán desactivar más tarde.
* **Formato** : **No usar emoticonos** en el código ni en mensajes de log.

## 2) Principios de edición (muy importante)

1. **Respeta el código estable** : Si el usuario indica que ciertos bloques  **no deben modificarse** ,  **no los toques** . Si es imprescindible, propón cambios aislados y justificados.
2. **Cambios mínimos** : Prefiere refactorizaciones **incrementales** y atómicas.
3. **Retrocompatibilidad** : No rompas interfaces públicas (nombres de clases/métodos/funciones) sin migración clara.
4. **Explica lo no obvio** : Si introduces lógica compleja, añade comentarios extensos al final del bloque/función.
5. **Logs primero** : Todo nuevo flujo debe traer logs suficientemente detallados de entrada, salida y decisiones.

## 3) Reglas de decisión del agente

1. Si hay **AGENTS.md** en la subcarpeta actual, úsalo con **prioridad** sobre el de la raíz.
2. Si el usuario da una instrucción explícita,  **esa instrucción manda** .
3. En caso de conflicto entre estilo y funcionalidad, **mantén la funcionalidad** y documenta el compromiso.

## 4) Mantenimiento

* Este archivo es  **vivo** : actualízalo cuando cambien convenciones o tooling.
* Añade ejemplos concretos de comando/estructura según el proyecto.
* Revisa que las secciones reflejen  **cómo trabajas realmente** .

# Reglas para crear nuevas estrategias

## Objetivo

Añadir estrategias nuevas que:

1. Respeten las **interfaces y firmas existentes** (param names, base class, helpers).
2. Usen **OOP con `_BaseLoggedStrategy`** y las **utilidades ya implementadas** para sizing y pivotes.
3. Mantengan **compatibilidad binaria** con los módulos de backtest/optimización/plots/export.

## Convenciones obligatorias

1. **Clase base**

* Hereda SIEMPRE de **`_BaseLoggedStrategy`** (aporta logs de órdenes/trades y helpers de size). No crear otras bases.

2. **Parámetros (firma estable)**

* Define `params = (('n1', ...), ('n2', ...), ('n3', ...), ('size_pct', 0.05), ...)`.
* Conserva estos nombres para que el optimizador y el front no fallen. Puedes añadir más, pero sin quitar los básicos.

3. **Sizing**

* Para tamaño fijo por equity:  **`self.size_percent(self.p.size_pct)`** .
* Para dimensionar por stop: **`self.size_percent_by_stop(self.p.size_pct, stop_price[, data=...])`** (antes calcula el stop).
* No reimplementar lógicas de margen/pasos; usa los helpers ya disponibles.

4. **Pivotes / Stops**

* Usa **`Pivot3Candle`** si necesitas últimos máximos/mínimos confirmados; consume **`last_min`** o **`last_max`** como stops.
* Inicializa el indicador en `__init__` y **toma el valor del stop ANTES** de emitir la orden.

5. **Señales y cruces**

* Prefiere **`bt.indicators.CrossOver`** para condiciones de cruce (más robusto que `>` simple).
* Mantén la lógica “entrada/gestión/salida” en `next()` con  **muchos logs** .

6. **Export de indicadores (plotting)**

* Implementa `export_indicators(self) -> dict` devolviendo arrays serializables (listas).
* Nombres sencillos y consistentes (ej.: `SMA1`, `SMA2`, `RSI`, `BB_Upper`, `PIVOT_LAST_MIN`, etc.).
* Sigue el patrón de las estrategias existentes.

7. **Side-effects**

* Nada en el top-level del módulo. No crear archivos ni hacer IO en `import`.
* Toda ejecución real ocurre en `next()` o al invocar explícitamente funciones/`main`. (Ver regla previa de “sin side-effects en import”.)

8. **Multi-asset / Ratio**

* Si operas dos activos con un  **ratio de señales** , adopta el patrón de  **`Bollinger_MultiAsset`** :
  * `datas[0] = ratio (señales)`, `datas[1] = base`, `datas[2] = quote` (o documenta claramente el orden).
  * Maneja **posiciones por-data** con `self.getposition(data)`.
  * Cierra en ambos `data` de manera consistente.

9. **Compatibilidad**

* No cambies nombres de clases ya existentes.
* No elimines métodos esperados por el pipeline (`export_indicators`, params estándar, etc.).
* Mantén comportamiento **determinista** (sin sleeps ni aleatoriedad no controlada).

Notas finales para agentes:

* Si cambias el orden de `datas` en una estrategia multi-asset, **documenta con precisión** en el docstring y en `__init__` (nombres como `self.ratio`, `self.base`, `self.quote`).
* Las estrategias deben ser  **autoexplicativas** : si la lógica no se entiende a primera vista, añade  **comentarios extensos al final de cada función/bloque** .
* Mantén **muchos logs** (con `print` o `logger`) en las fases clave: setup de señal, cálculo de tamaños, decisión de entrada/salida, actualización de stops, reset de flags. Luego se desactivarán si molestan.
