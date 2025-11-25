# ================================================================
# strategies.py — Estrategias Backtrader (refactor suave)
# ================================================================
# Cambios respecto a tu versión anterior:
#   - Eliminada la función sync_multi_asset_data (ya está en DataManager).
#   - Mantenida max_min_search (idéntica, útil para pivotes).
#   - _calc_size se mueve a _BaseLoggedStrategy (reutilizable),
#     sin cambiar la forma de llamarla desde las estrategias.
#   - Lógica de las estrategias: EXACTAMENTE igual.
#   - Comentarios extensos y logs para depurar paso a paso.
#   
# ================================================================

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import backtrader as bt
import math

import math
import backtrader as bt

import math

import math
import backtrader as bt

import math
from typing import Optional, Tuple, Dict

class size_constructor:
    """
    Mixin de utilidades de sizing para Backtrader.
    Depende de:
      - self.broker           → broker de Backtrader (getvalue, getcommissioninfo)
      - self.data             → data principal si no se pasa `data` explícito
      - self.getposition(d)   → posición para un `data` (multi-data). Si no existe, usa self.position

    Métodos:
      - size_margin(data=None, fallback_margin=1.0) -> (leverage, margin)
      - size_fractal(size, data=None, lot_step=None, min_size=0.0) -> size_ajustada
      - size_data(data=None) -> {'price', 'equity', 'current'}
    """

    # ------------------- Helpers de sizing -------------------

    def size_margin(self, data=None, fallback_margin: float = 1.0) -> Tuple[float, float]:
        """
        Devuelve (leverage, margin) a partir del CommInfo del broker para `data`.
        - Si no hay margin definido, usa `fallback_margin` (1.0 por defecto) → leverage=1.0.
        """
        data = data or self.data
        comm_info = self.broker.getcommissioninfo(data)
        # CommInfo estándar expone parámetros en comm_info.p (si existe)
        if hasattr(comm_info, 'p'):
            margin = getattr(comm_info.p, 'margin', None)
        else:
            margin = None

        if margin is None or margin <= 0:
            margin = fallback_margin

        leverage = 1.0 / margin if margin and margin > 0 else 1.0
        return leverage, margin

    def size_fractal(
        self,
        size: float,
        data=None,
        lot_step: Optional[float] = None,
        min_size: float = 0.0
    ) -> float:
        """
        Ajusta `size` al paso mínimo permitido (si existe), permitiendo tamaños < 1.
        - Mantiene fraccionales por defecto (no redondea a entero).
        - Si `lot_step` no se pasa, intenta detectarlo de CommInfo (p.lot_step o p.size_step).
        - `min_size` fuerza un mínimo absoluto si el broker lo requiere.
        """
        data = data or self.data

        # Intentar descubrir un paso de lote desde CommInfo
        if lot_step is None:
            ci = self.broker.getcommissioninfo(data)
            if hasattr(ci, 'p'):
                lot_step = getattr(ci.p, 'lot_step', None) or getattr(ci.p, 'size_step', None)

        s = float(size)

        # Encaje al paso mínimo si existe
        if lot_step and lot_step > 0:
            # número de pasos completos
            steps = math.floor(abs(s) / lot_step)
            units = steps * lot_step
            s = math.copysign(units, s)

        # Enforce tamaño mínimo (si procede)
        if 0.0 < abs(s) < min_size:
            s = math.copysign(min_size, s)

        return s

    def size_data(self, data=None) -> Dict[str, float]:
        """
        Lecturas internas para sizing:
          - price: último close
          - equity: valor total de la cuenta
          - current: tamaño de la posición actual en `data`
        Compatible con multi-data.
        """
        data = data or self.data
        price = float(data.close[0])
        equity = float(self.broker.getvalue())

        # Multi-data: getposition(data) si existe; si no, self.position
        if hasattr(self, 'getposition'):
            pos = self.getposition(data)
        else:
            pos = getattr(self, 'position', None)

        current = float(getattr(pos, 'size', 0.0)) if pos is not None else 0.0
        return {'price': price, 'equity': equity, 'current': current}


class Pivot3Candle(bt.Indicator):
    """
    Pivotes 3 velas (sin look-ahead):
      - Máx local confirmado si: high[-3] < high[-2] > high[-1]
      - Mín local confirmado si:  low [-3] >  low [-2] < low [-1]
    Marca pivotes puntuales y mantiene la "memoria" del último pivote
    en las líneas last_pivot_* y en los atributos last_max/last_min.
    """
    lines = ('pivot_max', 'pivot_min', 'last_pivot_max', 'last_pivot_min')
    plotinfo = dict(subplot=False)
    plotlines = dict(
        pivot_max=dict(_name='Maximo Local', marker='v', markersize=8.0, color='black', fillstyle='full', ls=''),
        pivot_min=dict(_name='Minimo Local', marker='^', markersize=8.0, color='black', fillstyle='full', ls=''),
        last_pivot_max=dict(_name='PIVOT_LAST_MAX', ls='--'),
        last_pivot_min=dict(_name='PIVOT_LAST_MIN', ls='--'),
    )

    def __init__(self):
        self.addminperiod(3)
        # estado persistente (se vuelca en cada vela; no dependemos de [-1])
        self._lpmax = float('nan')
        self._lpmin = float('nan')
        # accesibles desde estrategias
        self.last_max = None
        self.last_min = None

    def next(self):
        
        # 1) detectar pivote central en (-2)
        h3 = float(self.data.high[-3]); h2 = float(self.data.high[-2]); h1 = float(self.data.high[-1])
        l3 = float(self.data.low[-3]);  l2 = float(self.data.low[-2]);  l1 = float(self.data.low[-1])

        # Detectar si hay pivote en esta vela
        has_pivot_max = False
        has_pivot_min = False

        if (h2 > h3) and (h2 > h1):
            self.lines.pivot_max[0] = h2
            self._lpmax = h2  # actualizar memoria
            has_pivot_max = True
        else:
            self.lines.pivot_max[0] = float('nan')

        if (l2 < l3) and (l2 < l1):
            self.lines.pivot_min[0] = l2
            self._lpmin = l2  # actualizar memoria
            has_pivot_min = True
        else:
            self.lines.pivot_min[0] = float('nan')

        # 2) Persistencia condicional para máximos
        if has_pivot_max:
            # Si hay pivote, siempre actualizar
            self._lpmax = h2
        else:
            # Si no hay pivote, actualizar solo si el high actual es mayor que last_max
            current_high = float(self.data.high[0])
            if not math.isnan(self._lpmax):
                if current_high > self._lpmax:
                    self._lpmax = current_high
            else:
                # Si no hay last_max previo, inicializar con el high actual
                self._lpmax = current_high

        # 3) Persistencia condicional para mínimos
        if has_pivot_min:
            # Si hay pivote, siempre actualizar
            self._lpmin = l2
        else:
            # Si no hay pivote, actualizar solo si el low actual es menor que last_min
            current_low = float(self.data.low[0])
            if not math.isnan(self._lpmin):
                if current_low < self._lpmin:
                    self._lpmin = current_low
            else:
                # Si no hay last_min previo, inicializar con el low actual
                self._lpmin = current_low

        # 4) volcar memoria a las líneas de "último pivote" (se plotean siempre)
        self.lines.last_pivot_max[0] = self._lpmax
        self.lines.last_pivot_min[0] = self._lpmin

        # 5) sincronizar atributos para usarlos como stop en estrategias
        if not math.isnan(self._lpmax):
            self.last_max = self._lpmax
        if not math.isnan(self._lpmin):
            self.last_min = self._lpmin

        return self.last_max, self.last_min
        
# ================================================================
# BASE: estrategia con logs y utilidades comunes
# ================================================================
class _BaseLoggedStrategy(bt.Strategy, size_constructor):
    """
    Clase base para estrategias:
      - Registra órdenes y trades cerrados (para depurar/exportar).
      - Expone líneas 'pivot_max' y 'pivot_min' para plotear pivotes.
    """

    def __init__(self):


        # Acumuladores de eventos (para auditoría)
        self._orders_log: List[Dict[str, Any]] = []  # Todas las órdenes completadas
        self._closed_trades: List[Dict[str, Any]] = []  # Trades cerrados reconstruidos

    # Añadir dentro de _BaseLoggedStrategy
    def size_percent(
        self,
        pct: float,
        data=None,
        *,
        lot_step: Optional[float] = None,
        min_size: float = 0.0,
        fallback_margin: float = 1.0
    ) -> float:
        """
        Calcula el nº de unidades para que la posición equivalga a `pct` del equity actual.
        - No abre/cierra posiciones.
        - No usa la posición actual ni aplica dirección.
        - Devuelve SIEMPRE un tamaño positivo (float) que luego usarás en buy()/sell().
        Ej.: pct=0.05 → 5% del equity. Si pct<0, se toma abs(pct).

        Params:
        pct: fracción del equity objetivo (0.05 = 5%).
        data: feed (por defecto self.data).
        lot_step: paso mínimo de lote; si None intenta detectarlo desde CommInfo.
        min_size: tamaño mínimo absoluto (si el broker lo exige).
        fallback_margin: margen por defecto si CommInfo no define margin.

        Return:
        float >= 0.0 con el tamaño (unidades).
        """
        data = data or self.data

        # Lecturas internas
        sd = self.size_data(data)  # {'price','equity','current'} -> aquí ignoramos 'current'
        price = sd['price']
        equity = sd['equity']

        if price <= 0.0 or equity <= 0.0 or pct == 0.0:
            return 0.0

        # Apalancamiento efectivo del broker
        leverage, _ = self.size_margin(data, fallback_margin=fallback_margin)

        # Tamaño bruto (sin dirección): valor objetivo / precio
        target_value = equity * abs(pct) * leverage
        size = target_value / price

        # Encajar a pasos y mínimos permitidos (tamaños fraccionales OK)
        size = self.size_fractal(size, data=data, lot_step=lot_step, min_size=min_size)

        return max(0.0, float(size))

    def size_percent_by_stop(
        self,
        risk_pct: float,
        stop_price: float,
        *,
        data=None,
        cushion: float = 0.0,
        mult: float = 1.0,
        pct_cap: float = 0.50
    ) -> float:
        """
        Dimensiona por riesgo fijo: el riesgo debe ser risk_pct del poder de compra total.
        Riesgo = (precio - stop) × size = equity × risk_pct × leverage
        """
        data = data or self.data
        sd = self.size_data(data)
        price = sd['price']
        equity = sd['equity']
        
        if risk_pct <= 0 or stop_price is None or price <= 0 or equity <= 0:
            return 0.0

        # Distancia efectiva al stop
        D = abs(price - float(stop_price))
        if D <= 0:
            return 0.0
        if cushion > 0:
            D *= (1.0 + float(cushion))

        # Obtener leverage (ANTES de usarlo)
        leverage, _ = self.size_margin(data)

        # Riesgo fijo basado en poder de compra total (equity × leverage)
        # Riesgo objetivo = equity × risk_pct × leverage
        # size = riesgo_objetivo / distancia_stop
        riesgo_dolares = equity * abs(risk_pct) * leverage
        size = riesgo_dolares / (D * float(mult))

        # Ajustar a pasos y mínimos permitidos
        size = self.size_fractal(size, data=data, lot_step=None, min_size=0.0)
        
        # Límite superior por prudencia
        max_size_by_cap = (equity * float(pct_cap) * leverage) / price if leverage > 0 else 0.0
        if max_size_by_cap > 0:
            size = min(size, max_size_by_cap)
        
        return max(0.0, float(size))

    def _export_indicators_helper(self, *indicators, names=None):
        """
        Helper para exportar indicadores de Backtrader a diccionario.
        Convierte las líneas de indicadores en listas serializables para plotting/export.

        Uso en estrategias:
          def export_indicators(self):
              return self._export_indicators_helper(
                  self.bb.upperband, self.bb.middleband, self.bb.lowerband,
                  names=['BB_Upper','BB_Middle','BB_Lower']
              )

        Params:
          *indicators: indicadores/líneas de Backtrader a exportar.
          names: lista opcional de nombres (si no se proporciona, usa L1, L2, ...).

        Returns:
          dict con los indicadores exportados como listas.
        """
        indicators_dict = {}
        
        for i, ind in enumerate(indicators, 1):
            name = names[i-1] if names and i-1 < len(names) else f"L{i}"
            
            if hasattr(ind, 'array') and hasattr(ind, '__len__') and len(ind):
                indicators_dict[name] = list(ind.array)
                print(f"[EXPORT] {name}: {len(indicators_dict[name])} puntos")
            else:
                print(f"[EXPORT][SKIP] {name}: sin datos o no es línea Backtrader")
        
        print(f"[EXPORT] Series: {list(indicators_dict.keys())}")
        return indicators_dict

    # ---------------- Notificaciones: órdenes y trades ------------
    def notify_order(self, order):
        """
        Guarda una traza detallada de las órdenes COMPLETED para reconstruir trades.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self._orders_log.append({
                'dt': order.data.datetime.datetime(0),
                'size': order.executed.size,
                'price': order.executed.price,
                'value': order.executed.value,
                'comm': order.executed.comm,
                'is_buy': order.isbuy(),
                'ref': order.ref,
                'data_name': getattr(order.data, '_name', 'data0'),
            })
            return

        # Canceled / Margin / Rejected -> no guardamos más info (pero podrías loguear si lo deseas)

    def notify_trade(self, trade):
        """
        Reconstruye y guarda un trade CERRADO basándose en el log de órdenes
        realizadas EN EL PERIODO del trade y sólo del feed correspondiente.
        """
        if not trade.isclosed:
            return

        open_dt = bt.num2date(trade.dtopen)
        close_dt = bt.num2date(trade.dtclose)
        data_name = getattr(trade.data, '_name', None)

        # Filtra por rango temporal y por activo
        related_orders = [
            o for o in self._orders_log
            if open_dt <= o['dt'] <= close_dt and (data_name is None or o.get('data_name') == data_name)
        ]

        if not related_orders:
            # Fallback: sin matching de órdenes, usamos datos del objeto trade (como ya hacías)
            self._closed_trades.append({
                'open_datetime': open_dt,
                'close_datetime': close_dt,
                'size': 0.0,
                'price_open': trade.price,
                'price_close': trade.data.close[0],
                'pnl': float(trade.pnl),
                'pnl_comm': float(trade.pnlcomm),
                'barlen': int(trade.barlen),
            })
            return

        # Agrupar por fecha (abre/cierra puede estar compuesto de varias órdenes)
        from collections import defaultdict
        by_dt = defaultdict(list)
        for o in related_orders:
            by_dt[o['dt']].append(o)

        dts = sorted(by_dt.keys())
        open_date = dts[0]
        close_date = dts[-1]

        # Precio medio ponderado en la apertura
        open_orders = by_dt[open_date]
        
        # Filtrar órdenes de apertura: cuando hay close() seguido de buy() en la misma barra,
        # necesitamos identificar solo las órdenes de apertura reales del trade
        # Calculamos el tamaño neto para determinar la dirección del trade
        net_size = sum(o['size'] for o in open_orders)
        
        # Filtrar según la dirección del trade:
        # - Si net_size > 0: trade largo, solo órdenes de compra (is_buy=True, size>0)
        # - Si net_size < 0: trade corto, solo órdenes de venta (is_buy=False, size<0)
        if net_size > 0:
            # Trade largo: solo órdenes de compra (excluye órdenes de cierre de venta)
            opening_orders = [o for o in open_orders if o['is_buy'] and o['size'] > 0]
        elif net_size < 0:
            # Trade corto: solo órdenes de venta (excluye órdenes de cierre de compra)
            opening_orders = [o for o in open_orders if not o['is_buy'] and o['size'] < 0]
        else:
            # Si net_size es 0, usar todas las órdenes (fallback)
            opening_orders = open_orders
        
        # Si no encontramos órdenes de apertura filtradas, usar todas (fallback)
        if not opening_orders:
            opening_orders = open_orders
        
        # Si hay múltiples órdenes del mismo tipo en la misma fecha,
        # tomar solo la última (la que realmente abrió el trade)
        # Esto ocurre cuando close() y buy()/sell() se ejecutan en la misma barra
        if len(opening_orders) > 1:
            signs = set(1 if o['size'] > 0 else -1 for o in opening_orders)
            if len(signs) == 1:
                # Todas son del mismo tipo: tomar solo la última orden
                # (la última en ejecutarse es la que abrió el trade)
                opening_orders = [opening_orders[-1]]
        
        # Calcular tamaño desde las órdenes de apertura filtradas
        size_open = sum(o['size'] for o in opening_orders)
        
        total_val_open = sum(abs(o['size']) * o['price'] for o in opening_orders)
        total_size_open = sum(abs(o['size']) for o in opening_orders)
        price_open = total_val_open / total_size_open if total_size_open > 0 else opening_orders[0]['price']

        # Precio de cierre: de la última orden
        close_orders = by_dt[close_date]
        price_close = close_orders[-1]['price']

        self._closed_trades.append({
            'open_datetime': open_date,
            'close_datetime': close_date,
            'size': float(size_open),  # Usar tamaño calculado desde órdenes de apertura filtradas
            'price_open': float(price_open),
            'price_close': float(price_close),
            'pnl': float(trade.pnl),
            'pnl_comm': float(trade.pnlcomm),
            'barlen': int(trade.barlen),
        })

# ================================================================
# ESTRATEGIAS (lógica original conservada)
# ================================================================

class DosMedias(_BaseLoggedStrategy):
    """
    Estrategia “siempre en mercado” basada en cruce de dos SMAs.
      - n1: SMA rápida
      - n2: SMA lenta
      - n3: no usado
      - size_pct: % de equity por operación 
    """
    params = (('n1', 7), ('n2', 20), ('n3', None), ('size_pct', 0.05))

    def __init__(self):
        # Añadir esta condición en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        self.sma_fast = bt.talib.SMA(self.data.close, timeperiod=self.p.n1)
        self.sma_slow = bt.talib.SMA(self.data.close, timeperiod=self.p.n2)
        self.xover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        # Cruce alcista: media rápida cruza por encima → Largo
        if self.xover[0] > 0:
            self.close()
            self.buy(size=self.size_percent(self.p.size_pct))
        
        # Cruce bajista: media rápida cruza por debajo → Corto
        elif self.xover[0] < 0:
            self.close()
            self.sell(size=self.size_percent(self.p.size_pct))

    def export_indicators(self):
        return self._export_indicators_helper(
            self.sma_fast, self.sma_slow,
            names=['SMA_Fast', 'SMA_Slow']
        )

########################################################################################################################

# ==========================================================================
# ESTRATEGIAS DE EVALUACION EN MULTIACTIVO BASADAS EN ESTRATEGIAS INICIALES 
# ==========================================================================
# Idea general:
#   - StrategyMultiAsset es una estrategia genérica que ENVUELVE a una
#     estrategia "single asset" ya existente (inner_cls).
#   - Aplica SU __init__ y SU next() a cada data por separado mediante
#     un contexto interno (_InnerContext), pero usando SIEMPRE el mismo
#     broker/capital de esta clase.
#   - Resultado: una sola cuenta / curva de equity, con todas las
#     posiciones de todos los activos sumadas vela a vela (cartera).
#   - Usar make_multi(strategy_cls) para convertir una estrategia single-asset en multi-asset.
#     si pasamos los activos solo en data01 evalua activo por activo, sin evaluar la cartera completa.
# ================================================================

class StrategyMultiAsset(_BaseLoggedStrategy):
    """
    Wrapper genérico multi-activo.
    ...
    """
    
    # Atributo de clase para parámetros por activo
    _params_by_asset = None  # Dict[str, Dict[str, Any]] mapea asset_name -> params

    def __init__(self):
        """
        Constructor:
          - Inicializa la parte base (_BaseLoggedStrategy).
          - Crea un contexto interno por cada data.
          - Aplica inner_cls.__init__(ctx) sobre cada contexto (indicadores).
        """
        super().__init__()

        if self.inner_cls is None:
            raise ValueError("Debes definir inner_cls en la subclase de StrategyMultiAsset")

        # Lista de contextos, uno por cada data
        self._contexts: List["StrategyMultiAsset._InnerContext"] = []

        print("================================================================")
        print(f"[INIT] StrategyMultiAsset base={self.inner_cls.__name__}")
        print(f"       Nº de datas detectados={len(self.datas)}")
        print("================================================================")

        # Crear un contexto interno para cada data
        for idx, data in enumerate(self.datas):
            data_name = getattr(data, "_name", f"data{idx}")
            print(f"[INIT]  -> Creando contexto interno para data[{idx}] = {data_name}")

            # Obtener asset_name desde el DataFrame original
            asset_name = None
            if hasattr(data, 'p') and hasattr(data.p, 'dataname'):
                df = data.p.dataname
                asset_name = getattr(df, '_asset_name', None)
            
            # Si no se encontró, intentar desde el nombre del datafeed
            if not asset_name:
                asset_name = getattr(data, '_asset_name', data_name)

            ctx = StrategyMultiAsset._InnerContext(
                outer=self,
                data=data,
                data_index=idx,
                asset_name=asset_name,
                params_by_asset=self._params_by_asset
            )

            # MUY IMPORTANTE:
            #   Aquí reutilizamos __init__ de la estrategia original (inner_cls),
            #   pero pasando ctx como "self". Esto crea los indicadores en ctx
            #   igual que si la estrategia operara sobre un solo activo.
            ctx._is_multi_ctx = True
            self.inner_cls.__init__(ctx)

            self._contexts.append(ctx)

        # Guardamos puntero al next "desvinculado" de inner_cls
        self._inner_next = self.inner_cls.next

    def next(self):
        """
        Método que Backtrader llama en cada nueva vela.

        Aquí:
          - Obtenemos la fecha global (primer data) solo para logs.
          - Recorremos todos los contextos.
          - Ejecutamos inner_cls.next(ctx) para cada activo.
        """
        if self.datas:
            dt = self.datas[0].datetime.datetime(0)
        else:
            dt = None

        #print("----------------------------------------------------------------")
        #print(f"[MULTI NEXT] {dt} | n_datas={len(self.datas)} "
        #      f"| estrategia_base={self.inner_cls.__name__}")
        #print("----------------------------------------------------------------")

        for ctx in self._contexts:
            data_name = getattr(ctx.data, "_name", f"data{ctx.data_index}")
            #print(f"[MULTI NEXT] Ejecutando lógica base en '{data_name}'")
            # inner_cls.next(ctx) -> ctx hace de “self” para la estrategia base
            self._inner_next(ctx)

    def export_indicators(self):
        """
        Exporta indicadores.

        Por simplicidad y compatibilidad con tu pipeline actual, devolvemos
        los indicadores del PRIMER contexto (primer activo), utilizando el
        método export_indicators(self) de inner_cls si existe.
        """
        if not self._contexts:
            print("[MULTI EXPORT] Sin contextos internos, no hay indicadores")
            return {}

        ctx0 = self._contexts[0]

        if hasattr(self.inner_cls, "export_indicators"):
            print("[MULTI EXPORT] Llamando a export_indicators() del primer contexto")
            return self.inner_cls.export_indicators(ctx0)

        print("[MULTI EXPORT] inner_cls no implementa export_indicators()")
        return {}

    # ------------------------------------------------------------
    # Contexto interno por data (un activo)
    # ------------------------------------------------------------
    class _InnerContext:
        """
        Contexto interno que simula ser una instancia de inner_cls para
        UN solo activo.
        """

        def __init__(self, outer: "_BaseLoggedStrategy", data, data_index: int, 
                    asset_name: str = None, params_by_asset: dict = None):
            self._outer = outer
            self.data = data
            self.datas = [data]
            self.data_index = data_index
            self._asset_name = asset_name

            # Compartimos broker con la estrategia externa
            self.broker = outer.broker

            # Calcular número total de activos para rebalanceo equitativo
            num_assets = len(outer.datas) if hasattr(outer, 'datas') else 1

            # Parámetros: usar específicos del activo si están disponibles, sino usar los de outer
            if params_by_asset and asset_name and asset_name in params_by_asset:
                # Crear objeto Params con los parámetros específicos del activo
                asset_params = params_by_asset[asset_name]
                # Combinar con parámetros base de outer.p
                combined_params = {}
                # Primero copiar todos los parámetros base
                for key in dir(outer.p):
                    if not key.startswith('_'):
                        try:
                            combined_params[key] = getattr(outer.p, key)
                        except:
                            pass
                # Luego sobrescribir con parámetros específicos del activo
                combined_params.update(asset_params)
                
                # Rebalancear size_pct equitativamente: dividir por número de activos
                if num_assets > 1 and 'size_pct' in combined_params:
                    base_size_pct = combined_params['size_pct']
                    rebalanced_size_pct = base_size_pct / num_assets
                    combined_params['size_pct'] = rebalanced_size_pct
                    print(f"[CTX INIT] size_pct rebalanceado para {asset_name}: {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos)")
                elif num_assets > 1:
                    # Si no hay size_pct en asset_params, usar el de outer.p
                    base_size_pct = getattr(outer.p, 'size_pct', 0.05)
                    rebalanced_size_pct = base_size_pct / num_assets
                    combined_params['size_pct'] = rebalanced_size_pct
                    print(f"[CTX INIT] size_pct rebalanceado para {asset_name}: {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos, usando base)")
                
                # Crear objeto Params personalizado
                class AssetParams:
                    def __init__(self, params_dict):
                        for k, v in params_dict.items():
                            setattr(self, k, v)
                
                self.p = AssetParams(combined_params)
                print(f"[CTX INIT] Parámetros específicos para {asset_name}: {asset_params}")
            else:
                # Usar parámetros compartidos (comportamiento original)
                self.p = outer.p
                
                # Aplicar rebalanceo equitativo incluso si no hay params_by_asset
                if num_assets > 1 and hasattr(self.p, 'size_pct'):
                    base_size_pct = self.p.size_pct
                    rebalanced_size_pct = base_size_pct / num_assets
                    # Crear copia modificada de los params
                    class RebalancedParams:
                        def __init__(self, base_p, size_pct_val):
                            for key in dir(base_p):
                                if not key.startswith('_'):
                                    try:
                                        setattr(self, key, getattr(base_p, key))
                                    except:
                                        pass
                            self.size_pct = size_pct_val
                    self.p = RebalancedParams(outer.p, rebalanced_size_pct)
                    print(f"[CTX INIT] size_pct rebalanceado (sin params_by_asset): {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos)")

        # ------------------- Estado de la posición -------------------

        @property
        def position(self):
            """
            Devuelve la posición ESPECÍFICA de este activo usando el broker
            de la estrategia externa (outer).
            """
            pos = self._outer.getposition(self.data)
            size = float(getattr(pos, "size", 0.0)) if pos is not None else 0.0
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX POS] data={data_name} idx={self.data_index} pos.size={size}")
            return pos

        # ------------------- Helpers de tamaño -------------------

        def size_percent(self, pct: float, *args, **kwargs) -> float:
            """
            Delegamos en size_percent de outer, fijando data=este activo.
            """
            size = self._outer.size_percent(pct, data=self.data, *args, **kwargs)
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX SIZE] data={data_name} pct={pct} -> size={size}")
            return size

        def size_percent_by_stop(self, risk_pct: float, stop_price: float, *args, **kwargs) -> float:
            """
            Delegamos en size_percent_by_stop de outer, fijando data=este activo.
            """
            size = self._outer.size_percent_by_stop(
                risk_pct,
                stop_price,
                data=self.data,
                *args,
                **kwargs,
            )
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            print(
                f"[CTX SIZE STOP] data={data_name} "
                f"risk_pct={risk_pct} stop={stop_price} -> size={size}"
            )
            return size

        # -------------------------- Órdenes ------------------------------

        def buy(self, *args, **kwargs):
            """
            Lanza una orden de compra en ESTE activo, delegando en outer.buy().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX BUY] data={data_name}")
            return self._outer.buy(data=self.data, *args, **kwargs)

        def sell(self, *args, **kwargs):
            """
            Lanza una orden de venta (corto) en ESTE activo, delegando en outer.sell().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX SELL] data={data_name}")
            return self._outer.sell(data=self.data, *args, **kwargs)

        def close(self, *args, **kwargs):
            """
            Cierra la posición de ESTE activo, delegando en outer.close().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX CLOSE] data={data_name}")
            return self._outer.close(data=self.data, *args, **kwargs)

        # ------------------ Utilidades auxiliares ------------------------

        def _export_indicators_helper(self, *indicators, names=None):
            """
            Permite que inner_cls.export_indicators(ctx) reutilice el helper
            de la estrategia externa para exportar indicadores.
            """
            return self._outer._export_indicators_helper(*indicators, names=names)

        def notify_order(self, order):
            """
            Si inner_cls llega a llamar a notify_order, reenviamos a outer.
            """
            return self._outer.notify_order(order)

        def notify_trade(self, trade):
            """
            Si inner_cls llega a llamar a notify_trade, reenviamos a outer.
            """
            return self._outer.notify_trade(trade)


def make_multi(strategy_cls):
    """
    Envuelve una estrategia 'single-asset' en versión multi-activo (StrategyMultiAsset),
    preservando correctamente los params aunque vengan como objeto Backtrader Params.
    """
    if issubclass(strategy_cls, StrategyMultiAsset):
        return strategy_cls  # ya es multi

    def _extract_bt_params(pobj):
        # Intenta extraer nombres/valores de un objeto Params de Backtrader
        try:
            keys = pobj._getkeys()
            return tuple((k, getattr(pobj, k)) for k in keys)
        except Exception:
            return ()

    base_params = getattr(strategy_cls, 'params', ())
    if isinstance(base_params, (tuple, list)):
        try:
            wrapper_params = tuple(tuple(p) for p in base_params)
        except TypeError:
            wrapper_params = ()
    else:
        wrapper_params = _extract_bt_params(base_params)

    class _AutoMulti(StrategyMultiAsset):
        inner_cls = strategy_cls
        params = wrapper_params

    _AutoMulti.__name__ = f"{strategy_cls.__name__}"
    return _AutoMulti

  
