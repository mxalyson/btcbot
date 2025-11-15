"""
🎯 TARGET ENGINEERING PARA SCALPING - V2.0

Multi-horizon targets otimizados para scalping 15min.
Define exatamente o que o modelo deve aprender.

Author: Claude
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TargetEngineer:
    """
    Cria targets inteligentes para scalping.

    Estratégias:
    1. Multi-horizon classification (3/5/10 bars)
    2. Regression target (expected return)
    3. Risk-adjusted target (return/risk ratio)
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configurações de target
                {
                    'tp_pct': 0.003,  # 0.3% TP
                    'sl_pct': 0.003,  # 0.3% SL
                    'horizons': [3, 5, 10],  # Bars para avaliar
                    'fees': 0.0006,  # 0.06% per side
                    'slippage': 0.0001,  # 0.01%
                }
        """
        self.config = config or {}
        self.tp_pct = self.config.get('tp_pct', 0.003)  # 0.3%
        self.sl_pct = self.config.get('sl_pct', 0.003)  # 0.3%
        self.horizons = self.config.get('horizons', [3, 5, 10])
        self.fees = self.config.get('fees', 0.0006)  # 0.06%
        self.slippage = self.config.get('slippage', 0.0001)  # 0.01%

        # Cost total (entrada + saída)
        self.total_cost = (self.fees * 2) + (self.slippage * 2)

    def create_master_scalper_target(self, df: pd.DataFrame, atr_col='atr') -> pd.DataFrame:
        """
        Target do MODELO ANTIGO (que funcionava!)

        Sistema de VOTAÇÃO multi-horizon com threshold DINÂMICO baseado em ATR.

        - Horizons: 4, 6, 8 bars (1h, 1.5h, 2h)
        - Threshold dinâmico: 0.35% a 0.75% baseado em ATR
        - Votação: 2/3 horizontes devem concordar
        - Remove zona neutra (só sinais fortes!)
        - Retorna: 1 (UP) ou 0 (DOWN) - BINÁRIO
        """
        df_target = df.copy()

        # ATR-based dynamic threshold
        atr = df_target[atr_col] if atr_col in df_target.columns else df_target['close'] * 0.005
        atr_pct = (atr / df_target['close']) * 100

        # Dynamic threshold: 0.35% a 0.75% baseado em volatilidade
        dynamic_threshold = np.clip(atr_pct * 0.3, 0.35, 0.75)

        # Multi-horizon voting
        horizons = [4, 6, 8]  # 1h, 1.5h, 2h
        votes = []

        for horizon in horizons:
            # Retorno futuro em %
            future_returns = (df_target['close'].shift(-horizon) / df_target['close'] - 1) * 100

            # Voto: 1.0 (UP), 0.5 (NEUTRAL), 0.0 (DOWN)
            vote = pd.Series(0.5, index=df_target.index)
            vote[future_returns > dynamic_threshold] = 1.0
            vote[future_returns < -dynamic_threshold] = 0.0

            votes.append(vote)

        # Média dos votos
        avg_vote = sum(votes) / len(votes)

        # Target final: consenso forte necessário
        target = pd.Series(np.nan, index=df_target.index)
        target[avg_vote > 0.65] = 1  # UP: 2/3 concordam
        target[avg_vote < 0.35] = 0  # DOWN: 2/3 concordam
        # 0.35 <= avg_vote <= 0.65 = NEUTRAL (removido = np.nan)

        df_target['target_master'] = target
        df_target['vote_confidence'] = np.abs(avg_vote - 0.5) * 2

        return df_target

    def create_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria TODOS os targets para o modelo.

        Returns:
            DataFrame com targets adicionados
        """
        print("🎯 Creating scalping targets...")

        df_target = df.copy()

        # MASTER SCALPER target (modelo antigo que funciona!)
        df_target = self.create_master_scalper_target(df_target)
        print("  ✅ Master Scalper target (voting + dynamic threshold)")

        # 1. Multi-horizon binary targets
        df_target = self.create_multihorizon_targets(df_target)
        print("  ✅ Multi-horizon targets")

        # 2. Regression target (expected return)
        df_target = self.create_regression_target(df_target)
        print("  ✅ Regression target")

        # 3. Risk-adjusted target
        df_target = self.create_risk_adjusted_target(df_target)
        print("  ✅ Risk-adjusted target")

        # 4. Best horizon target (meta)
        df_target = self.create_best_horizon_target(df_target)
        print("  ✅ Best horizon meta-target")

        return df_target

    def create_multihorizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria targets binários para múltiplos horizontes.

        Target = 1 se:
        - Atinge TP (0.3%) dentro do horizonte
        - SEM atingir SL (-0.3%) antes

        Isso é CRÍTICO para scalping: queremos movimentos rápidos e limpos!
        """
        for horizon in self.horizons:
            target_name = f'target_{horizon}bars'

            # Calcula máximo e mínimo nos próximos N bars
            future_high = df['high'].shift(-1).rolling(horizon).max()
            future_low = df['low'].shift(-1).rolling(horizon).min()

            # Preço de entrada (close atual)
            entry = df['close']

            # TP e SL levels
            tp_long = entry * (1 + self.tp_pct)
            sl_long = entry * (1 - self.sl_pct)

            tp_short = entry * (1 - self.tp_pct)
            sl_short = entry * (1 + self.sl_pct)

            # LONG: TP hit sem SL hit
            long_tp_hit = future_high >= tp_long
            long_sl_hit = future_low <= sl_long
            target_long = (long_tp_hit & ~long_sl_hit).astype(int)

            # SHORT: TP hit sem SL hit
            short_tp_hit = future_low <= tp_short
            short_sl_hit = future_high >= sl_short
            target_short = (short_tp_hit & ~short_sl_hit).astype(int)

            # Combina: 1 = LONG, -1 = SHORT, 0 = NO TRADE
            df[target_name] = target_long - target_short

            # Adiciona flag de viabilidade (considera custos)
            net_return_long = (self.tp_pct - self.total_cost)
            net_return_short = (self.tp_pct - self.total_cost)

            df[f'{target_name}_viable'] = (
                ((target_long == 1) & (net_return_long > 0)) |
                ((target_short == 1) & (net_return_short > 0))
            ).astype(int)

        return df

    def create_regression_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target de regressão: retorno esperado nos próximos N bars.

        Isso permite que o modelo preveja a MAGNITUDE do movimento,
        não apenas a direção!
        """
        for horizon in self.horizons:
            target_name = f'return_{horizon}bars'

            # Retorno simples
            future_return = (
                (df['close'].shift(-horizon) - df['close']) / df['close']
            ) * 100  # Em percentual

            df[target_name] = future_return

            # Retorno ajustado por custos
            df[f'{target_name}_net'] = future_return - (self.total_cost * 100)

        return df

    def create_risk_adjusted_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target ajustado por risco (Sharpe-like).

        return / volatility = qualidade do movimento
        """
        for horizon in self.horizons:
            target_name = f'sharpe_{horizon}bars'

            # Retorno
            future_return = df[f'return_{horizon}bars']

            # Volatilidade no horizonte
            future_vol = (
                df['close'].pct_change().shift(-1).rolling(horizon).std() * 100
            )

            # Sharpe-like ratio
            df[target_name] = future_return / (future_vol + 1e-10)

        return df

    def create_best_horizon_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Meta-target: qual horizonte tem melhor risco/retorno?

        Útil para ensemble ou para escolher qual modelo usar.
        """
        # Coleta todos os targets multi-horizon
        target_cols = [f'target_{h}bars' for h in self.horizons]

        # Encontra qual horizonte tem maior sucesso
        target_matrix = df[target_cols].values

        # Best horizon = primeiro a dar sinal positivo
        best_horizon_idx = []
        for row in target_matrix:
            # Procura primeiro 1 ou -1
            non_zero = np.where(row != 0)[0]
            if len(non_zero) > 0:
                best_horizon_idx.append(non_zero[0])
            else:
                best_horizon_idx.append(-1)  # Nenhum sinal

        df['best_horizon'] = best_horizon_idx

        # Mapeia para valor do horizonte
        df['best_horizon_bars'] = df['best_horizon'].map(
            {i: h for i, h in enumerate(self.horizons)}
        )

        return df

    def analyze_targets(self, df: pd.DataFrame) -> Dict:
        """
        Analisa distribuição dos targets.

        Isso é CRUCIAL para entender se o target é viável!
        """
        stats = {}

        print("\n" + "="*70)
        print("📊 TARGET ANALYSIS")
        print("="*70)

        for horizon in self.horizons:
            target_col = f'target_{horizon}bars'

            if target_col not in df.columns:
                continue

            target_data = df[target_col].dropna()

            # Contagens
            long_count = (target_data == 1).sum()
            short_count = (target_data == -1).sum()
            neutral_count = (target_data == 0).sum()
            total = len(target_data)

            # Percentuais
            long_pct = (long_count / total) * 100
            short_pct = (short_count / total) * 100
            neutral_pct = (neutral_count / total) * 100
            signal_pct = long_pct + short_pct

            stats[f'{horizon}bars'] = {
                'long_count': long_count,
                'short_count': short_count,
                'neutral_count': neutral_count,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'signal_pct': signal_pct,
                'balance': long_count / (short_count + 1)
            }

            print(f"\n🎯 Target {horizon} bars:")
            print(f"   LONG:    {long_count:5d} ({long_pct:5.1f}%)")
            print(f"   SHORT:   {short_count:5d} ({short_pct:5.1f}%)")
            print(f"   NEUTRAL: {neutral_count:5d} ({neutral_pct:5.1f}%)")
            print(f"   SIGNAL:  {long_count + short_count:5d} ({signal_pct:5.1f}%)")
            print(f"   Balance: {long_count/(short_count+1):.2f} (ideal ~1.0)")

            # ⚠️ Avisos
            if signal_pct < 5:
                print(f"   ⚠️  WARNING: Muito pouco sinal ({signal_pct:.1f}%)")
            if signal_pct > 50:
                print(f"   ⚠️  WARNING: Muito sinal ({signal_pct:.1f}%) - talvez TP/SL muito largos")

            balance = long_count / (short_count + 1)
            if balance < 0.7 or balance > 1.3:
                print(f"   ⚠️  WARNING: Desbalanceado (ratio {balance:.2f})")

        print("\n" + "="*70)

        return stats

    def create_labels_for_training(self, df: pd.DataFrame,
                                   target_type: str = 'classification',
                                   horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features (X) e target (y) para treinamento.

        Args:
            target_type: 'master', 'classification', 'regression', ou 'risk_adjusted'
            horizon: Qual horizonte usar (3, 5, ou 10 bars) - ignored for 'master'

        Returns:
            X, y prontos para sklearn/lightgbm
        """
        df_clean = df.copy()

        # Escolhe target
        if target_type == 'master':
            # MASTER SCALPER target (modelo antigo que funciona!)
            target_col = 'target_master'
            # Binário: 0 (DOWN) ou 1 (UP)
            y = df_clean[target_col].astype(int)

        elif target_type == 'classification':
            target_col = f'target_{horizon}bars'
            # Converte -1, 0, 1 para 0, 1, 2 (LightGBM precisa >= 0)
            y = df_clean[target_col].map({-1: 0, 0: 1, 1: 2})

        elif target_type == 'regression':
            target_col = f'return_{horizon}bars_net'
            y = df_clean[target_col]

        elif target_type == 'risk_adjusted':
            target_col = f'sharpe_{horizon}bars'
            y = df_clean[target_col]

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # Remove linhas com NaN no target
        valid_idx = y.notna()
        df_clean = df_clean[valid_idx]
        y = y[valid_idx]

        # Remove colunas de target e datetime
        feature_cols = [c for c in df_clean.columns
                       if not c.startswith('target_')
                       and not c.startswith('return_')
                       and not c.startswith('sharpe_')
                       and not c.startswith('best_horizon')
                       and c not in ['open', 'high', 'low', 'close', 'volume']]

        X = df_clean[feature_cols]

        # Remove colunas não-numéricas (LightGBM não aceita 'object')
        numeric_cols = X.select_dtypes(include=['int', 'int64', 'float', 'float64', 'bool']).columns
        non_numeric = set(X.columns) - set(numeric_cols)
        if len(non_numeric) > 0:
            print(f"   ⚠️  Removing {len(non_numeric)} non-numeric columns: {list(non_numeric)}")
        X = X[numeric_cols]

        print(f"\n✅ Training data prepared:")
        print(f"   Target: {target_col} ({target_type})")
        print(f"   Samples: {len(X):,}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target distribution:")
        print(f"{y.value_counts().sort_index()}")

        return X, y


def optimize_target_params(df: pd.DataFrame,
                          tp_range=(0.002, 0.005),
                          sl_range=(0.002, 0.005),
                          horizon=5) -> Dict:
    """
    Encontra melhores parâmetros de TP/SL para o target.

    Testa diferentes combinações e retorna a que maximiza signal rate
    mantendo boa relação risk/reward.
    """
    print("\n🔍 Optimizing target parameters...")

    best_config = None
    best_score = 0

    tp_values = np.arange(tp_range[0], tp_range[1], 0.0005)
    sl_values = np.arange(sl_range[0], sl_range[1], 0.0005)

    results = []

    for tp in tp_values:
        for sl in sl_values:
            # Cria target com esses params
            engineer = TargetEngineer({
                'tp_pct': tp,
                'sl_pct': sl,
                'horizons': [horizon]
            })

            df_test = engineer.create_multihorizon_targets(df)
            target_col = f'target_{horizon}bars'

            # Analisa
            target_data = df_test[target_col].dropna()
            long_count = (target_data == 1).sum()
            short_count = (target_data == -1).sum()
            signal_count = long_count + short_count
            total = len(target_data)

            signal_rate = signal_count / total
            win_balance = long_count / (short_count + 1)

            # Score = signal_rate * balance_quality * risk_reward
            balance_quality = 1 - abs(win_balance - 1.0)  # Penaliza desbalanceamento
            risk_reward = tp / sl

            score = signal_rate * balance_quality * (risk_reward / 2)

            results.append({
                'tp': tp,
                'sl': sl,
                'signal_rate': signal_rate,
                'win_balance': win_balance,
                'risk_reward': risk_reward,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_config = {'tp_pct': tp, 'sl_pct': sl}

    # Mostra top 5
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)

    print("\n🏆 Top 5 configurations:")
    print(results_df.head().to_string(index=False))

    print(f"\n✅ Best config:")
    print(f"   TP: {best_config['tp_pct']:.4f} ({best_config['tp_pct']*100:.2f}%)")
    print(f"   SL: {best_config['sl_pct']:.4f} ({best_config['sl_pct']*100:.2f}%)")
    print(f"   Score: {best_score:.4f}")

    return best_config


if __name__ == "__main__":
    """
    Teste do target engineering.
    """
    # Dados de exemplo
    dates = pd.date_range('2024-01-01', periods=5000, freq='15min')
    np.random.seed(42)

    # Simula price action realista
    returns = np.random.normal(0.0001, 0.002, 5000)
    close_prices = 100 * (1 + returns).cumprod()

    df_test = pd.DataFrame({
        'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, 5000)),
        'high': close_prices * (1 + np.random.uniform(0, 0.005, 5000)),
        'low': close_prices * (1 + np.random.uniform(-0.005, 0, 5000)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 5000),
    }, index=dates)

    # Cria targets
    print("=" * 70)
    print("🧪 TESTING TARGET ENGINEERING")
    print("=" * 70)

    engineer = TargetEngineer({
        'tp_pct': 0.003,
        'sl_pct': 0.003,
        'horizons': [3, 5, 10],
        'fees': 0.0006,
        'slippage': 0.0001
    })

    df_with_targets = engineer.create_all_targets(df_test)

    # Analisa
    stats = engineer.analyze_targets(df_with_targets)

    # Prepara para treinamento
    X, y = engineer.create_labels_for_training(
        df_with_targets,
        target_type='classification',
        horizon=5
    )

    print(f"\n✅ Target engineering test complete!")
    print(f"📊 Data shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
