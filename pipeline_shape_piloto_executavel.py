"""
Pipeline parametrizada para curva piloto de shape
Projeto: Otimização Operacional e Simulador de Blocos e Sublocos

Entrada esperada (Base Demanda):
- setor
- ciclo
- data_captacao
- data_abertura_ciclo
- data_fechamento_ciclo
- bloco
- regiao
- volume (ou quantidade de pedidos)

Observação:
A data de abertura/fechamento usada para a série temporal deve ser a do CICLO,
não uma data dependente do bloco do setor. Bloco é usado apenas como nível
hierárquico de shrinkage para estabilizar setores com pouco histórico.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ShapeConfig:
    setor_col: str = "setor"
    ciclo_col: str = "ciclo"
    data_col: str = "data_captacao"
    abertura_col: str = "data_abertura_ciclo"
    fechamento_col: str = "data_fechamento_ciclo"
    bloco_col: str = "bloco"
    regiao_col: str = "regiao"
    volume_col: str = "volume"
    min_ciclos_setor: int = 3
    k_setor: float = 3.0   # força do prior de bloco, em "ciclos equivalentes"
    k_bloco: float = 6.0   # força do prior de região
    grid_size: int = 21    # bins de posição relativa [0,1]

def _normalize(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v[v < 0] = 0
    s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)

def preparar_base(df, cfg):
    req = [cfg.setor_col,cfg.ciclo_col,cfg.data_col,cfg.abertura_col,
           cfg.fechamento_col,cfg.bloco_col,cfg.regiao_col,cfg.volume_col]
    falt = [c for c in req if c not in df.columns]
    if falt:
        raise ValueError(f"Colunas obrigatórias ausentes: {falt}")

    x = df.copy()
    for c in [cfg.data_col,cfg.abertura_col,cfg.fechamento_col]:
        x[c] = pd.to_datetime(x[c])

    # Regra de negócio: retirar captação fora da janela oficial.
    x = x[(x[cfg.data_col] >= x[cfg.abertura_col]) &
          (x[cfg.data_col] <= x[cfg.fechamento_col])].copy()

    dur = (x[cfg.fechamento_col] - x[cfg.abertura_col]).dt.days
    pos = (x[cfg.data_col] - x[cfg.abertura_col]).dt.days
    x["u"] = np.where(dur > 0, pos / dur, 0.0)
    x["u"] = x["u"].clip(0,1)
    x["bin"] = np.minimum((x["u"] * cfg.grid_size).astype(int), cfg.grid_size-1)
    return x

def curvas_por_ciclo(x, group_cols, cfg):
    """Retorna uma curva normalizada por ciclo e depois a média empírica."""
    g = (x.groupby(group_cols + [cfg.ciclo_col, "bin"], as_index=False)[cfg.volume_col]
           .sum())
    g["tot"] = g.groupby(group_cols + [cfg.ciclo_col])[cfg.volume_col].transform("sum")
    g = g[g["tot"] > 0].copy()
    g["share"] = g[cfg.volume_col] / g["tot"]

    full_index = pd.MultiIndex.from_product(
        [*[g[c].dropna().unique() for c in group_cols],
         g[cfg.ciclo_col].dropna().unique(),
         range(cfg.grid_size)],
        names=group_cols+[cfg.ciclo_col,"bin"]
    )
    # Reindexing full cartesian product may create invalid group combinations.
    # Safer: complete bins within each observed group/cycle.
    rows = []
    for keys, z in g.groupby(group_cols+[cfg.ciclo_col], dropna=False):
        z2 = z.set_index("bin")["share"].reindex(range(cfg.grid_size), fill_value=0)
        keytuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols+[cfg.ciclo_col], keytuple))
        for b,val in z2.items():
            rows.append({**row, "bin": b, "share": val})
    return pd.DataFrame(rows)

def _mean_curve(cycle_curves, filter_dict, cfg):
    z = cycle_curves.copy()
    for c,v in filter_dict.items():
        z = z[z[c] == v]
    if z.empty:
        return None, 0
    n_cycles = z[cfg.ciclo_col].nunique()
    curve = z.groupby("bin")["share"].mean().reindex(range(cfg.grid_size), fill_value=0).values
    return _normalize(curve), n_cycles

def extrair_shape(df, setor, periodo_inicio=None, periodo_fim=None, cfg=ShapeConfig()):
    """
    Retorna share normalizado s_i,t para um setor.
    Shrinkage: setor -> bloco -> região.
    """
    x = preparar_base(df, cfg)
    if periodo_inicio is not None:
        x = x[x[cfg.data_col] >= pd.Timestamp(periodo_inicio)]
    if periodo_fim is not None:
        x = x[x[cfg.data_col] <= pd.Timestamp(periodo_fim)]

    xs = x[x[cfg.setor_col] == setor]
    if xs.empty:
        raise ValueError(f"Setor {setor} não encontrado no período.")

    # Usa bloco/região históricos modais apenas para formar os priors.
    bloco = xs[cfg.bloco_col].mode().iloc[0]
    regiao = xs[cfg.regiao_col].mode().iloc[0]

    cc_setor = curvas_por_ciclo(x, [cfg.setor_col], cfg)
    cc_bloco = curvas_por_ciclo(x, [cfg.bloco_col], cfg)
    cc_regiao = curvas_por_ciclo(x, [cfg.regiao_col], cfg)

    s_setor, n_s = _mean_curve(cc_setor, {cfg.setor_col:setor}, cfg)
    s_bloco, n_b = _mean_curve(cc_bloco, {cfg.bloco_col:bloco}, cfg)
    s_regiao, n_r = _mean_curve(cc_regiao, {cfg.regiao_col:regiao}, cfg)

    if s_regiao is None:
        raise ValueError("Não foi possível formar a curva regional.")
    if s_bloco is None:
        s_bloco = s_regiao

    # Shrinkage bloco -> região
    w_b = n_b / (n_b + cfg.k_bloco)
    prior_bloco = _normalize(w_b*s_bloco + (1-w_b)*s_regiao)

    # Shrinkage setor -> prior de bloco
    if s_setor is None:
        final = prior_bloco
        w_s = 0.0
    else:
        w_s = n_s / (n_s + cfg.k_setor)
        final = _normalize(w_s*s_setor + (1-w_s)*prior_bloco)

    out = pd.DataFrame({
        "setor": setor,
        "bin_t": np.arange(1, cfg.grid_size+1),
        "posicao_relativa": np.linspace(0,1,cfg.grid_size),
        "share_shape": final
    })
    meta = {
        "setor": setor, "bloco_prior": bloco, "regiao": regiao,
        "n_ciclos_setor": n_s, "n_ciclos_bloco": n_b, "n_ciclos_regiao": n_r,
        "peso_setor": w_s, "peso_bloco": w_b,
        "soma_share": float(out["share_shape"].sum())
    }
    return out, meta

def escolher_regiao_piloto(df, cfg=ShapeConfig()):
    """
    Critério pragmático:
    1) maior volume histórico;
    2) desempate por maior número de ciclos;
    3) depois maior número de setores.
    """
    x = preparar_base(df, cfg)
    tab = (x.groupby(cfg.regiao_col)
             .agg(volume=(cfg.volume_col,"sum"),
                  ciclos=(cfg.ciclo_col,"nunique"),
                  setores=(cfg.setor_col,"nunique"))
             .sort_values(["volume","ciclos","setores"], ascending=False))
    return tab.index[0], tab

if __name__ == "__main__":
    # Exemplo:
    # base = pd.read_csv("Base_Demanda.csv")
    # regiao, ranking = escolher_regiao_piloto(base)
    # setor = (base[base["regiao"]==regiao].groupby("setor")["volume"].sum().idxmax())
    # shape, meta = extrair_shape(base, setor, "2025-01-01", "2026-08-31")
    # shape.to_csv(f"shape_{setor}.csv", index=False)
    pass

# --- EXECUÇÃO DIAGNÓSTICA COM O CSV DISPONÍVEL ---
def executar_diagnostico_semanal():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    pasta = Path(__file__).resolve().parent
    entrada = pasta / 'serie_volume_setor(1).csv'
    if not entrada.exists():
        raise FileNotFoundError("Coloque 'serie_volume_setor(1).csv' na mesma pasta deste script.")

    df = pd.read_csv(entrada)
    req = {'cd_setor','aa_ciclo','nm_ciclo','mes_pedido','semana_pedido','total_volumes_mascarado'}
    falt = req - set(df.columns)
    if falt:
        raise ValueError(f'Colunas ausentes: {sorted(falt)}')

    df['cycle_id'] = df['aa_ciclo'].astype(str) + '-' + df['nm_ciclo'].astype(str).str.zfill(2)
    n_ciclos = df[['cd_setor','cycle_id']].drop_duplicates().groupby('cd_setor').size()
    volumes = df.groupby('cd_setor')['total_volumes_mascarado'].sum()
    elegiveis = n_ciclos[n_ciclos >= 8].index
    setor = volumes.loc[elegiveis].idxmax() if len(elegiveis) else volumes.idxmax()

    x = df[df['cd_setor'] == setor].copy().sort_values(['aa_ciclo','nm_ciclo','mes_pedido','semana_pedido'])
    x['posicao_semanal'] = x.groupby('cycle_id').cumcount() + 1
    x['total_ciclo'] = x.groupby('cycle_id')['total_volumes_mascarado'].transform('sum')
    x = x[x['total_ciclo'] > 0].copy()
    x['share_ciclo'] = x['total_volumes_mascarado'] / x['total_ciclo']
    shape = x.groupby('posicao_semanal', as_index=False)['share_ciclo'].mean().rename(columns={'share_ciclo':'share_shape'})
    shape['share_shape'] /= shape['share_shape'].sum()

    csv_saida = pasta / 'curva_piloto_diagnostica_semanal.csv'
    png_saida = pasta / 'curva_piloto_diagnostica_semanal.png'
    shape.to_csv(csv_saida, index=False)

    plt.figure(figsize=(8.5,4.5))
    plt.plot(shape['posicao_semanal'], shape['share_shape'], marker='o')
    plt.xlabel('Posição semanal relativa observada no ciclo')
    plt.ylabel('Share normalizado')
    plt.title(f'Diagnóstico de shape semanal — setor {setor}')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(png_saida, dpi=180, bbox_inches='tight')
    plt.show()

    print(f'Setor piloto: {setor}')
    print(f'Ciclos disponíveis: {int(n_ciclos.loc[setor])}')
    print(f'Soma dos shares: {shape["share_shape"].sum():.12f}')
    print(f'Gráfico salvo em: {png_saida}')
    print(f'Curva salva em: {csv_saida}')
    print('OBS.: resultado diagnóstico semanal; região/shrinkage final dependem da Base Demanda completa.')


if __name__ == '__main__':
    executar_diagnostico_semanal()
