import csv
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.easter import easter
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ==========================================
# Parâmetros gerais
# ==========================================
DIR_REPOSITORIO = Path(__file__).resolve().parents[2]
DIR_DADOS = DIR_REPOSITORIO / "src" / "processamento_dados"
CAMINHO_CSV = DIR_DADOS / "base_tratada_v2.csv"
DIR_EXTERNOS = DIR_DADOS
CAMINHO_SALARIO = DIR_EXTERNOS / "media_salarial.csv"
CAMINHO_DESEMPREGO = DIR_EXTERNOS / "taxa_desemprego.csv"
CAMINHO_CENSO = DIR_EXTERNOS / "censo_2022.xlsx"
SAIDA = DIR_REPOSITORIO / "data" / "generated" / "prophet"

ALVO = "total_itens_mascarado"  # ou "total_volumes_mascarado"

# Colunas do calendário de ciclos (conhecidas com antecedência, seguras no futuro)
REGRESSORES = ["dias_para_fechamento", "dia_ciclo"]

# Janela expansiva: o treino começa sempre no início do histórico e cresce a cada janela.
DIAS_TREINO = 180  # tamanho do treino da PRIMEIRA janela
DIAS_TESTE = 30  # horizonte de previsão
PASSO = 30  # = DIAS_TESTE => janelas de teste sem sobreposição

# ==========================================
# FEATUREs
# ==========================================
# Cada feature é um bloco de colunas que entra (ou não) no Prophet.
FEATURES = {
    "ciclo": {"rotulo": "ciclo", "colunas": REGRESSORES},
    "quinto_du": {"rotulo": "5º dia útil", "colunas": ["quinto_dia_util"]},
    "salario": {"rotulo": "salário", "colunas": ["salario_medio"]},
    "desemprego": {"rotulo": "desemprego", "colunas": ["taxa_desemprego"]},
    "censo": {
        "rotulo": "censo",
        "colunas": ["censo_log_pop", "censo_log_dens"],
    },  # constantes no tempo
}
GRUPOS_EXTERNOS = {"salario", "desemprego", "censo"}  # features que dependem dos arquivos do IBGE


FEATURES_ATIVAS = [
    "ciclo",
    "quinto_du",
    "salario",
    "desemprego",
    # "censo",
]

# Quais combinações testar automaticamente (use uma ou mais):
#   "individual": a base + cada feature sozinha
#   "pares":      todas as combinações de 2 features
#   "ablacao":    todas as features menos uma (mostra o que cada uma acrescenta ao conjunto)
#   "todas":      todas as features juntas
MODOS = ["individual", "todas"]

# Combinações específicas (podem usar qualquer feature de FEATURES), ex.:
#   EXPERIMENTOS_CUSTOM = [["ciclo", "quinto_du"], ["ciclo", "quinto_du", "salario"]]
EXPERIMENTOS_CUSTOM = []

# Prophet sem nenhuma feature (só sazonalidade semanal e feriados). Sem o ingênuo, ele é a
# referência das colunas "ganho_vs_base_%"; se for False, essa coluna não é gerada.
INCLUIR_BASE = True

# ------------------------------------------
# Quinto dia útil
# ------------------------------------------
# Flag = 1 no quinto dia útil do mês (prazo de pagamento de salário, que pode puxar a demanda).
# A regra trabalhista de pagamento costuma contar o sábado como dia útil; a convenção
# bancária não. Ajuste conforme o que você quer testar.
SABADO_E_DIA_UTIL = False
QUINTO_DU_DIAS_APOS = 0  # 0 = só o próprio dia; N = o dia e os N dias corridos seguintes
EXTRA_NAO_UTEIS = []  # datas adicionais "YYYY-MM-DD" (feriados estaduais/municipais)

# ------------------------------------------
# Regressores externos (IBGE)
# ------------------------------------------
# Os dados do IBGE são por UF; cada CD atende várias UFs. O indicador do CD é a média
# dos indicadores das UFs ponderada pela participação da UF no volume do CD.
#
# Divulgação trimestral: no dia d, só se conhece o trimestre já publicado.
# LAG_TRIMESTRES = 1 => um dia do trimestre T usa o valor do trimestre T-1
# (a PNAD sai ~45 dias depois do fim do trimestre). Valores fora do intervalo
# disponível (ex.: 3º tri 2026) repetem o último trimestre conhecido.
LAG_TRIMESTRES = 1

M_BASE = "Prophet base (sem features)"

# Sigla -> (código IBGE, nome)
UFS = {
    "RO": (11, "Rondônia"),
    "AC": (12, "Acre"),
    "AM": (13, "Amazonas"),
    "RR": (14, "Roraima"),
    "PA": (15, "Pará"),
    "AP": (16, "Amapá"),
    "TO": (17, "Tocantins"),
    "MA": (21, "Maranhão"),
    "PI": (22, "Piauí"),
    "CE": (23, "Ceará"),
    "RN": (24, "Rio Grande do Norte"),
    "PB": (25, "Paraíba"),
    "PE": (26, "Pernambuco"),
    "AL": (27, "Alagoas"),
    "SE": (28, "Sergipe"),
    "BA": (29, "Bahia"),
    "MG": (31, "Minas Gerais"),
    "ES": (32, "Espírito Santo"),
    "RJ": (33, "Rio de Janeiro"),
    "SP": (35, "São Paulo"),
    "PR": (41, "Paraná"),
    "SC": (42, "Santa Catarina"),
    "RS": (43, "Rio Grande do Sul"),
    "MS": (50, "Mato Grosso do Sul"),
    "MT": (51, "Mato Grosso"),
    "GO": (52, "Goiás"),
    "DF": (53, "Distrito Federal"),
}
COD_POR_SIGLA = {s: c for s, (c, _) in UFS.items()}
SIGLA_POR_COD = {c: s for s, (c, _) in UFS.items()}
SIGLA_POR_NOME = {n: s for s, (_, n) in UFS.items()}

# Regressores descartados por serem constantes no treino: (modelo, cd) -> {nomes}
REG_DESCARTADOS = defaultdict(set)


# ==========================================
# Experimentos (combinações de features)
# ==========================================
def nome_modelo(combo: tuple) -> str:
    if not combo:
        return M_BASE
    return "Prophet: " + " + ".join(FEATURES[f]["rotulo"] for f in combo)


def gerar_experimentos() -> dict:
    """Devolve {nome do modelo: tupla de features}, sem duplicatas."""
    desconhecidas = [f for f in FEATURES_ATIVAS if f not in FEATURES]
    for c in EXPERIMENTOS_CUSTOM:
        desconhecidas += [f for f in c if f not in FEATURES]
    if desconhecidas:
        raise ValueError(
            f"Features inexistentes: {sorted(set(desconhecidas))}. Opções: {list(FEATURES)}"
        )

    ativas = list(FEATURES_ATIVAS)
    combos = []
    if INCLUIR_BASE:
        combos.append(())
    if "individual" in MODOS:
        combos += [(f,) for f in ativas]
    if "pares" in MODOS:
        combos += list(itertools.combinations(ativas, 2))
    if "ablacao" in MODOS and len(ativas) > 1:
        combos += [tuple(x for x in ativas if x != f) for f in ativas]
    if "todas" in MODOS and ativas:
        combos.append(tuple(ativas))
    combos += [tuple(c) for c in EXPERIMENTOS_CUSTOM]

    ordem = list(FEATURES)  # ordem canônica para nomes consistentes
    experimentos = {}
    for c in combos:
        canonico = tuple(f for f in ordem if f in c)
        experimentos.setdefault(nome_modelo(canonico), canonico)
    return experimentos


EXPERIMENTOS = gerar_experimentos()
MODELOS_PROPHET = list(EXPERIMENTOS)
FEATURES_USADAS = {f for combo in EXPERIMENTOS.values() for f in combo}


def regressores_do_modelo(nome: str) -> list:
    cols = []
    for f in EXPERIMENTOS[nome]:
        cols += FEATURES[f]["colunas"]
    return cols


# ==========================================
# Feriados e datas comerciais (efeito no Prophet)
# ==========================================
def _enesimo_dia_semana(ano: int, mes: int, dia_semana: int, n: int) -> pd.Timestamp:
    primeiro = pd.Timestamp(ano, mes, 1)
    deslocamento = (dia_semana - primeiro.weekday()) % 7
    return primeiro + pd.Timedelta(days=deslocamento + 7 * (n - 1))


def gerar_feriados_brasileiros(ds_min: pd.Timestamp, ds_max: pd.Timestamp) -> pd.DataFrame:
    linhas = []
    for ano in range(ds_min.year, ds_max.year + 1):
        pascoa = pd.Timestamp(easter(ano))
        linhas.extend(
            [
                (pd.Timestamp(ano, 1, 1), "Ano Novo"),
                (pascoa - pd.Timedelta(days=47), "Carnaval"),  # terça; janela cobre seg/qua
                (pascoa - pd.Timedelta(days=2), "Sexta-feira Santa"),
                (pd.Timestamp(ano, 4, 21), "Tiradentes"),
                (pd.Timestamp(ano, 5, 1), "Dia do Trabalho"),
                (pascoa + pd.Timedelta(days=60), "Corpus Christi"),
                (pd.Timestamp(ano, 9, 7), "Independência"),
                (pd.Timestamp(ano, 10, 12), "Nossa Senhora Aparecida"),
                (pd.Timestamp(ano, 11, 2), "Finados"),
                (pd.Timestamp(ano, 11, 15), "Proclamação da República"),
                (pd.Timestamp(ano, 12, 25), "Natal"),
                # Datas comerciais relevantes para cosméticos
                (_enesimo_dia_semana(ano, 5, 6, 2), "Dia das Mães"),
                (pd.Timestamp(ano, 6, 12), "Dia dos Namorados"),
                (_enesimo_dia_semana(ano, 8, 6, 2), "Dia dos Pais"),
                (_enesimo_dia_semana(ano, 11, 3, 4) + pd.Timedelta(days=1), "Black Friday"),
            ]
        )
    feriados = pd.DataFrame(
        {"holiday": [n for _, n in linhas], "ds": pd.to_datetime([d for d, _ in linhas])}
    )
    feriados["lower_window"] = -1
    feriados["upper_window"] = 1
    return feriados.drop_duplicates(subset=["ds", "holiday"]).reset_index(drop=True)


# ==========================================
# Quinto dia útil
# ==========================================
def datas_nao_uteis(ano: int) -> set:
    """Feriados nacionais e dias de Carnaval (não úteis para efeito bancário)."""
    pascoa = pd.Timestamp(easter(ano))
    datas = {
        pd.Timestamp(ano, 1, 1),
        pascoa - pd.Timedelta(days=48),  # segunda de Carnaval
        pascoa - pd.Timedelta(days=47),  # terça de Carnaval
        pascoa - pd.Timedelta(days=2),  # Sexta-feira Santa
        pd.Timestamp(ano, 4, 21),
        pd.Timestamp(ano, 5, 1),
        pascoa + pd.Timedelta(days=60),  # Corpus Christi
        pd.Timestamp(ano, 9, 7),
        pd.Timestamp(ano, 10, 12),
        pd.Timestamp(ano, 11, 2),
        pd.Timestamp(ano, 11, 15),
        pd.Timestamp(ano, 12, 25),
    }
    if ano >= 2024:  # Consciência Negra é feriado nacional desde 2024
        datas.add(pd.Timestamp(ano, 11, 20))
    return datas


def datas_quinto_dia_util(ds_min: pd.Timestamp, ds_max: pd.Timestamp) -> list:
    nao_uteis = {pd.Timestamp(d) for d in EXTRA_NAO_UTEIS}
    for ano in range(ds_min.year, ds_max.year + 1):
        nao_uteis |= datas_nao_uteis(ano)
    limite_semana = 6 if SABADO_E_DIA_UTIL else 5  # weekday: seg=0 ... sáb=5
    quintos = []
    for periodo in pd.period_range(ds_min.to_period("M"), ds_max.to_period("M"), freq="M"):
        dias = pd.date_range(periodo.start_time, periodo.end_time.normalize())
        uteis = [d for d in dias if d.weekday() < limite_semana and d not in nao_uteis]
        if len(uteis) >= 5:
            quintos.append(uteis[4])
    return quintos


def adicionar_quinto_du(serie: pd.DataFrame) -> pd.DataFrame:
    """Flag derivada só do calendário (não usa o alvo): segura no futuro."""
    serie = serie.copy()
    flag = np.zeros(len(serie), dtype=int)
    for q in datas_quinto_dia_util(serie["ds"].min(), serie["ds"].max()):
        dentro = serie["ds"].between(q, q + pd.Timedelta(days=QUINTO_DU_DIAS_APOS))
        flag |= dentro.to_numpy().astype(int)
    serie["quinto_dia_util"] = flag
    return serie


def diagnosticar_quinto_du(series: dict) -> None:
    """Comparação descritiva (não causal: não separa dia da semana nem ciclo)."""
    linhas = []
    for cd, s in series.items():
        rel = s["y"] / s["y"].mean()
        marcado = s["quinto_dia_util"] == 1
        linhas.append(
            {
                "cd_cd": cd,
                "dias_marcados": int(marcado.sum()),
                "vol_rel_5DU": rel[marcado].mean(),
                "vol_rel_demais": rel[~marcado].mean(),
            }
        )
    print("\n--- Quinto dia útil: volume relativo à média do CD ---")
    print(pd.DataFrame(linhas).round(3).to_string(index=False))


# ==========================================
# Dados
# ==========================================
def carregar_base() -> pd.DataFrame:
    colunas = ["data_pedido", "cd_cd", "estado", "IsDateValid", ALVO] + REGRESSORES
    df = pd.read_csv(CAMINHO_CSV, usecols=colunas, parse_dates=["data_pedido"])

    valido = df["IsDateValid"].astype(str).str.lower().eq("true")
    print(f"Linhas: {len(df):,} | descartadas por IsDateValid=False: {(~valido).sum():,}")
    df = df[valido].copy()

    # O CSV usa ponto decimal (ex.: 2818.0): NÃO remover o ponto.
    if df[ALVO].dtype == object:
        df[ALVO] = df[ALVO].astype(str).str.replace(",", ".", regex=False)
    df[ALVO] = pd.to_numeric(df[ALVO], errors="coerce")
    print(f"Valores não numéricos no alvo: {df[ALVO].isna().sum():,}")
    df[ALVO] = df[ALVO].fillna(0)
    return df


def diagnosticar(df: pd.DataFrame) -> None:
    print("\n--- Diagnóstico ---")
    print("Período:", df["data_pedido"].min().date(), "->", df["data_pedido"].max().date())
    for cd, g in df.groupby("cd_cd"):
        dias_obs = g["data_pedido"].nunique()
        dias_total = (g["data_pedido"].max() - g["data_pedido"].min()).days + 1
        print(
            f"CD {cd}: {dias_obs} dias observados de {dias_total} "
            f"({dias_total - dias_obs} ausentes)"
        )
    for r in REGRESSORES:
        n = df.groupby(["cd_cd", "data_pedido"])[r].nunique()
        print(f"{r}: {(n > 1).mean() * 100:.1f}% dos pares (CD, dia) com mais de um valor")


# ==========================================
# Regressores externos (IBGE)
# ==========================================
def _indice_trimestre(ano, trimestre):
    """Inteiro monotônico: 1º tri 2024 -> 2024*4 + 0."""
    return np.asarray(ano) * 4 + (np.asarray(trimestre) - 1)


def _ler_ibge_trimestral(caminho: Path, nome_valor: str) -> pd.DataFrame:
    """Lê as tabelas SIDRA (salário e desemprego) e devolve uf, q, <nome_valor>."""
    # Linhas com número de colunas diferente (títulos, notas): lê com csv e preenche
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas_csv = list(csv.reader(f, delimiter=";"))
    largura = max(len(linha) for linha in linhas_csv)
    bruto = pd.DataFrame(
        [linha + [None] * (largura - len(linha)) for linha in linhas_csv],
        dtype=object,
    )
    # Linha de cabeçalho com os trimestres
    linha_cab = bruto.index[
        bruto.apply(lambda r: r.astype(str).str.contains("trimestre").any(), axis=1)
    ][0]
    cab = bruto.loc[linha_cab]
    cols_tri = {}
    for j, rot in cab.items():
        m = pd.Series([str(rot)]).str.extract(r"(\d)º trimestre (\d{4})").iloc[0]
        if m.notna().all():
            cols_tri[j] = int(_indice_trimestre(int(m[1]), int(m[0])))

    # Salário: (Nível, Cód., Nome, ...) com Nível == "UF"; desemprego: (Cód., Nome, ...)
    formato_nivel = str(cab[0]).strip() == "Nível"
    linhas = []
    for _, r in bruto.iloc[linha_cab + 1 :].iterrows():
        if formato_nivel:
            if str(r[0]).strip() != "UF":
                continue
            cod = r[1]
        else:
            cod = r[0]
        try:
            cod = int(str(cod).strip())
        except ValueError:
            continue
        if cod not in SIGLA_POR_COD:  # ignora Brasil e Grandes Regiões (códigos 1 a 5)
            continue
        for j, q in cols_tri.items():
            v = pd.to_numeric(str(r[j]).replace(",", "."), errors="coerce")  # X, ... => NaN
            linhas.append({"uf": SIGLA_POR_COD[cod], "q": q, nome_valor: v})
    return pd.DataFrame(linhas)


def _ler_censo(caminho: Path) -> pd.DataFrame:
    """Lê população e densidade por UF do Censo 2022."""
    bruto = pd.read_excel(caminho, sheet_name=0, header=None)
    linhas = []
    for _, r in bruto.iterrows():
        nome = str(r[0]).strip()
        if nome in SIGLA_POR_NOME:
            linhas.append(
                {
                    "uf": SIGLA_POR_NOME[nome],
                    "pop": pd.to_numeric(r[1], errors="coerce"),
                    "dens": pd.to_numeric(r[3], errors="coerce"),
                }
            )
    out = pd.DataFrame(linhas)
    out["censo_log_pop"] = np.log(out["pop"])
    out["censo_log_dens"] = np.log(out["dens"])
    return out[["uf", "censo_log_pop", "censo_log_dens"]]


def pesos_uf_por_cd(df: pd.DataFrame) -> pd.DataFrame:
    """Participação de cada UF no volume total do CD (soma 1 por CD).

    Usa o histórico inteiro: é uma característica geográfica estável do CD, não varia por dia.
    """
    p = df.groupby(["cd_cd", "estado"], as_index=False)[ALVO].sum()
    p = p[p[ALVO] > 0].rename(columns={"estado": "uf"})
    p["peso"] = p[ALVO] / p.groupby("cd_cd")[ALVO].transform("sum")
    return p[["cd_cd", "uf", "peso"]]


def _media_ponderada(g: pd.DataFrame, col: str) -> float:
    ok = g[col].notna()
    return np.average(g.loc[ok, col], weights=g.loc[ok, "peso"]) if ok.any() else np.nan


def construir_externos(df: pd.DataFrame, usados: set) -> dict:
    """Devolve {cd: (serie_trimestral_df indexada por q ou None, constantes_dict)}.

    Só lê os arquivos das features que estão em uso nos experimentos.
    """
    pesos = pesos_uf_por_cd(df)
    sal = _ler_ibge_trimestral(CAMINHO_SALARIO, "salario_medio") if "salario" in usados else None
    des = (
        _ler_ibge_trimestral(CAMINHO_DESEMPREGO, "taxa_desemprego")
        if "desemprego" in usados
        else None
    )
    censo = _ler_censo(CAMINHO_CENSO) if "censo" in usados else None

    ufs_base = set(pesos["uf"])
    for nome, t in (("salário", sal), ("desemprego", des), ("censo", censo)):
        if t is not None:
            faltando = ufs_base - set(t["uf"])
            if faltando:
                print(f"AVISO: UFs da base sem dado de {nome}: {sorted(faltando)}")

    tri = None
    for t in (sal, des):
        if t is not None:
            tri = t if tri is None else tri.merge(t, on=["uf", "q"], how="outer")

    out = {}
    for cd, g_pesos in pesos.groupby("cd_cd"):
        serie_tri = None
        if tri is not None:
            g = g_pesos.merge(tri, on="uf", how="left")
            cols_valor = [c for c in ("salario_medio", "taxa_desemprego") if c in g.columns]
            linhas = [
                {"q": q, **{c: _media_ponderada(gq, c) for c in cols_valor}}
                for q, gq in g.groupby("q")
            ]
            serie_tri = pd.DataFrame(linhas).set_index("q").sort_index()
        const = {}
        if censo is not None:
            gc = g_pesos.merge(censo, on="uf", how="left")
            const = {c: _media_ponderada(gc, c) for c in ("censo_log_pop", "censo_log_dens")}
        out[cd] = (serie_tri, const)
    return out


def adicionar_externos(serie: pd.DataFrame, serie_tri, const: dict) -> pd.DataFrame:
    """Mapeia cada dia para o trimestre já divulgado (com defasagem) e anexa as colunas."""
    serie = serie.copy()
    if serie_tri is not None:
        q_dia = _indice_trimestre(serie["ds"].dt.year, serie["ds"].dt.quarter)
        q_ref = np.clip(q_dia - LAG_TRIMESTRES, serie_tri.index.min(), serie_tri.index.max())
        completo = serie_tri.reindex(
            range(serie_tri.index.min(), serie_tri.index.max() + 1)
        ).ffill()
        for col in serie_tri.columns:
            serie[col] = completo[col].reindex(q_ref).to_numpy()
    for col, v in const.items():
        serie[col] = v
    return serie


def montar_series_por_cd(df: pd.DataFrame, externos: dict) -> dict:
    agg = {ALVO: "sum", **{r: "first" for r in REGRESSORES}}
    diario = (
        df.groupby(["cd_cd", "data_pedido"], as_index=False)
        .agg(agg)
        .rename(columns={"data_pedido": "ds", ALVO: "y"})
        .dropna(subset=REGRESSORES)
        .sort_values(["cd_cd", "ds"])
    )
    series = {}
    for cd, g in diario.groupby("cd_cd"):
        g = g.drop(columns="cd_cd").reset_index(drop=True)
        if externos:
            serie_tri, const = externos[cd]
            g = adicionar_externos(g, serie_tri, const)
        series[cd] = adicionar_quinto_du(g)
    return series


def diagnosticar_externos(series: dict) -> None:
    """Quanto cada regressor externo realmente varia dentro de cada CD."""
    cols = [c for f in FEATURES_USADAS & GRUPOS_EXTERNOS for c in FEATURES[f]["colunas"]]
    if not cols:
        return
    linhas = []
    for cd, s in series.items():
        linha = {"cd_cd": cd}
        for c in cols:
            linha[f"{c}_distintos"] = s[c].nunique()
            linha[f"{c}_min"] = s[c].min()
            linha[f"{c}_max"] = s[c].max()
        linhas.append(linha)
    print("\n--- Regressores externos por CD (valores distintos e faixa) ---")
    print(pd.DataFrame(linhas).round(2).to_string(index=False))


# ==========================================
# Métricas
# ==========================================
def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    erro = y_pred - y_true  # dashboard: Bias > 0 => superestima
    soma = np.abs(y_true).sum()
    return {
        "MAE": np.abs(erro).mean(),
        "WMAPE": np.abs(erro).sum() / soma if soma != 0 else np.nan,
        "Bias": erro.mean(),
    }


def metricas_por(df: pd.DataFrame, chaves: list) -> pd.DataFrame:
    linhas = []
    for valores, g in df.groupby(chaves, sort=True):
        valores = valores if isinstance(valores, tuple) else (valores,)
        linhas.append({**dict(zip(chaves, valores)), **calculate_metrics(g["y"], g["yhat"])})
    return pd.DataFrame(linhas)


# ==========================================
# Modelo
# ==========================================
def ajustar_prophet(treino: pd.DataFrame, feriados: pd.DataFrame, regressores: list):
    # Regressor constante no treino é colinear com o intercepto: não traz informação
    regs = [r for r in regressores if treino[r].nunique() > 1]
    descartados = [r for r in regressores if r not in regs]
    modelo = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False,  # ciclo anual pouco estimável com ~21 meses de histórico
        holidays=feriados,
    )
    for r in regs:
        modelo.add_regressor(r)
    modelo.fit(treino[["ds", "y"] + regs])
    return modelo, regs, descartados


# ==========================================
# Janela expansiva (por CD)
# ==========================================
def avaliar_cd(cd, serie: pd.DataFrame, feriados: pd.DataFrame) -> pd.DataFrame:
    """Treino sempre do início do histórico até a origem; a origem avança PASSO dias por janela."""
    inicio0 = serie["ds"].min()
    fim_dados = serie["ds"].max()
    partes = []
    janela = 0
    deslocamento = 0  # dias que a origem já avançou

    while True:
        fim_treino = inicio0 + pd.Timedelta(days=DIAS_TREINO + deslocamento)  # exclusivo
        fim_teste = fim_treino + pd.Timedelta(days=DIAS_TESTE)  # exclusivo
        if fim_teste > fim_dados + pd.Timedelta(days=1):
            break

        treino = serie[serie["ds"] < fim_treino]
        teste = serie[(serie["ds"] >= fim_treino) & (serie["ds"] < fim_teste)]
        if len(treino) >= 60 and len(teste) > 0:
            janela += 1
            base = teste[["ds", "y"]].copy()
            base["cd_cd"] = cd
            base["janela"] = janela
            base["horizonte"] = (base["ds"] - fim_treino).dt.days + 1
            base["treino_inicio"] = treino["ds"].min()
            base["teste_inicio"] = teste["ds"].min()
            base["n_treino"] = len(treino)

            for nome in MODELOS_PROPHET:
                modelo, regs, descartados = ajustar_prophet(
                    treino, feriados, regressores_do_modelo(nome)
                )
                REG_DESCARTADOS[(nome, cd)].update(descartados)
                yhat = modelo.predict(teste[["ds"] + regs])["yhat"].clip(lower=0).to_numpy()
                partes.append(base.assign(yhat=yhat, modelo=nome))
        deslocamento += PASSO

    if not partes:
        raise ValueError(
            f"CD {cd}: histórico insuficiente para treino={DIAS_TREINO} + teste={DIAS_TESTE}."
        )
    out = pd.concat(partes, ignore_index=True)
    out["residuo"] = out["y"] - out["yhat"]
    return out


# ==========================================
# Gráficos
# ==========================================
def _montar_cores() -> dict:
    cores = {M_BASE: "#9D9D9D"}
    paleta = list(plt.cm.tab20.colors)
    i = 0
    for m in MODELOS_PROPHET:
        if m not in cores:
            cores[m] = paleta[(2 * i) % len(paleta)]
            i += 1
    return cores


CORES = _montar_cores()


def plotar_comparacao_janelas(previsoes: pd.DataFrame) -> Path:
    caminho = SAIDA / "grafico_comparacao_janelas.png"
    por_janela = metricas_por(previsoes, ["modelo", "janela"])
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
    for eixo, (col, rot) in zip(eixos, [("MAE", "MAE"), ("WMAPE", "WMAPE"), ("Bias", "Bias")]):
        for modelo, g in por_janela.groupby("modelo"):
            eixo.plot(g["janela"], g[col], marker="o", label=modelo, color=CORES[modelo])
        if col == "Bias":
            eixo.axhline(0, color="black", linewidth=0.8, linestyle="--")
        eixo.set(title=rot, xlabel="Janela de validação", ylabel=rot)
        eixo.grid(alpha=0.3)
    eixos[0].legend(fontsize=8)
    fig.suptitle("Desempenho por janela expansiva (todos os CDs agrupados)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def plotar_metricas_por_horizonte(previsoes: pd.DataFrame) -> Path:
    caminho = SAIDA / "grafico_metricas_por_horizonte.png"
    por_h = metricas_por(previsoes, ["modelo", "horizonte"])
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
    for eixo, (col, rot) in zip(eixos, [("MAE", "MAE"), ("WMAPE", "WMAPE"), ("Bias", "Bias")]):
        for modelo, g in por_h.groupby("modelo"):
            eixo.plot(g["horizonte"], g[col], marker="o", label=modelo, color=CORES[modelo])
        if col == "Bias":
            eixo.axhline(0, color="black", linewidth=0.8, linestyle="--")
        eixo.set(title=rot, xlabel="Horizonte (dias à frente)", ylabel=rot)
        eixo.grid(alpha=0.3)
    eixos[0].legend(fontsize=8)
    fig.suptitle("Métricas por horizonte de previsão")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def plotar_bias_residuos(previsoes: pd.DataFrame, modelo: str) -> Path:
    caminho = SAIDA / "grafico_bias_residuos.png"
    p = previsoes[previsoes["modelo"] == modelo]
    g = p.groupby("ds", as_index=False)["residuo"].sum().sort_values("ds")
    g["bias_movel"] = g["residuo"].rolling(7, min_periods=1).mean()

    fig, eixo = plt.subplots(figsize=(14, 6))
    eixo.scatter(
        g["ds"],
        g["residuo"],
        s=18,
        alpha=0.55,
        color="#4C78A8",
        label="Resíduo diário (real - previsto, soma dos CDs)",
    )
    eixo.plot(g["ds"], g["bias_movel"], color="#E45756", linewidth=2, label="Média móvel 7 dias")
    eixo.axhline(0, color="black", linewidth=1, linestyle="--")
    eixo.set(title=f"Bias ao longo do tempo ({modelo})", xlabel="Data", ylabel="Resíduo")
    eixo.xaxis.set_major_locator(mdates.AutoDateLocator())
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    eixo.legend()
    eixo.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


# ==========================================
# Execução
# ==========================================
def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    print(f"Modelos Prophet a comparar ({len(MODELOS_PROPHET)}), janela expansiva:")
    for nome in MODELOS_PROPHET:
        print(f"  - {nome}  [{', '.join(regressores_do_modelo(nome)) or 'sem regressores'}]")

    df = carregar_base()
    diagnosticar(df)
    externos_usados = FEATURES_USADAS & GRUPOS_EXTERNOS
    externos = construir_externos(df, externos_usados) if externos_usados else {}
    series = montar_series_por_cd(df, externos)
    diagnosticar_externos(series)
    if "quinto_du" in FEATURES_USADAS:
        diagnosticar_quinto_du(series)
    feriados = gerar_feriados_brasileiros(df["data_pedido"].min(), df["data_pedido"].max())

    previsoes = pd.concat(
        [avaliar_cd(cd, s, feriados) for cd, s in series.items()], ignore_index=True
    )
    previsoes.to_csv(SAIDA / "previsoes_detalhadas.csv", index=False)

    por_cd_janela = metricas_por(previsoes, ["modelo", "cd_cd", "janela"])
    resumo = (
        por_cd_janela.groupby(["modelo", "cd_cd"])
        .agg(
            janelas=("janela", "count"),
            MAE_medio=("MAE", "mean"),
            WMAPE_medio=("WMAPE", "mean"),
            WMAPE_pior=("WMAPE", "max"),
            WMAPE_desvio=("WMAPE", "std"),
            Bias_medio=("Bias", "mean"),
        )
        .round(2)
    )
    resumo.to_csv(SAIDA / "resumo_por_cd.csv")
    tabela_cd = resumo["WMAPE_medio"].unstack("modelo")
    print("\n--- WMAPE médio por CD (linhas) x modelo (colunas) ---")
    print(tabela_cd.round(2).to_string())
    print("\n--- Melhor modelo por CD (menor WMAPE médio) ---")
    print(tabela_cd.idxmin(axis=1).to_string())

    geral = metricas_por(previsoes.assign(todos="todos"), ["modelo", "todos"]).drop(columns="todos")
    if M_BASE in set(geral["modelo"]):
        wmape_base = geral.loc[geral["modelo"] == M_BASE, "WMAPE"].iloc[0]
        geral["ganho_vs_base_%"] = (wmape_base - geral["WMAPE"]) / wmape_base * 100
    geral["features"] = geral["modelo"].map(
        lambda m: " + ".join(EXPERIMENTOS[m]) if EXPERIMENTOS[m] else "-"
    )
    geral = geral.sort_values("WMAPE").round(2)
    geral.to_csv(SAIDA / "comparacao_modelos.csv", index=False)
    print(
        "\n==================== Comparação (todas as janelas e CDs, "
        "ordenada por WMAPE) ===================="
    )
    print(geral.to_string(index=False))

    melhor = geral.iloc[0]["modelo"]
    plotar_comparacao_janelas(previsoes)
    plotar_metricas_por_horizonte(previsoes)
    plotar_bias_residuos(previsoes, melhor)

    print("\n--- Regressores descartados por serem constantes no treino ---")
    for (modelo, cd), regs in sorted(
        REG_DESCARTADOS.items(), key=lambda x: (x[0][0], str(x[0][1]))
    ):
        if regs:
            print(f"{modelo} | CD {cd}: {sorted(regs)}")
    print(f"\nArquivos salvos em {SAIDA}")


if __name__ == "__main__":
    main()
