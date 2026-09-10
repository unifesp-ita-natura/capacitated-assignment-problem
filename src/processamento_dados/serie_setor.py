import pandas as pd

tabela_demanda1 = pd.read_csv("Unifesp_Demanda.csv")
tabela_demanda2 = pd.read_csv("Unifesp_Demanda_v2 - Unifesp_Demanda (4).csv.csv")
tabela_demanda3 = pd.read_csv("Unifesp_Demanda - Unifesp_Demanda.csv")

df_demanda = pd.concat([tabela_demanda1, tabela_demanda2, tabela_demanda3], ignore_index=True)

# Padronizando o código do setor antes da agregação.
df_demanda["cd_setor"] = (
    df_demanda["cd_setor"].astype(str).str.replace(r"^S", "", regex=True).str.strip()
)
df_demanda["data_pedido"] = pd.to_datetime(
    df_demanda["data_pedido"],
    format="mixed",
    dayfirst=True,
)
df_demanda["mes_pedido"] = df_demanda["data_pedido"].dt.month

df_demanda.to_csv("demanda_unificada.csv", index=False)

# verificando linhas duplicadas:
print(df_demanda.duplicated().any())

# tabela com total de volumes por ciclo por setor
serie_volume_setor = df_demanda.groupby(
    ["cd_setor", "aa_ciclo", "nm_ciclo"],
    as_index=False,
    dropna=False,
)["total_volumes_mascarado"].sum()

serie_volume_setor.to_csv("serie_volume_por_setor_e_ciclo.csv", index=False)

# tabela de com total de volumes de setor por dia
df_demanda["data_pedido"] = pd.to_datetime(df_demanda["data_pedido"])
calendario_completo = pd.date_range(
    start=df_demanda["data_pedido"].min(), end=df_demanda["data_pedido"].max(), freq="D"
)

dias_sem_pedido = calendario_completo[~calendario_completo.isin(df_demanda["data_pedido"])]


print(f"Dias sem pedidos: {len(dias_sem_pedido)}")

# os dias exatos sem pedidos
# df_ausentes = pd.DataFrame(
#   {'dias_sem_pedidos:': dias_sem_pedido.strftime('%d/%m/%Y')})

# df_ausentes.to_csv('dias_sem_pedidos.csv', index=False)

serie_volume_dia = df_demanda.groupby(
    ["cd_setor", "data_pedido"],
    as_index=False,
    dropna=False,
)["total_volumes_mascarado"].sum()

serie_volume_dia.to_csv("serie_volume_por_setor_e_dia.csv", index=False)

# tabela com total de volumes por ciclo por mês
serie_volume_mes = df_demanda.groupby(
    ["aa_ciclo", "nm_ciclo", "mes_pedido"],
    as_index=False,
    dropna=False,
)["total_volumes_mascarado"].sum()

serie_volume_mes.to_csv("serie_volume_por_ciclo_e_mes.csv", index=False)

# tabela com total de volumes de setor por mês

serie_volume_setor_mes = df_demanda.groupby(
    ["cd_setor", "aa_ciclo", "nm_ciclo", "mes_pedido"],
    as_index=False,
    dropna=False,
)["total_volumes_mascarado"].sum()

serie_volume_setor_mes.to_csv("serie_volume_por_setor_e_mes.csv", index=False)

# tabela com total de volumes por setor por cd_cd
serie_volume_cd = df_demanda.groupby(
    ["cd_cd"],
    as_index=False,
    dropna=False,
)["total_volumes_mascarado"].sum()

serie_volume_cd.to_csv("serie_volume_por_cd.csv", index=False)
