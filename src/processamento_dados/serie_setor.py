import pandas as pd

tabela_demanda1 = pd.read_csv("Unifesp_Demanda.csv")
tabela_demanda2 = pd.read_csv("Unifesp_Demanda_v2 - Unifesp_Demanda (4).csv.csv")
tabela_demanda3 = pd.read_csv("Unifesp_Demanda - Unifesp_Demanda.csv")
tabela_simulador = pd.read_excel("Simulador Bloco e Subbloco_v2.xlsx", sheet_name="Calendário FV")

df_demanda = pd.concat([tabela_demanda1, tabela_demanda2, tabela_demanda3], ignore_index=True)

# corrigindo a coluna setor da tabela do simulado
tabela_simulador["cd_setor_limpo"] = (
    tabela_simulador["COD SETOR"].astype(str).str.replace("S", "", regex=False).str.strip()
)

df_demanda["data_pedido"] = pd.to_datetime(df_demanda["data_pedido"], format="mixed", dayfirst=True)
df_demanda["mes_pedido"] = df_demanda["data_pedido"].dt.month
df_demanda["semana_pedido"] = df_demanda["data_pedido"].dt.isocalendar().week

# limpando espaços extras na coluna setor da tabela demanda
df_demanda["cd_setor_limpo"] = df_demanda["cd_setor"].astype(str).str.strip()

df_completo = pd.merge(df_demanda, tabela_simulador, on="cd_setor_limpo", how="left")

colunas_agrupadas = ["cd_setor", "aa_ciclo", "nm_ciclo", "mes_pedido", "semana_pedido"]

serie_volume = df_completo.groupby(colunas_agrupadas, as_index=False)[
    "total_volumes_mascarado"
].sum()

serie_volume.to_csv("serie_volume_setor.csv", index=False)

print("Script executado com sucesso!")
