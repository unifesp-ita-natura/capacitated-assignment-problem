from pathlib import Path

import pandas as pd

RAW_PATH = Path("../data/raw")
assert RAW_PATH.exists(), "Create raw folder and add the necessary files inside."

PROCESSED_PATH = Path("../data/processed")
PROCESSED_PATH.mkdir(exist_ok=True)

SIMULADOR_FILE = RAW_PATH / "Simulador Bloco e Subbloco_v2.xlsx"
assert SIMULADOR_FILE.exists(), "Missing Simulador file add to data/raw folder."

excel = pd.ExcelFile(SIMULADOR_FILE, engine="calamine")

df_calendario = excel.parse("Calendário FV")
df_calendario = df_calendario.dropna(how="all", axis=1)
df_calendario = df_calendario.drop(columns=["#"])
output_path = PROCESSED_PATH / "calendario_fv.csv"
df_calendario.to_csv(output_path, index=False)

df_demand = excel.parse("Base Demanda")
DEMAND_COLUMNS_NEEDED = [
    "data_pedido",
    "cd_setor",
    "cd_cd",
    "nm_ciclo",
    "aa_ciclo",
    "total_pedidos",
    "total_volumes",
    "total_itens",
]
df_demand = df_demand[DEMAND_COLUMNS_NEEDED]
df_demand["ciclo"] = df_demand["aa_ciclo"].astype(str) + df_demand["nm_ciclo"].astype(
    str
).str.zfill(2)
df_calendario["CICLOS"] = df_calendario["CICLOS"].astype(str)

n_before = len(df_demand)
df = df_demand.merge(
    df_calendario,
    left_on=["ciclo", "cd_setor"],
    right_on=["CICLOS", "COD SETOR"],
    how="left",
    indicator=True,
)
unmatched = df["_merge"].eq("left_only").sum()
print(f"{unmatched}/{n_before} demand rows unmatched to calendar ({unmatched / n_before:.1%})")

df = df_demand.merge(df_calendario, left_on=["ciclo", "cd_setor"], right_on=["CICLOS", "COD SETOR"])
output_file = PROCESSED_PATH / "demanda_with_calendar.csv"
df.to_csv(output_file, index=False)

df["Dt Abertura"] = pd.to_datetime(df["Dt Abertura"])
df["Dt Fechamento"] = pd.to_datetime(df["Dt Fechamento"])

df_level = (
    df.groupby(["cd_setor", "ciclo"], as_index=False)
    .agg(
        date=("Dt Abertura", "first"),
        cycle_duration=("Qtde dias", "first"),
        total_pedidos=("total_pedidos", "sum"),
        total_volumes=("total_volumes", "sum"),
        total_itens=("total_itens", "sum"),
    )
    .sort_values(by=["cd_setor", "date"])
    .reset_index(drop=True)
)
df_level.head()

output_file = PROCESSED_PATH / "demanda_level.csv"
df_level.to_csv(output_file, index=False)

df_shape = df.copy()
df_shape["data_pedido"] = pd.to_datetime(df_shape["data_pedido"])
df_shape["Dt Abertura"] = pd.to_datetime(df_shape["Dt Abertura"])
df_shape["relative_date"] = (df_shape["data_pedido"] - df_shape["Dt Abertura"]).dt.days / df[
    "Qtde dias"
]
# exclude orders outside of cycle (don't know if that's the best way to deal with it)
df_shape = df_shape[df_shape["relative_date"] <= 1]
# daily aggregation: collapses multiple distribution centers / records on the same day for a sector
# into single daily totals
df_shape = df_shape.groupby(
    ["cd_setor", "ciclo", "data_pedido", "relative_date"],
    as_index=False,
)[["total_pedidos", "total_volumes", "total_itens"]].sum()

# extract cycle totals (items, orders, volume)
df_shape[["ciclo_total_pedidos", "ciclo_total_volumes", "ciclo_total_itens"]] = df_shape.groupby(
    ["cd_setor", "ciclo"]
)[["total_pedidos", "total_volumes", "total_itens"]].transform("sum")

# share conversion: convert per day (items, orders, volume) to share of cycle total per sector
df_shape["share_pedidos"] = df_shape["total_pedidos"] / df_shape["ciclo_total_pedidos"]
df_shape["share_volumes"] = df_shape["total_volumes"] / df_shape["ciclo_total_volumes"]
df_shape["share_itens"] = df_shape["total_itens"] / df_shape["ciclo_total_itens"]

df_shape.head()
output_file = PROCESSED_PATH / "demanda_shape.csv"
df_shape.to_csv(output_file, index=False)
