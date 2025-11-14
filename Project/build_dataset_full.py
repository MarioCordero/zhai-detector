# Project/build_dataset_full.py
# ------------------------------------------------------------
# Genera dataset con TODOS los predios usando la estructura:
#
# zhai-detector/
# ├── Datasets/
# │   ├── raw/
# │   │   ├── CATASTRO_MAPA_PREDIAL/...
# │   │   └── ZONAS_HOMOGENEAS_ALAJUELA/...
# │   └── processed/
# └── Project/
#     └── build_dataset_full.py
# ------------------------------------------------------------

import os
import gc
import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree

# ===========================
# Paths base
# ===========================
HERE = Path(__file__).resolve()
BASE_DIR = HERE.parents[1]           # carpeta zhai-detector/
DATA_RAW = BASE_DIR / "Datasets" / "raw"
DATA_PROCESSED = BASE_DIR / "Datasets" / "processed"

PATH_PREDIOS = DATA_RAW / "CATASTRO_MAPA_PREDIAL" / "MAPA_CATASTRAL.shp"
PATH_ZH      = DATA_RAW / "ZONAS_HOMOGENEAS_ALAJUELA" / "ZONAS_HOMOGÉNEAS_2025.shp"

OUT_DIR      = DATA_PROCESSED
OUT_CSV      = OUT_DIR / "predios_multizh_FULL.csv"
OUT_PARQUET  = OUT_DIR / "predios_multizh_FULL.parquet"

BATCH = 5000
SIMPLIFY_TOL = 0.0

# ===========================
# Helpers
# ===========================
def compactness(area, perimeter):
    if perimeter == 0 or np.isnan(area) or np.isnan(perimeter):
        return np.nan
    return 4 * np.pi * area / (perimeter ** 2)

def strat_info(df, col="debe_actualizarse"):
    vc = df[col].value_counts(dropna=False)
    frac = (vc / vc.sum()).round(3)
    return pd.DataFrame({"count": vc, "frac": frac})

def dedup_first(df, cols, id_col):
    keep = [id_col] + [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates(subset=[id_col], keep="first").copy()

# ===========================
# 1) Carga shapefiles
# ===========================
print("Usando rutas:")
print("  Predios:", PATH_PREDIOS)
print("  ZH     :", PATH_ZH)

if not PATH_PREDIOS.exists():
    raise FileNotFoundError(f"No se encontró {PATH_PREDIOS}. ¿Descomprimiste el ZIP en Datasets/raw?")
if not PATH_ZH.exists():
    raise FileNotFoundError(f"No se encontró {PATH_ZH}. ¿Descomprimiste el ZIP en Datasets/raw?")

print("\nCargando shapefiles…")
pred_raw = gpd.read_file(PATH_PREDIOS)
zh       = gpd.read_file(PATH_ZH)

if pred_raw.crs != zh.crs:
    print("Reproyectando predios al CRS de ZH…")
    pred_raw = pred_raw.to_crs(zh.crs)

print(f"Predios (raw): {len(pred_raw)}  |  ZH: {len(zh)}")

# Elegir ID principal
id_name = "PRM_IDENTI" if "PRM_IDENTI" in pred_raw.columns else "PREDIO"
pred_raw[id_name] = pred_raw[id_name].astype(str)

dup = pred_raw[id_name].duplicated().sum()
if dup > 0:
    print(f"⚠️  Aviso: {dup} filas duplicadas por {id_name}. Se unificarán con dissolve.")

# ===========================
# 2) Geometría única por predio
# ===========================
print(f"Consolidando geometría única por {id_name} (dissolve)…")
pred_u = pred_raw[[id_name, "geometry"]].dissolve(by=id_name, as_index=False)
print(f"Predios únicos tras dissolve: {len(pred_u)}")

if SIMPLIFY_TOL > 0:
    print(f"Simplificando geometrías (tolerancia = {SIMPLIFY_TOL})…")
    pred_u["geometry"] = pred_u.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    zh["geometry"]     = zh.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

# ===========================
# 3) STRtree
# ===========================
print("Construyendo STRtree sobre ZH…")
zh_geom = list(zh["geometry"].values)
tree = STRtree(zh_geom)
zh_cod = zh["COD_ZONAH"].to_numpy()
wkb_to_cod = {g.wkb: c for g, c in zip(zh_geom, zh_cod)}

# ===========================
# 4) Conteo de ZH por predio (por lotes)
# ===========================
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_COUNTS = OUT_DIR / "_counts_tmp.csv"
if TMP_COUNTS.exists():
    TMP_COUNTS.unlink()

n = len(pred_u)
num_chunks = max(1, math.ceil(n / BATCH))
print(f"Contando ZH por predio en {num_chunks} lotes (BATCH={BATCH})…")

first = True
for i in range(num_chunks):
    start = i * BATCH
    end = min((i + 1) * BATCH, n)
    ch = pred_u.iloc[start:end].copy()
    print(f"  Lote {i+1}/{num_chunks}  (filas: {len(ch)})")

    rows = []
    for _, r in ch.iterrows():
        g = r.geometry
        try:
            candidates = tree.query(g, predicate="intersects")
        except Exception:
            rows.append({id_name: r[id_name], "n_zh_unicas": 0, "debe_actualizarse": 0})
            continue

        cods = set()
        if len(candidates) > 0 and isinstance(candidates[0], (np.integer, int)):
            for idx in candidates:
                cods.add(zh_cod[int(idx)])
        else:
            for geom in candidates:
                cods.add(wkb_to_cod.get(geom.wkb, None))
            cods.discard(None)

        n_zh = len(cods)
        rows.append({
            id_name: r[id_name],
            "n_zh_unicas": n_zh,
            "debe_actualizarse": int(n_zh >= 2)
        })

    pd.DataFrame(rows).to_csv(TMP_COUNTS, mode="a", header=first, index=False)
    first = False
    del rows, ch
    gc.collect()

print("✔ Conteos por predio guardados en:", TMP_COUNTS)

# ===========================
# 5) Features geométricas
# ===========================
pred_geom = pred_u[[id_name, "geometry"]].copy()
pred_geom["area_predio"]  = pred_geom.geometry.area.astype("float64")
pred_geom["perim_predio"] = pred_geom.geometry.length.astype("float64")
pred_geom["compactness"]  = pred_geom.apply(
    lambda r: compactness(r["area_predio"], r["perim_predio"]), axis=1
)
pred_geom = pred_geom.drop(columns="geometry")

# ===========================
# 6) Atributos admin / uso
# ===========================
admin_cols = [c for c in ["PRM_PROVIN", "PRM_CANTON", "PRM_DISTRI"] if c in pred_raw.columns]
uso_cols   = [c for c in ["USO"] if c in pred_raw.columns]

pred_admin = dedup_first(pred_raw, admin_cols, id_name)
pred_uso   = dedup_first(pred_raw, uso_cols, id_name)

# ===========================
# 7) Ensamble final
# ===========================
counts = pd.read_csv(TMP_COUNTS)

print("Combinando atributos…")
df = pred_geom.merge(pred_admin, on=id_name, how="left", validate="one_to_one")
df = df.merge(pred_uso, on=id_name, how="left", validate="one_to_one")
df = df.merge(counts, on=id_name, how="left", validate="one_to_one")

df["n_zh_unicas"]       = df["n_zh_unicas"].fillna(0).astype(int)
df["debe_actualizarse"] = df["debe_actualizarse"].fillna(0).astype(int)

for col in ["PRM_PROVIN", "PRM_CANTON", "PRM_DISTRI", "USO"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

df = df.rename(columns={id_name: "UID_PREDIO"})
df["UID_PREDIO"] = df["UID_PREDIO"].astype(str)

print("\nDistribución del target (FULL):")
print(strat_info(df, "debe_actualizarse"))

# ===========================
# 8) Guardar
# ===========================
df.to_csv(OUT_CSV, index=False)
print("\nCSV guardado ->", OUT_CSV, df.shape)

try:
    df.to_parquet(OUT_PARQUET, index=False)
    print("Parquet guardado ->", OUT_PARQUET)
except Exception as e:
    print("Parquet no guardado (opcional):", e)

if TMP_COUNTS.exists():
    TMP_COUNTS.unlink()

print("\n✅ Listo: dataset con TODOS los predios generado en Datasets/processed/")