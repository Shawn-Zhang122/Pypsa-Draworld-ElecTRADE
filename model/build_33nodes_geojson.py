import geopandas as gpd
import json
import zipfile
from pathlib import Path
import sys

# ----------------------------
# Paths
# ----------------------------
ZIP_PATH = Path("data/raw/中国标准地图-审图号GS(2020)4619号-shp格式.zip")
SPLITS_PATH = Path("config/splits_33nodes.json")
OUTPUT_PATH = Path("docs/geo/china_33nodes.geojson")
EXTRACT_DIR = Path("data/raw/extract")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Extract ZIP
# ----------------------------
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

# ----------------------------
# Identify layers by feature count (FIX 1: Ignore __MACOSX)
# ----------------------------
pref_shp = None  # Expected 347
prov_shp = None  # Expected 34

for shp in EXTRACT_DIR.rglob("*.shp"):
    # Avoid the common Mac extraction error
    if "__MACOSX" in str(shp) or shp.name.startswith("._"):
        continue
    try:
        tmp = gpd.read_file(shp, engine="pyogrio")
        if len(tmp) == 347:
            pref_shp = shp
        elif len(tmp) == 34:
            prov_shp = shp
    except Exception:
        continue

if pref_shp is None or prov_shp is None:
    sys.exit(f"Layers not found. Pref: {pref_shp}, Prov: {prov_shp}")

print(f"Prefecture: {pref_shp}")
print(f"Province: {prov_shp}")

# Load and align CRS
pref = gpd.read_file(pref_shp, engine="pyogrio")
prov = gpd.read_file(prov_shp, engine="pyogrio")
pref = pref.to_crs(prov.crs)

# Detect name columns
prov_name_col = next((c for c in prov.columns if "name" in c.lower()), None)
city_col = next((c for c in pref.columns if "name" in c.lower()), None)

# ----------------------------
# Spatial join: attach province_name to each prefecture
# ----------------------------
joined = (
    gpd.sjoin(pref, prov[[prov_name_col, "geometry"]], how="left", predicate="intersects")
    .rename(columns={prov_name_col: "province_cn"})
)

# ----------------------------
# Mapping Logic (FIX 2: Split Hebei/IM logic)
# ----------------------------
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

hb_north = set(splits["Hebei"]["HEBEI_NORTH"])
hb_south = set(splits["Hebei"]["HEBEI_SOUTH"])
nm_east = set(splits["Inner Mongolia"]["NM_EAST"])
nm_west = set(splits["Inner Mongolia"]["NM_WEST"])

province_map = {
    "北京市": "Beijing", "天津市": "Tianjin", "山西省": "Shanxi", "山东省": "Shandong",
    "上海市": "Shanghai", "江苏省": "Jiangsu", "浙江省": "Zhejiang", "安徽省": "Anhui",
    "福建省": "Fujian", "江西省": "Jiangxi", "河南省": "Henan", "湖北省": "Hubei",
    "湖南省": "Hunan", "广东省": "Guangdong", "广西壮族自治区": "Guangxi", "海南省": "Hainan",
    "贵州省": "Guizhou", "云南省": "Yunnan", "辽宁省": "Liaoning", "吉林省": "Jilin",
    "黑龙江省": "Heilongjiang", "陕西省": "Shaanxi", "甘肃省": "Gansu", "青海省": "Qinghai",
    "宁夏回族自治区": "Ningxia", "新疆维吾尔自治区": "Xinjiang", "西藏自治区": "Tibet",
    "重庆市": "Chongqing", "四川省": "Sichuan"
}

def assign_node(row) -> str:
    p = str(row["province_cn"])
    c = str(row[city_col])

    # 1. Check Specific Splits
    if "河北" in p:
        if c in hb_north: return "HEBEI_NORTH"
        if c in hb_south: return "HEBEI_SOUTH"
        return "HEBEI_SOUTH" # Fallback if name matches partially

    if "内蒙古" in p:
        if c in nm_east: return "NM_EAST"
        if c in nm_west: return "NM_WEST"
        return "NM_WEST" # Fallback

    # 2. General Mapping
    for cn, en in province_map.items():
        if cn in p or p in cn:
            return en
    return None

joined["node"] = joined.apply(assign_node, axis=1)

# ----------------------------
# Dissolve & Export (FIX 3: Projection for Web)
# ----------------------------
# Drop rows with no node mapping (HK/Macau/Taiwan/Errors)
final_gdf = joined.dropna(subset=["node"])

gdf_33 = final_gdf[["node", "geometry"]].dissolve(by="node", as_index=False)
gdf_33 = gdf_33.rename(columns={"node": "name"})

# IMPORTANT: Convert to Lat/Long (WGS84) for index.html/MapLibre
gdf_33 = gdf_33.to_crs("EPSG:4326")

# Simplify geometry slightly to keep geojson file size small
gdf_33["geometry"] = gdf_33["geometry"].simplify(0.01, preserve_topology=True)

gdf_33.to_file(OUTPUT_PATH, driver="GeoJSON")

# ----------------------------
# Format Checking Log
# ----------------------------
print("\n--- FORMAT CHECK ---")
print(f"Nodes Generated: {len(gdf_33)}")
for node in sorted(gdf_33['name'].unique()):
    cities = final_gdf[final_gdf['node'] == node][city_col].tolist()
    print(f"{node}: {', '.join(cities[:3])}...") # Printing first 3 cities per node

print(f"\nSuccessfully saved to: {OUTPUT_PATH}")