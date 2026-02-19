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
# Identify layers by feature count
# ----------------------------

pref_shp = None  # 347
prov_shp = None  # 34

for shp in EXTRACT_DIR.rglob("*.shp"):
    if "__MACOSX" in str(shp):
        continue
    try:
        tmp = gpd.read_file(shp, engine="pyogrio")
        if len(tmp) == 347:
            pref_shp = shp
        elif len(tmp) == 34:
            prov_shp = shp
    except Exception:
        continue

if pref_shp is None:
    sys.exit("Prefecture layer (347 features) not found")
if prov_shp is None:
    sys.exit("Province layer (34 features) not found")

print("Prefecture layer:", pref_shp)
print("Province layer:", prov_shp)

pref = gpd.read_file(pref_shp, engine="pyogrio")
prov = gpd.read_file(prov_shp, engine="pyogrio")
pref = pref.to_crs(prov.crs)

# ----------------------------
# Detect key columns
# ----------------------------

prov_name_col = next((c for c in prov.columns if "name" in c.lower()), None)
if prov_name_col is None:
    print("Province columns:", list(prov.columns))
    sys.exit("Province name column not found")

city_col = next((c for c in pref.columns if "name" in c.lower()), None)
if city_col is None:
    print("Prefecture columns:", list(pref.columns))
    sys.exit("Prefecture name column not found")

print("Province name column:", prov_name_col)
print("City name column:", city_col)

# ----------------------------
# Spatial join: attach province_name to each prefecture polygon
# ----------------------------

joined = (
    gpd.sjoin(
        pref,
        prov[[prov_name_col, "geometry"]],
        how="left",
        predicate="intersects",
    )
    .rename(columns={prov_name_col: "province_name"})
    .dropna(subset=["province_name"])
)

# Remove HK/Macao/Taiwan (not in 33-node scope)
joined = joined[~joined["province_name"].isin(["香港", "澳门", "台湾"])]

# ----------------------------
# Load splits (must match shp prefecture names exactly)
# ----------------------------

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

hb_north = set(splits["Hebei"]["HEBEI_NORTH"])
hb_south = set(splits["Hebei"]["HEBEI_SOUTH"])
nm_east = set(splits["Inner Mongolia"]["NM_EAST"])
nm_west = set(splits["Inner Mongolia"]["NM_WEST"])

# ----------------------------
# Province CN -> EN mapping (non-split provinces only)
# ----------------------------

province_map = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "山西": "Shanxi",
    "山东": "Shandong",
    "上海": "Shanghai",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "西藏": "Tibet",
    "重庆": "Chongqing",
    "四川": "Sichuan",
}

# ----------------------------
# Assign 33-node labels
# ----------------------------

def assign_node(row) -> str:
    province = row["province_name"]
    city = row[city_col]

    if province == "河北":
        if city in hb_north:
            return "HEBEI_NORTH"
        if city in hb_south:
            return "HEBEI_SOUTH"
        return "河北"  # explicit leftover bucket for debugging

    if province == "内蒙古":
        if city in nm_east:
            return "NM_EAST"
        if city in nm_west:
            return "NM_WEST"
        return "内蒙古"  # explicit leftover bucket for debugging

    return province_map.get(province, province)

joined["node"] = joined.apply(assign_node, axis=1)

# ----------------------------
# Validate: Hebei/Inner Mongolia must be fully split
# ----------------------------

leftover_hb = (joined["node"] == "河北").sum()
leftover_nm = (joined["node"] == "内蒙古").sum()

if leftover_hb or leftover_nm:
    if leftover_hb:
        bad = sorted(joined.loc[joined["node"] == "河北", city_col].unique())
        print("Unmatched Hebei prefectures:", bad)
    if leftover_nm:
        bad = sorted(joined.loc[joined["node"] == "内蒙古", city_col].unique())
        print("Unmatched Inner Mongolia prefectures:", bad)
    sys.exit("ERROR: Split lists do not fully cover Hebei/Inner Mongolia prefectures")

# ----------------------------
# Dissolve to 33 nodes
# ----------------------------

gdf_33 = joined[["node", "geometry"]].dissolve(by="node", as_index=False)

# Ensure WGS84 for web maps
gdf_33 = gdf_33.to_crs("EPSG:4326")

# Very mild simplification
gdf_33["geometry"] = gdf_33["geometry"].simplify(
    0.05,  # increase tolerance slightly
    preserve_topology=True
)

gdf_33 = gdf_33.rename(columns={"node": "name"})

print("Nodes generated:", sorted(gdf_33["name"].unique()))
print("Count:", len(gdf_33))

# Hard stop if not 33
EXPECTED = set(province_map.values()) | {"HEBEI_NORTH", "HEBEI_SOUTH", "NM_EAST", "NM_WEST"}
GOT = set(gdf_33["name"])

if len(gdf_33) != 33:
    print("ERROR: Expected 33 nodes. Current:", len(gdf_33))
    print("Unexpected:", sorted(GOT - EXPECTED))
    print("Missing:", sorted(EXPECTED - GOT))
    sys.exit(1)

# ----------------------------
# Export
# ----------------------------

gdf_33.to_file(OUTPUT_PATH, driver="GeoJSON")
print("Generated:", OUTPUT_PATH)
