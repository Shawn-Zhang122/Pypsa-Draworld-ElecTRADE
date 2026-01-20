import json
from datetime import date

def export_snapshot(prices, flows, out_path):
    snapshot = {
        "schema_version": "0.1",
        "date": date.today().isoformat(),
        "model_version": "pypsa",
        "git_commit": "AUTO",
        "prices": prices,
        "flows": flows
    }

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
