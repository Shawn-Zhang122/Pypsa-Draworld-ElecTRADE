# Snapshot Schema v0.1

```json
{
  "schema_version": "0.1",
  "date": "YYYY-MM-DD",
  "model_version": "pypsa-x.y",
  "git_commit": "abcdef",
  "prices": {
    "Beijing": [24 values],
    "Hebei": [24 values]
  },
  "flows": [
    {
      "from": "Hebei",
      "to": "Beijing",
      "hour": 12,
      "mw": 320
    }
  ]
}
