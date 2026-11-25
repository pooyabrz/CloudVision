# Working with paired observations

Each row of `processed/manifest.csv` identifies a single UTC observation and its NPZ file. The `kelvin` array is the infrared input. The `mask` array is the CloudCast reference. The source product identifiers, time offsets and SHA-256 hashes are retained alongside each scene. The released tiles are generated on demand.

```python
from src.data import rows, scene
from src.schema import ROOT

directory = ROOT / "data/processed"
record = rows(directory)[0]
infrared, target = scene(directory, record)
print(record["timestamp"], infrared.shape, target.shape)
```

Run `python -m scripts.audit` to validate the collection. The [data card](../docs/data-card.md) documents how to reproduce the download, interpret the labels and import co-registered NetCDF files.
