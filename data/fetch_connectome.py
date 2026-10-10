"""Download the FlyWire adult fruit-fly connectome (v783) into data/raw/.

Sources:
  - Shiu et al. 2024 (Nature), github.com/philshiu/Drosophila_brain_model:
    connectivity with synapse counts and neurotransmitter-derived signs.
  - Schlegel et al. 2024 (Nature), github.com/flyconnectome/flywire_annotations:
    per-neuron annotations (super_class, cell_class, side, ...).

FlyWire data is CC BY-NC 4.0: anything derived from it is non-commercial.
"""
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent / "raw"

FILES = {
    "Connectivity_783.parquet":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "Completeness_783.csv":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "neuron_annotations.tsv":
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}


def download(name: str, url: str) -> None:
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {name}")
        return
    print(f"downloading {name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)


if __name__ == "__main__":
    main()
