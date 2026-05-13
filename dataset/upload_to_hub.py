# upload_to_hub.py — push final_hpc_dataset.json to HuggingFace Hub
# Run: python upload_to_hub.py

import json, os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import login

DATASET_PATH = os.path.join(os.path.dirname(__file__), "final_hpc_dataset.json")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")   # set via: export HF_TOKEN=hf_...
HF_REPO      = "nieche/DL-Architectural-DAGs-2026"

print(f"Loading {DATASET_PATH} ...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"  Loaded {len(data)} models  ({os.path.getsize(DATASET_PATH)/1e6:.1f} MB)")


models_rows = []
nodes_rows  = []

for m_id, content in data.items():
    models_rows.append({
        "model_id":             m_id,
        "source":               content["source"],
        "model_name":           content["model_name"],
        "export_mode":          content.get("export_mode", "strict"),
        "node_count":           int(content["node_count"]),
        "total_flops":          float(content["total_flops"]),
        "total_transfer_bytes": float(content["total_transfer_bytes"]),
    })
    for idx, (node_name, feat) in enumerate(content["graph"].items()):
        nodes_rows.append({
            "model_id":     m_id,
            "source":       content["source"],
            "model_name":   content["model_name"],
            "node_index":   idx,
            "node_name":    node_name,
            "flops":        float(feat["flops"]),
            "mem_byte":     float(feat["mem_byte"]),
            "transfer_byte":float(feat["transfer_byte"]),
            "parents":      feat["parents"],
        })

avg_nodes = len(nodes_rows) / max(len(models_rows), 1)
print(f"  models split : {len(models_rows):,} rows")
print(f"  nodes  split : {len(nodes_rows):,} rows  (avg {avg_nodes:.0f} nodes/model)")

models_features = Features({
    "model_id":             Value("string"),
    "source":               Value("string"),
    "model_name":           Value("string"),
    "export_mode":          Value("string"),
    "node_count":           Value("int32"),
    "total_flops":          Value("float64"),
    "total_transfer_bytes": Value("float64"),
})

nodes_features = Features({
    "model_id":     Value("string"),
    "source":       Value("string"),
    "model_name":   Value("string"),
    "node_index":   Value("int32"),
    "node_name":    Value("string"),
    "flops":        Value("float64"),
    "mem_byte":     Value("float64"),
    "transfer_byte":Value("float64"),
    "parents":      Sequence(Value("string")),
})


print("\nBuilding datasets ...")
models_ds = Dataset.from_list(models_rows, features=models_features)
nodes_ds  = Dataset.from_list(nodes_rows,  features=nodes_features)
print("models:", models_ds)
print("nodes :", nodes_ds)


CARD_BODY = """

# DL Architectural DAGs — HPC Features

A research dataset of **computation graphs (DAGs)** extracted from 200+ deep learning models
via `torch.export`, annotated with hardware-performance-counter (HPC) proxy features per node.

Designed for research on **operator scheduling, memory mapping, and performance prediction**
of DNN workloads on heterogeneous hardware (CPUs, GPUs, accelerators).

---

## Splits

| Split | Rows | Description |
|-------|-----:|-------------|
| `models` | ~200 | One row per model — metadata + aggregate stats |
| `nodes` | ~300 000 | One row per graph node — per-operation HPC features |

---

## Schema

### `models` split

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | string | Unique key: `{source}__{model_name}` |
| `source` | string | `timm` / `torchvision` / `hf` |
| `model_name` | string | Original model identifier |
| `export_mode` | string | `strict` or `non-strict` (`torch.export` mode used) |
| `node_count` | int32 | Total number of graph nodes |
| `total_flops` | float64 | Sum of FLOPs across all nodes |
| `total_transfer_bytes` | float64 | Sum of output-tensor bytes across all nodes |

### `nodes` split

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | string | Foreign key → `models.model_id` |
| `source` | string | `timm` / `torchvision` / `hf` |
| `model_name` | string | Model this node belongs to |
| `node_index` | int32 | Topological order index in the graph |
| `node_name` | string | ATen IR node name (e.g. `addmm`, `convolution`) |
| `flops` | float64 | Estimated FLOPs (MACs × 2) |
| `mem_byte` | float64 | Input-tensor bytes read by this node (weight-load proxy) |
| `transfer_byte` | float64 | Output-tensor bytes produced (data-volume / communication proxy) |
| `parents` | list[string] | Predecessor node names — encodes DAG edges |

---

## Models Covered

| Library | Count | Architectures |
|---------|------:|---------------|
| **timm** | ~100 | ResNet, EfficientNet, ViT, DeiT, Swin, ConvNeXt, MobileNet, DenseNet, RegNet, MLP-Mixer, ResMLP, CaiT, LeViT, NFNet, Twins, PVT, HRNet, CrossViT, EfficientFormer, VGG, … |
| **torchvision** | ~51 | AlexNet, VGG, SqueezeNet, ResNet, DenseNet, Inception, GoogLeNet, MobileNet, ShuffleNet, EfficientNet, RegNet, ViT-B, Swin-T/S/B, ConvNeXt, … |
| **HuggingFace Transformers** | ~50 | BERT family, DistilBERT, RoBERTa, ALBERT, ELECTRA, GPT-2, OPT, GPT-Neo, DeBERTa, MobileBERT, FNet, MPNet, TinyBERT, domain-BERTs, … |

---

## FLOPs Estimation Method

| Operation | Formula |
|-----------|---------|
| `mm` / `addmm` (Linear) | `2 × M × N × K` |
| `bmm` (batched matmul) | `2 × B × M × N × K` |
| `convolution` (any-D, grouped) | `2 × output_elements × (C_in/g × kernel_dims)` |
| `scaled_dot_product_attention` | `2 × B × H × L² × D` |
| `einsum` (DeBERTa disentangled) | `2 × output_elements` |
| element-wise (ReLU, Add, LayerNorm, …) | `output_elements` |

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("nieche/DL-Architectural-DAGs-2026")

# All nodes of ResNet-50 from timm
resnet_nodes = ds["nodes"].filter(lambda x: x["model_id"] == "timm__resnet50")

# Top-10 most FLOPs-intensive nodes across the whole dataset
df = ds["nodes"].to_pandas()
print(df.nlargest(10, "flops")[["model_id", "node_name", "flops"]])
```

---

## Citation

```bibtex
@dataset{dl_architectural_dags_2026,
  title  = {DL Architectural DAGs — HPC Features},
  year   = {2026},
  url    = {https://huggingface.co/datasets/nieche/DL-Architectural-DAGs-2026}
}
```
"""


print("\n" + "="*60)
print(f"  ÖZET")
print("="*60)
print(f"  Modeller   : {len(models_rows):,}")
print(f"  Node satır : {len(nodes_rows):,}")
print(f"  Hedef repo : {HF_REPO}")
print("="*60)
ans = input("\nHuggingFace'e yüklensin mi? [e/h]: ").strip().lower()
if ans not in ("e", "evet", "y", "yes"):
    print("İptal edildi.")
    exit(0)


print("\nLogging in to HuggingFace ...")
login(token=HF_TOKEN)

print(f"Pushing to {HF_REPO} ...")
# Push each split as a separate config (different schemas require separate configs)
models_ds.push_to_hub(HF_REPO, config_name="models", split="train", private=False)
print("  ✓ models split pushed")
nodes_ds.push_to_hub(HF_REPO, config_name="nodes", split="train", private=False)
print("  ✓ nodes split pushed")

# Update README: keep auto-generated YAML frontmatter, append our description
import re, tempfile
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)
readme_local = api.hf_hub_download(
    repo_id=HF_REPO, filename="README.md",
    repo_type="dataset", token=HF_TOKEN
)
with open(readme_local, "r", encoding="utf-8") as f:
    auto_readme = f.read()

# Extract only the YAML frontmatter (between first --- and second ---)
fm_match = re.match(r'^(---[\s\S]*?---)\n', auto_readme)
frontmatter = fm_match.group(1) if fm_match else ""
final_readme = frontmatter + "\n" + CARD_BODY

api.upload_file(
    path_or_fileobj=final_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Update dataset description",
)
print("  ✓ README updated")

print(f"\n✓ Done → https://huggingface.co/datasets/{HF_REPO}")
