"""
extract_hpc_dataset.py
======================
Standalone script — run from the terminal, not inside Jupyter.
Extracts FLOPs / MemByte / TransferByte for up to 200 deep-learning models
and saves results to  final_hpc_dataset.json  in the same directory.

Usage:
    python extract_hpc_dataset.py [--target 200] [--save-every 5]

Resume:  If final_hpc_dataset.json already exists, already-processed
         models are skipped automatically.
"""

import argparse, gc, json, logging, os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
from torch.export import export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "extraction.log"),
            mode="a", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────
def get_safe_numel(val):
    if val is None:               return 0
    if hasattr(val, "numel"):     return val.numel()
    if isinstance(val, (list, tuple)): return sum(get_safe_numel(v) for v in val)
    return 0

def get_safe_size(val):
    return get_safe_numel(val) * 4   # float32 = 4 bytes / element


def extract_dag_features(graph_module):
    dag = {}
    for node in graph_module.graph.nodes:
        flops, mem_byte, transfer_byte = 0, 0, 0
        out_val = node.meta.get("val")
        tgt     = node.target

        if node.op == "placeholder":
            dag[node.name] = {
                "flops": 0.0, "mem_byte": 0.0,
                "transfer_byte": float(get_safe_size(out_val)),
                "parents": [],
            }
            continue

        if node.op in ("call_module", "call_function"):
            for arg in node.args:
                if hasattr(arg, "meta") and "val" in arg.meta:
                    mem_byte += get_safe_size(arg.meta["val"])

        if out_val is not None:
            transfer_byte = get_safe_size(out_val)

        # FLOPs
        if tgt in (torch.ops.aten.mm.default,
                   torch.ops.aten.addmm.default,
                   torch.ops.aten.bmm.default):
            try:
                i1, i2 = (1, 2) if tgt == torch.ops.aten.addmm.default else (0, 1)
                a1 = node.args[i1].meta.get("val")
                a2 = node.args[i2].meta.get("val")
                if a1 is not None and a2 is not None:
                    s1, s2 = list(a1.shape), list(a2.shape)
                    B  = s1[0] if tgt == torch.ops.aten.bmm.default else 1
                    M  = s1[-2] if len(s1) >= 2 else 1
                    N  = s1[-1]
                    K  = s2[-1] if len(s2) >= 2 else s2[0]
                    flops = 2 * B * M * N * K
            except Exception: pass

        elif tgt == torch.ops.aten.convolution.default:
            try:
                w = node.args[1].meta.get("val")
                if w is not None and out_val is not None:
                    k_ops = 1
                    for d in list(w.shape)[1:]:
                        k_ops *= d
                    flops = 2 * get_safe_numel(out_val) * k_ops
            except Exception: pass

        elif "scaled_dot_product_attention" in str(tgt):
            try:
                q = node.args[0].meta.get("val")
                if q is not None and len(q.shape) >= 4:
                    B, H, L, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
                    flops = 2 * B * H * L * L * D
            except Exception: pass

        elif tgt == torch.ops.aten.einsum.default:
            try:
                if out_val is not None:
                    flops = get_safe_numel(out_val) * 2
            except Exception: pass

        elif out_val is not None:
            flops = get_safe_numel(out_val)

        dag[node.name] = {
            "flops":         float(flops),
            "mem_byte":      float(mem_byte),
            "transfer_byte": float(transfer_byte),
            "parents":       [p.name for p in node.all_input_nodes],
        }
    return dag


def safe_export(model, args, kwargs):
    try:
        ep = export(model, args=args, kwargs=kwargs, strict=True)
        return ep.graph_module, "strict"
    except Exception as e1:
        try:
            ep = export(model, args=args, kwargs=kwargs, strict=False)
            return ep.graph_module, "non-strict"
        except Exception as e2:
            raise RuntimeError(
                f"strict=True : {e1!s:.100}\n"
                f"strict=False: {e2!s:.100}"
            ) from e2


def get_vision_input(model_name):
    name_lo = model_name.lower()
    if any(k in name_lo for k in ("inception_v3", "inception_v4", "inceptionresnet")):
        return (torch.randn(1, 3, 299, 299),), {}
    # SwinV2 and MaxViT require 256×256; original Swin uses 224×224
    if any(k in name_lo for k in ("swinv2", "maxvit")):
        return (torch.randn(1, 3, 256, 256),), {}
    return (torch.randn(1, 3, 224, 224),), {}


def get_hf_input(config):
    from transformers import AutoConfig
    if hasattr(config, "use_cache"):
        config.use_cache = False

    model_type = getattr(config, "model_type", "").lower()
    vocab_size  = getattr(config, "vocab_size", 30522)
    seq_len     = 128

    if model_type == "longformer":
        window = getattr(config, "attention_window", [512])
        window = window[0] if isinstance(window, list) else window
        seq_len = max(window, 128)
    elif model_type in ("big_bird", "bigbird_roberta"):
        seq_len = 512
    elif model_type == "canine":
        vocab_size = 65536

    vocab_size = min(vocab_size, 250002)
    input_ids  = torch.randint(0, max(vocab_size, 2), (1, seq_len))
    return (input_ids,), {"return_dict": False}


# ──────────────────────────────────────────────────────────────────────
# Model candidate lists
# ──────────────────────────────────────────────────────────────────────
TIMM_CANDIDATES = [
    # ResNets
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "wide_resnet50_2", "wide_resnet101_2",
    "resnext50_32x4d", "resnext101_32x8d",
    # EfficientNets
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "efficientnet_lite0", "tf_efficientnet_b0",
    # ViT
    "vit_tiny_patch16_224", "vit_small_patch16_224",
    "vit_base_patch16_224", "vit_base_patch32_224",
    "vit_small_patch32_224",
    # DeiT
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit3_small_patch16_224", "deit3_base_patch16_224",
    # Swin
    "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    # Swin V2
    "swinv2_tiny_window8_256", "swinv2_small_window8_256",
    # ConvNeXt
    "convnext_tiny", "convnext_small", "convnext_base",
    # ConvNeXt V2
    "convnextv2_tiny", "convnextv2_small", "convnextv2_base",
    # MobileNets
    "mobilenetv2_100", "mobilenetv2_110d", "mobilenetv2_140",
    "mobilenetv3_small_050", "mobilenetv3_small_100",
    "mobilenetv3_large_100", "mobilenetv3_large_075",
    # DenseNets
    "densenet121", "densenet161", "densenet169", "densenet201",
    # RegNets
    "regnetx_002", "regnetx_004", "regnetx_008", "regnetx_016",
    "regnety_002", "regnety_004", "regnety_008", "regnety_016",
    # MLP-Mixer
    "mixer_s16_224", "mixer_b16_224",
    # ResMLP
    "resmlp_12_224", "resmlp_24_224", "resmlp_36_224",
    # CaiT
    "cait_xxs24_224", "cait_s24_224",
    # LeViT
    "levit_128s", "levit_128", "levit_192", "levit_256", "levit_384",
    # NFNets
    "nfnet_f0", "nfnet_f1", "eca_nfnet_l0", "eca_nfnet_l1",
    # SKNet
    "skresnet18", "skresnet34",
    # Res2Net
    "res2net50_26w_4s",
    # Inception
    "inception_v3",
    # Twins
    "twins_svt_small", "twins_svt_base",
    "twins_pcpvt_small", "twins_pcpvt_base",
    # PVT
    "pvt_v2_b0", "pvt_v2_b1", "pvt_v2_b2",
    # ConvMixer
    "convmixer_768_32", "convmixer_1024_20",
    # MaxViT
    "maxvit_tiny_tf_224",
    # HRNet
    "hrnet_w18_small", "hrnet_w18",
    # CrossViT
    "crossvit_tiny_240", "crossvit_small_240",
    # EfficientFormer
    "efficientformer_l1", "efficientformer_l3",
    # GENet
    "gernet_s", "gernet_m",
    # VGG
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    # Historic baselines
    "alexnet", "squeezenet1_0",
]

TV_CANDIDATES = [
    "alexnet",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    "squeezenet1_0", "squeezenet1_1",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "wide_resnet50_2",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "inception_v3", "googlenet",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x2_0",
    "mnasnet0_5", "mnasnet1_0",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4",
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf",
    "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf",
    "vit_b_16", "vit_b_32",
    "swin_t", "swin_s", "swin_b",
    "convnext_tiny", "convnext_small", "convnext_base",
]

HF_CANDIDATES = [
    # BERT family
    ("bert-base-uncased",                       "encoder"),
    ("bert-base-cased",                         "encoder"),
    ("bert-large-uncased",                      "encoder"),
    ("bert-base-multilingual-uncased",          "encoder"),
    ("bert-base-multilingual-cased",            "encoder"),
    ("prajjwal1/bert-tiny",                     "encoder"),
    ("prajjwal1/bert-mini",                     "encoder"),
    ("prajjwal1/bert-small",                    "encoder"),
    ("prajjwal1/bert-medium",                   "encoder"),
    # DistilBERT
    ("distilbert-base-uncased",                 "encoder"),
    ("distilbert-base-cased",                   "encoder"),
    ("distilbert-base-multilingual-cased",      "encoder"),
    # RoBERTa
    ("roberta-base",                            "encoder"),
    ("xlm-roberta-base",                        "encoder"),
    # ALBERT
    ("albert-base-v2",                          "encoder"),
    ("albert-large-v2",                         "encoder"),
    ("albert-xlarge-v2",                        "encoder"),
    # ELECTRA
    ("google/electra-small-discriminator",      "encoder"),
    ("google/electra-small-generator",          "encoder"),
    ("google/electra-base-discriminator",       "encoder"),
    ("google/electra-base-generator",           "encoder"),
    # GPT-2 (≤ medium)
    ("gpt2",                                    "decoder"),
    ("gpt2-medium",                             "decoder"),
    ("distilgpt2",                              "decoder"),
    # OPT
    ("facebook/opt-125m",                       "decoder"),
    ("facebook/opt-350m",                       "decoder"),
    # GPT-Neo 125 M
    ("EleutherAI/gpt-neo-125m",                 "decoder"),
    # DeBERTa
    ("microsoft/deberta-base",                  "encoder"),
    ("microsoft/deberta-v3-base",               "encoder"),
    ("microsoft/deberta-v3-small",              "encoder"),
    ("microsoft/deberta-v2-base",               "encoder"),
    # MobileBERT
    ("google/mobilebert-uncased",               "encoder"),
    # SqueezeBERT
    ("squeezebert/squeezebert-uncased",         "encoder"),
    # ConvBERT
    ("YituTech/conv-bert-base",                 "encoder"),
    ("YituTech/conv-bert-small",                "encoder"),
    ("YituTech/conv-bert-medium-small",         "encoder"),
    # FNet
    ("google/fnet-base",                        "encoder"),
    ("google/fnet-large",                       "encoder"),
    # MPNet
    ("microsoft/mpnet-base",                    "encoder"),
    # CamemBERT
    ("camembert-base",                          "encoder"),
    # Funnel Transformer
    ("funnel-transformer/small-base",           "encoder"),
    ("funnel-transformer/medium-base",          "encoder"),
    # TinyBERT
    ("huawei-noah/TinyBERT_General_4L_312D",    "encoder"),
    ("huawei-noah/TinyBERT_General_6L_768D",    "encoder"),
    # Domain-specific BERTs
    ("allenai/scibert_scivocab_uncased",        "encoder"),
    ("dmis-lab/biobert-v1.1",                   "encoder"),
    ("ProsusAI/finbert",                        "encoder"),
    ("nlpaueb/legal-bert-base-uncased",         "encoder"),
    ("microsoft/codebert-base",                 "encoder"),
    # Sentence Transformers
    ("sentence-transformers/all-MiniLM-L6-v2",         "encoder"),
    ("sentence-transformers/all-MiniLM-L12-v2",        "encoder"),
    ("sentence-transformers/paraphrase-MiniLM-L6-v2",  "encoder"),
    # Multilingual
    ("bert-base-german-cased",                  "encoder"),
    ("dccuchile/bert-base-spanish-wwm-cased",   "encoder"),
    ("cl-tohoku/bert-base-japanese",            "encoder"),
    # Tiny GPT
    ("sshleifer/tiny-gpt2",                     "decoder"),
    # OpenAI GPT-1
    ("openai-gpt",                              "decoder"),
    # XLNet
    ("xlnet-base-cased",                        "decoder"),
    # LayoutLM
    ("microsoft/layoutlm-base-uncased",         "encoder"),
    # Longformer
    ("allenai/longformer-base-4096",            "encoder"),
    # BigBird
    ("google/bigbird-roberta-base",             "encoder"),
    # CANINE
    ("google/canine-s",                         "encoder"),
    ("google/canine-c",                         "encoder"),
]


# ──────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5)
    args_cli = parser.parse_args()

    TARGET     = args_cli.target
    SAVE_EVERY = args_cli.save_every

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "final_hpc_dataset.json")
    failed_path  = os.path.join(script_dir, "failed_models.json")

    # Resume
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        log.info(f"Resuming — {len(dataset)} models already in dataset.")
    else:
        dataset = {}

    failed_models  = {}
    completed_keys = set(dataset.keys())
    success_count  = len(dataset)

    all_candidates = (
        [("timm",        name, None)      for name in TIMM_CANDIDATES] +
        [("torchvision", name, None)      for name in TV_CANDIDATES]   +
        [("hf",          name, arch_tag)  for name, arch_tag in HF_CANDIDATES]
    )

    log.info(f"Candidates: {len(all_candidates)}  Target: {TARGET}  Remaining: {TARGET - success_count}")

    import timm as _timm
    from torchvision import models as tv_models
    from transformers import AutoConfig, AutoModel

    # ── Warmup: first torch.export call has a one-time init cost/bug ──
    try:
        class _Tiny(torch.nn.Module):
            def forward(self, x): return x + 1
        torch.export.export(_Tiny(), args=(torch.ones(1),))
        log.info("torch.export warmup OK")
    except Exception as _e:
        log.info(f"torch.export warmup skipped: {_e!s:.60}")

    t0 = time.time()

    for idx, (source, model_name, arch_tag) in enumerate(all_candidates):
        if success_count >= TARGET:
            break

        key = f"{source}__{model_name.replace('/', '__')}"
        if key in completed_keys:
            success_count = len(dataset)
            continue

        gm = None
        try:
            # ── build model ──────────────────────────────────────────
            if source == "timm":
                model = _timm.create_model(model_name, pretrained=False).eval()
                inp_args, inp_kwargs = get_vision_input(model_name)

            elif source == "torchvision":
                fn = getattr(tv_models, model_name, None)
                if fn is None:
                    raise AttributeError(f"torchvision has no '{model_name}'")
                try:
                    model = fn(weights=None).eval()
                except TypeError:
                    model = fn(pretrained=False).eval()
                inp_args, inp_kwargs = get_vision_input(model_name)

            elif source == "hf":
                config = AutoConfig.from_pretrained(model_name)
                inp_args, inp_kwargs = get_hf_input(config)
                model = AutoModel.from_config(config).eval()

            else:
                raise ValueError(f"Unknown source: {source}")

            # ── export ────────────────────────────────────────────────
            with torch.no_grad():
                gm, export_mode = safe_export(model, inp_args, inp_kwargs)

            # ── extract ───────────────────────────────────────────────
            dag = extract_dag_features(gm)
            total_flops    = sum(v["flops"]         for v in dag.values())
            total_transfer = sum(v["transfer_byte"] for v in dag.values())

            dataset[key] = {
                "source":               source,
                "model_name":           model_name,
                "export_mode":          export_mode,
                "node_count":           len(dag),
                "total_flops":          total_flops,
                "total_transfer_bytes": total_transfer,
                "graph":                dag,
            }
            success_count += 1
            elapsed = time.time() - t0
            log.info(
                f"[{success_count:>3d}/{TARGET}] ✓ {key[:50]:<50s}  "
                f"nodes={len(dag):>4d}  GFLOPs={total_flops/1e9:>7.2f}  "
                f"mode={export_mode}  ({elapsed/60:.1f}m)"
            )

            if success_count % SAVE_EVERY == 0:
                with open(dataset_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f)
                log.info(f"  → checkpoint saved ({success_count} models)")

        except Exception as exc:
            err = str(exc).replace("\n", " ")[:150]
            failed_models[key] = err
            log.info(f"  [x] {key[:50]:<50s}  {err[:100]}")

        finally:
            # Safe cleanup — model/gm may be undefined if assignment failed early
            try:    del model
            except Exception: pass
            try:    del gm
            except Exception: pass
            gc.collect()
            try:
                import torch._dynamo
                torch._dynamo.reset()
            except Exception:
                pass

    # Final save
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_models, f, indent=2)

    total_time = time.time() - t0
    log.info("=" * 60)
    log.info(f"Finished in {total_time/60:.1f} min")
    log.info(f"Success: {success_count}  |  Failed: {len(failed_models)}")
    log.info(f"Dataset: {dataset_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
