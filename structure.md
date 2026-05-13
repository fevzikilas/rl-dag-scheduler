

---

## Veri Akışı — Baştan Sona

---

### ADIM 0 — Ham Veri: data_loader.py

HuggingFace'ten indirilen her model bir `nx.DiGraph`:

```
node: "conv1"  →  attrs: {flops: 2e9, mem_byte: 4e6, transfer_byte: 800e3}
edge: "conv1" → "bn1"   (bağımlılık: bn1 çalışmadan önce conv1 bitmeli)
```

`compute_node_features()` her node için **5 sayı** üretir, hepsini min-max normalize eder:

```python
feat[i] = [flops_norm,        # hesaplama yükü
           mem_norm,           # bellek ayak izi
           transfer_norm,      # aktivasyon tensor boyutu
           depth_norm,         # topolojik derinlik (kritik path)
           in_degree_norm]     # kaç parent var
```

`build_edge_index()` kenarları COO formatına çevirir + her node'a self-loop ekler:
```python
edge_index: shape (2, E)
# [0,:] = source node indeksleri
# [1,:] = dest node indeksleri
```

**Dosya:** data_loader.py

---

### ADIM 1 — GNN Encoder: gnn_policy.py → `NodeEncoder`

hpc_env.py'nin `reset()` metodunda **episode başında bir kez** çalışır:

```python
# hpc_env.py satır ~100
feat, _ = compute_node_features(self.g)       # (n, 5)
ei      = build_edge_index(self.g, self.node_idx)  # (2, E)

with torch.no_grad():
    x_t  = torch.tensor(feat).to(self.device)
    ei_t = torch.tensor(ei).to(self.device)
    self.node_embeddings = self.encoder(x_t, ei_t).cpu().numpy()
    #                                              ↑ (n, 128)
```

`NodeEncoder.forward()` içinde ne olur:

```
(n, 5) → Linear(5→128) → ReLU   →   h₀: (n, 128)
                                          ↓
                                  SimpleMPLayer #1
                                  ┌─────────────────────────────┐
                                  │ her node için:              │
                                  │  agg = mean(komşu h değerleri)  │
                                  │  h_new = ELU(Linear([h, agg]))  │
                                  │  h = LayerNorm(h_new + h)   │  ← residual
                                  └─────────────────────────────┘
                                          ↓ (tekrar ×3 katman)
                                  h_final: (n, 128)
```

GAT versiyonu (eğer PyG kuruluysa) aynı şeyi yapar ama `agg` hesabında **attention ağırlığı** kullanır — her komşunun katkısı eşit değil, önemli komşular daha fazla ağırlık alır. Şu an fallback SimpleMPLayer kullanıyor (mean-aggregation).

**Sonuç:** Her node için 128 boyutlu bir embedding. Grafikteki tüm komşuluk yapısı bu 128 sayıya sıkıştırılmış.

**Dosya:** gnn_policy.py — `NodeEncoder.forward()`

---

### ADIM 2 — Her Adımdaki Observation: hpc_env.py → `_get_obs()`

`node_embeddings` matrisinde episode boyunca sadece **bir satır** kullanılır — o anki task'ın embedding'i:

```python
task_emb = self.node_embeddings[task_idx]   # (128,)  ← sıradaki task
proc_feat = [                                # (3×4 = 12,)
    proc_avail_time[0] / max_time,  proc0_flops, proc0_mem, proc0_load,
    proc_avail_time[1] / max_time,  proc1_flops, proc1_mem, proc1_load,
    proc_avail_time[2] / max_time,  proc2_flops, proc2_mem, proc2_load,
]
valid_mask = [True, False, True]             # (3,) GPU_0'ın memleri doldu mesela
```

**Dosya:** hpc_env.py — `_get_obs()`

---

### ADIM 3 — PPO Policy İçinde Feature Extraction: `HPCFeaturesExtractor`

Observation SB3'e ulaşınca `HPCFeaturesExtractor.forward()` çalışır:

```
task_emb  (128,) → task_net: Linear(128→128)→ReLU→Linear(128→128)→ReLU → (128,)
proc_feat  (12,) → proc_net: Linear(12→64)→ReLU→Linear(64→64)→ReLU    →  (64,)
                                                                  concat ↓
                                                               (192,)
                                                  combine: Linear(192→256)→ReLU
                                                               ↓
                                                           features: (256,)
```

Bu 256 boyutlu vektör PPO'nun policy ve value ağlarına girer.

**Dosya:** gnn_policy.py — `HPCFeaturesExtractor.forward()`

---

### ADIM 4 — PPO Policy/Value Kafaları: train.py

```python
policy_kwargs = dict(
    net_arch = dict(pi=[256, 128], vf=[256, 128])
)
```

```
features (256,)
    ├── policy head:  Linear(256→256)→ReLU→Linear(256→128)→ReLU → logits(3,)
    │                                                    ↓ valid_mask uygulanır
    │                                              Categorical(3) → action ∈ {0,1,2}
    │
    └── value head:   Linear(256→256)→ReLU→Linear(256→128)→ReLU → V(s): scalar
```

`valid_mask` burada devreye girer: memory sınırını ihlal eden processor'ların logit'i `-inf` yapılır, seçilemez.

---

### ADIM 5 — Environment Step: hpc_env.py → `step(action)`

Ajan `action=1` (GPU_0) derse:

```python
# 1. EFT hesapla
earliest_start = max(
    proc_avail_time[1],                     # GPU_0 ne zaman boşalıyor
    max(parent_finish + comm_overhead)       # tüm parent'lar ne zaman bitti
)
# comm_overhead = 0 eğer parent da GPU_0'daysa
#               = transfer_byte / 5GB/s  farklı processor'daysa

# 2. Commit
finish_time = earliest_start + flops / GPU_flops_cap
task_proc_assigned[task_idx] = 1
proc_avail_time[1] = finish_time

# 3. Reward
reward = -idle_created * 0.01

# 4. Children unlock
for child in successors(task):
    if all parents done → ready_queue.append(child)
```

**Dosya:** hpc_env.py — `step()`

---

### ADIM 6 — PPO Güncelleme: train.py

Her 1024 step sonunda:

```
rollout buffer doldu
    ↓
GAE ile advantage hesapla: A(s,a) = r + γV(s') - V(s)
    ↓
10 epoch minibatch gradient descent:
    policy loss = -min(r·A, clip(r, 0.8, 1.2)·A)   # PPO clip
    value  loss = MSE(V(s), r + γV(s'))
    entropy bonus = +0.01 * H(π)                     # exploration
    ↓
Ağırlıklar güncellendi → sonraki 1024 step
```

---

### Özet Şema

```
nx.DiGraph (172 model)
    │
    │ compute_node_features()
    ↓
feat: (n, 5)  +  edge_index: (2, E)
    │
    │ NodeEncoder.forward()   [episode başında 1 kez]
    ↓
node_embeddings: (n, 128)   ← tüm graph bilgisi burada
    │
    │ her step'te: node_embeddings[current_task_idx]
    ↓
task_emb: (128,)
    │
    │ + proc_feat: (12,)  +  valid_mask: (3,)
    │ HPCFeaturesExtractor.forward()
    ↓
features: (256,)
    │
    │ PPO policy head   (valid_mask ile maskeleme)
    ↓
action: {0, 1, 2}   → hangi processor
    │
    │ hpc_env.step(action)   → EFT hesapla → commit → reward
    ↓
next observation  →  buffer  →  PPO update (her 1024 step)
```