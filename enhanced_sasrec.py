"""
Enhanced SASRec - Complete Rewrite
====================================
Architecture:
  1. Feature-based product encoding: title (hash-trick), category_1/2/3 (embeddings),
     selling_price + product_rating (numeric projection) -> hidden_dim
  2. Runtime-injected action embeddings (view / click / hover / wishlist)
     added to product embedding BEFORE intent routing
  3. Multi-intent gating -- NO max-pooling, NO mean-pooling anywhere
     session_repr = sum_k (alpha_k * intent_k)
     where alpha_k is conditioned on the last product + its category embeddings
  4. Sequence-to-Sequence contrastive loss (InfoNCE on early/late halves)
     L_total = L_recommendation + lambda * L_contrastive
  5. Pseudo-session generator -- catalog-only training, category-consistent
     transitions, price-aware negatives, sessions >= 8 enforced
  6. Minimum sequence length = 8 HARD enforced at training AND inference
     Violations raise ValueError immediately -- no padding, no fallback

PyTorch only. Deterministic behaviour via seeded generators.
Every tensor shape documented in comments.
"""

import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MIN_SEQ_LEN: int = 8                       # Hard global minimum
TITLE_VOCAB_SIZE: int = 8192               # Hash-trick vocabulary
TITLE_MAX_WORDS: int = 16                  # Truncate titles longer than this
ACTION_VOCAB: List[str] = ["view", "click", "hover", "wishlist"]
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_VOCAB)}


# ---------------------------------------------------------------------------
# SEQUENCE LENGTH ENFORCEMENT
# ---------------------------------------------------------------------------

def enforce_min_seq_len(seq_len: int, context: str = "") -> None:
    """
    Hard error if seq_len < MIN_SEQ_LEN.
    Never pads, never reduces, never falls back -- raises immediately.
    """
    if seq_len < MIN_SEQ_LEN:
        raise ValueError(
            f"[HARD ERROR] Sequence length {seq_len} is below the required minimum "
            f"of {MIN_SEQ_LEN}. Context: '{context}'. "
            f"Do NOT pad. Do NOT reduce length. Fix the calling code to supply "
            f">= {MIN_SEQ_LEN} events."
        )


# ---------------------------------------------------------------------------
# TITLE TOKENISATION (hash trick -- no external NLP models)
# ---------------------------------------------------------------------------

def tokenise_title(title: str) -> List[int]:
    """
    Convert a product title to a list of vocabulary indices via MD5 hash trick.
    Returns at most TITLE_MAX_WORDS token ids (1-indexed; 0 is padding).
    No pretrained model, no vocabulary file.
    """
    words = str(title).lower().split()[:TITLE_MAX_WORDS]
    return [
        (int(hashlib.md5(w.encode()).hexdigest(), 16) % (TITLE_VOCAB_SIZE - 1)) + 1
        for w in words
    ]


# ---------------------------------------------------------------------------
# CATALOG FEATURE BUILDER -- preprocesses dataset.csv in memory
# ---------------------------------------------------------------------------

class CatalogFeatureBuilder:
    """
    Reads the product catalog (dataset.csv columns only) and builds
    all feature tensors required by the model. Everything stays in memory.

    Column mapping (authoritative):
        category_1, category_2, category_3, title, product_rating,
        selling_price, mrp, seller_name, seller_rating,
        description, highlights, image_links
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        self.df = df
        self.n_products: int = len(df)

        # Category vocabularies (0 = unknown / padding)
        self.cat1_vocab: Dict[str, int] = self._build_vocab(df["category_1"])
        self.cat2_vocab: Dict[str, int] = self._build_vocab(df["category_2"])
        self.cat3_vocab: Dict[str, int] = self._build_vocab(df["category_3"])

        # Price (selling_price column)
        self.price_raw: List[float] = self._parse_currency(df["selling_price"])
        price_max: float = max(self.price_raw) if max(self.price_raw) > 0 else 1.0
        self.price_norm: List[float] = [p / price_max for p in self.price_raw]

        # MRP (for discount calculation by laygen)
        self.mrp_raw: List[float] = self._parse_currency(df["mrp"])

        # Product rating (0-5)
        self.rating_raw: List[float] = (
            pd.to_numeric(df["product_rating"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 5.0)
            .tolist()
        )
        self.rating_norm: List[float] = [r / 5.0 for r in self.rating_raw]

        # Title token sequences, padded to TITLE_MAX_WORDS with 0
        self.title_ids: List[List[int]] = []
        for t in df["title"].fillna(""):
            toks = tokenise_title(str(t))[:TITLE_MAX_WORDS]
            padded = toks + [0] * (TITLE_MAX_WORDS - len(toks))
            self.title_ids.append(padded)

        # Per-product category id lists
        self.cat1_ids: List[int] = [
            self.cat1_vocab.get(str(v).strip(), 0) for v in df["category_1"].fillna("")
        ]
        self.cat2_ids: List[int] = [
            self.cat2_vocab.get(str(v).strip(), 0) for v in df["category_2"].fillna("")
        ]
        self.cat3_ids: List[int] = [
            self.cat3_vocab.get(str(v).strip(), 0) for v in df["category_3"].fillna("")
        ]

        # Product id mapping (1-indexed to match convention)
        self.pid_to_idx: Dict[int, int] = {i + 1: i for i in range(self.n_products)}
        self.idx_to_pid: Dict[int, int] = {i: i + 1 for i in range(self.n_products)}

        # Category-level product groups (for session generation)
        _cat1_col = df["category_1"].fillna("Unknown").str.strip()
        self.cat1_groups: Dict[str, List[int]] = {}
        for idx, cat in enumerate(_cat1_col.items()):
            row_idx, cat_name = cat
            pid = idx + 1
            self.cat1_groups.setdefault(cat_name, []).append(pid)

        # Price-tier groups (quartile bucketing, for negative sampling)
        prices_arr = np.array(self.price_raw)
        positive_prices = prices_arr[prices_arr > 0]
        if len(positive_prices) < 4:
            positive_prices = np.array([1.0, 2.0, 3.0, 4.0])
        self.price_quartiles: np.ndarray = np.percentile(positive_prices, [25, 50, 75])
        self.price_tier_groups: Dict[int, List[int]] = {t: [] for t in range(4)}
        for idx in range(self.n_products):
            tier = self._price_tier(self.price_raw[idx])
            self.price_tier_groups[tier].append(idx + 1)  # 1-indexed pid

    # -- helpers ---

    @staticmethod
    def _build_vocab(series: pd.Series) -> Dict[str, int]:
        unique = sorted({str(v).strip() for v in series.fillna("")})
        return {v: i + 1 for i, v in enumerate(unique)}   # 0 = unknown

    @staticmethod
    def _parse_currency(series: pd.Series) -> List[float]:
        out = []
        for v in series:
            try:
                out.append(max(0.0, float(str(v).replace("Rs.", "").replace("INR", "")
                                          .replace("\u20b9", "").replace(",", "").strip())))
            except (ValueError, AttributeError):
                out.append(0.0)
        return out

    def _price_tier(self, price: float) -> int:
        if price <= self.price_quartiles[0]:
            return 0
        elif price <= self.price_quartiles[1]:
            return 1
        elif price <= self.price_quartiles[2]:
            return 2
        return 3

    # -- public API ---

    @property
    def num_cat1(self) -> int:
        return len(self.cat1_vocab)

    @property
    def num_cat2(self) -> int:
        return len(self.cat2_vocab)

    @property
    def num_cat3(self) -> int:
        return len(self.cat3_vocab)

    def get_feature_tensors(
        self, product_ids: List[int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Build feature tensors for a flat list of product ids (1-indexed).
        Returns:
            word_ids  : [N, TITLE_MAX_WORDS]  long
            cat1_ids  : [N]                   long
            cat2_ids  : [N]                   long
            cat3_ids  : [N]                   long
            numerics  : [N, 2]                float  (norm_price, norm_rating)
        """
        indices = [self.pid_to_idx[pid] for pid in product_ids]

        word_ids = torch.tensor(
            [self.title_ids[i] for i in indices], dtype=torch.long, device=device
        )  # [N, TITLE_MAX_WORDS]

        cat1_t = torch.tensor(
            [self.cat1_ids[i] for i in indices], dtype=torch.long, device=device
        )  # [N]

        cat2_t = torch.tensor(
            [self.cat2_ids[i] for i in indices], dtype=torch.long, device=device
        )  # [N]

        cat3_t = torch.tensor(
            [self.cat3_ids[i] for i in indices], dtype=torch.long, device=device
        )  # [N]

        numerics = torch.tensor(
            [[self.price_norm[i], self.rating_norm[i]] for i in indices],
            dtype=torch.float,
            device=device,
        )  # [N, 2]

        return {
            "word_ids": word_ids,
            "cat1_ids": cat1_t,
            "cat2_ids": cat2_t,
            "cat3_ids": cat3_t,
            "numerics": numerics,
        }

    def get_last_product_cats(
        self, product_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return category ids for the last product in each sequence.
        product_ids : [B, T]
        returns     : cat1 [B], cat2 [B], cat3 [B]  long tensors on same device
        """
        last_pids: List[int] = product_ids[:, -1].tolist()
        indices = [self.pid_to_idx[pid] for pid in last_pids]
        dev = product_ids.device
        c1 = torch.tensor([self.cat1_ids[i] for i in indices], dtype=torch.long, device=dev)
        c2 = torch.tensor([self.cat2_ids[i] for i in indices], dtype=torch.long, device=dev)
        c3 = torch.tensor([self.cat3_ids[i] for i in indices], dtype=torch.long, device=dev)
        return c1, c2, c3


# ---------------------------------------------------------------------------
# TITLE ENCODER (no pretrained NLP)
# ---------------------------------------------------------------------------

class TitleEncoder(nn.Module):
    """
    Lightweight title encoder: hash-trick word embeddings -> masked mean-pool
    -> LayerNorm -> GELU -> linear projection.
    No external models. No vocabulary file.

    Input  : word_ids [B, TITLE_MAX_WORDS]  (0 = padding)
    Output : [B, out_dim]
    """

    def __init__(
        self,
        vocab_size: int = TITLE_VOCAB_SIZE,
        emb_dim: int = 64,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        nn.init.normal_(self.word_emb.weight, std=0.02)
        self.word_emb.weight.data[0].zero_()   # padding index = zero vector

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # word_ids : [B, T_title]
        mask = (word_ids != 0).float()                           # [B, T_title]
        emb = self.word_emb(word_ids)                            # [B, T_title, emb_dim]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)   # [B, 1]
        mean_emb = (emb * mask.unsqueeze(-1)).sum(dim=1) / counts  # [B, emb_dim]
        return self.proj(mean_emb)                               # [B, out_dim]


# ---------------------------------------------------------------------------
# PRODUCT FEATURE ENCODER
# ---------------------------------------------------------------------------

class ProductFeatureEncoder(nn.Module):
    """
    Fuses title, category_1/2/3, selling_price, product_rating into a
    single dense embedding vector of size hidden_dim.

    Inputs:
        word_ids  : [B, TITLE_MAX_WORDS]  long
        cat1_ids  : [B]                   long
        cat2_ids  : [B]                   long
        cat3_ids  : [B]                   long
        numerics  : [B, 2]                float  (norm_price, norm_rating)
    Output:
        prod_emb  : [B, hidden_dim]
    """

    def __init__(
        self,
        num_cat1: int,
        num_cat2: int,
        num_cat3: int,
        hidden_dim: int = 128,
        title_dim: int = 64,
        cat_dim: int = 16,
        num_feat_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.title_enc = TitleEncoder(
            vocab_size=TITLE_VOCAB_SIZE, emb_dim=64, out_dim=title_dim
        )
        self.cat1_emb = nn.Embedding(num_cat1 + 1, cat_dim, padding_idx=0)
        self.cat2_emb = nn.Embedding(num_cat2 + 1, cat_dim, padding_idx=0)
        self.cat3_emb = nn.Embedding(num_cat3 + 1, cat_dim, padding_idx=0)

        self.num_proj = nn.Sequential(
            nn.Linear(2, num_feat_dim),
            nn.LayerNorm(num_feat_dim),
            nn.GELU(),
        )

        in_dim = title_dim + 3 * cat_dim + num_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self,
        word_ids: torch.Tensor,   # [B, TITLE_MAX_WORDS]
        cat1_ids: torch.Tensor,   # [B]
        cat2_ids: torch.Tensor,   # [B]
        cat3_ids: torch.Tensor,   # [B]
        numerics: torch.Tensor,   # [B, 2]
    ) -> torch.Tensor:
        t = self.title_enc(word_ids)          # [B, title_dim]
        c1 = self.cat1_emb(cat1_ids)          # [B, cat_dim]
        c2 = self.cat2_emb(cat2_ids)          # [B, cat_dim]
        c3 = self.cat3_emb(cat3_ids)          # [B, cat_dim]
        n = self.num_proj(numerics)           # [B, num_feat_dim]
        fused = torch.cat([t, c1, c2, c3, n], dim=-1)
        # fused : [B, title_dim + 3*cat_dim + num_feat_dim]
        return self.fusion(fused)             # [B, hidden_dim]


# ---------------------------------------------------------------------------
# ACTION EMBEDDING LAYER
# ---------------------------------------------------------------------------

class ActionEmbeddingLayer(nn.Module):
    """
    Learnable embeddings for the 4 supported actions.
    Actions are NOT from dataset.csv -- they are runtime-injected from app.py.

    Vocabulary:  view(0) | click(1) | hover(2) | wishlist(3)

    Input  : action_ids [B, T]  long  (indices per ACTION_TO_IDX)
    Output : [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(ACTION_VOCAB), hidden_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    @staticmethod
    def encode_actions(actions: List[str]) -> List[int]:
        """
        Convert a list of action strings to integer ids.
        Raises KeyError for unknown actions.
        """
        return [ACTION_TO_IDX[a] for a in actions]

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        # action_ids : [B, T]
        return self.emb(action_ids)           # [B, T, hidden_dim]


# ---------------------------------------------------------------------------
# CAUSAL SELF-ATTENTION BLOCK
# ---------------------------------------------------------------------------

class CausalAttentionBlock(nn.Module):
    """
    Pre-norm causal multi-head self-attention + feed-forward block.
    Input / output shape: [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, D]
        T = x.shape[1]
        # Upper-triangular causal mask (True = ignore position)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )  # [T, T]

        normed = self.norm1(x)                                    # [B, T, D]
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out                                          # [B, T, D]
        x = x + self.ffn(self.norm2(x))                           # [B, T, D]
        return x


# ---------------------------------------------------------------------------
# INTENT GATING LAYER -- replaces ALL pooling operations
# ---------------------------------------------------------------------------

class IntentGatingLayer(nn.Module):
    """
    Computes the session representation WITHOUT any pooling.

        session_repr = sum_k (alpha_k * intent_k)

    alpha_k  : intent gating weights conditioned on the LAST product embedding
               AND its category_1/2/3 embeddings
    intent_k : soft-assignment weighted sum of sequence positions
               intent_k = sum_t (assignment_{t,k} * h_t)

    No max-pooling. No mean-pooling. No CLS average.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_intents: int = 4,
        cat_dim: int = 16,
        num_cat1: int = 6,
        num_cat2: int = 77,
        num_cat3: int = 300,
        gating_temperature: float = 0.5,  # Temperature for sharper intent distributions
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_intents = num_intents
        self.gating_temperature = gating_temperature

        # Learnable intent prototype vectors [K, D]
        self.intent_protos = nn.Parameter(torch.empty(num_intents, hidden_dim))
        nn.init.kaiming_uniform_(self.intent_protos, a=math.sqrt(5))
        
        # Learnable intent bias - initialized differently for diversity
        self.intent_bias = nn.Parameter(torch.randn(num_intents) * 0.5)

        # Category embeddings for gating conditioning
        self.cat1_cond = nn.Embedding(num_cat1 + 1, cat_dim, padding_idx=0)
        self.cat2_cond = nn.Embedding(num_cat2 + 1, cat_dim, padding_idx=0)
        self.cat3_cond = nn.Embedding(num_cat3 + 1, cat_dim, padding_idx=0)

        # Gating network: [seq_summary || cat1 || cat2 || cat3] -> K logits
        # seq_summary is mean of transformer outputs (gates on full sequence, not last item)
        cond_in = hidden_dim + 3 * cat_dim
        self.gating_net = nn.Sequential(
            nn.Linear(cond_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_intents),
        )

        # Per-position soft assignment: h_t -> K logits
        self.assign_proj = nn.Linear(hidden_dim, num_intents)

        # Intent-specific transformation after weighted aggregation
        self.intent_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for _ in range(num_intents)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self,
        seq_repr: torch.Tensor,       # [B, T, D]  transformer output
        seq_summary: torch.Tensor,    # [B, D]     mean-pooled sequence summary
        last_cat1: torch.Tensor,      # [B]         long
        last_cat2: torch.Tensor,      # [B]         long
        last_cat3: torch.Tensor,      # [B]         long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            session_repr   : [B, D]       final session embedding (no pooling)
            intent_vectors : [B, K, D]    K intent representations
            alpha_k        : [B, K]       intent gating weights
            assign_weights : [B, T, K]    per-position soft assignment weights
        """
        B, T, D = seq_repr.shape

        # Step 1: gating weights alpha_k conditioned on sequence summary + categories
        # Uses mean of full sequence instead of just last item for better context
        c1 = self.cat1_cond(last_cat1)   # [B, cat_dim]
        c2 = self.cat2_cond(last_cat2)   # [B, cat_dim]
        c3 = self.cat3_cond(last_cat3)   # [B, cat_dim]
        cond = torch.cat([seq_summary, c1, c2, c3], dim=-1)  # [B, D+3*cat_dim]
        alpha_logits = self.gating_net(cond) + self.intent_bias  # [B, K] + learned bias
        # Temperature scaling for sharper distributions
        alpha_k = F.softmax(alpha_logits / self.gating_temperature, dim=-1)  # [B, K]

        # Step 2: soft assignment of each position to each intent
        assign_logits = self.assign_proj(seq_repr)              # [B, T, K]
        # Softmax over T so weights sum to 1 across time dimension
        assign_weights = F.softmax(assign_logits, dim=1)        # [B, T, K]

        # Step 3: build intent vectors via weighted sum (NO pooling operator)
        intent_vectors_list: List[torch.Tensor] = []
        for k in range(self.num_intents):
            w_k = assign_weights[:, :, k]                       # [B, T]
            # Weighted sum of hidden states for intent k
            iv_k = torch.einsum("bt,btd->bd", w_k, seq_repr)   # [B, D]
            iv_k = self.intent_transforms[k](iv_k)              # [B, D]
            intent_vectors_list.append(iv_k)

        intent_vectors = torch.stack(intent_vectors_list, dim=1)  # [B, K, D]

        # Step 4: session_repr = sum_k alpha_k * intent_k
        # alpha_k unsqueezed: [B, K, 1] * intent_vectors [B, K, D] -> [B, K, D]
        session_repr = (alpha_k.unsqueeze(-1) * intent_vectors).sum(dim=1)  # [B, D]

        return session_repr, intent_vectors, alpha_k, assign_weights


# ---------------------------------------------------------------------------
# CONTRASTIVE INTENT LOSS (Seq2Seq InfoNCE)
# ---------------------------------------------------------------------------

class ContrastiveIntentLoss(nn.Module):
    """
    Sequence-to-Sequence contrastive learning over session intent embeddings.

    Positive pairs  : (early_half_i, late_half_i) from the SAME session
    Negative pairs  : (early_half_i, late_half_j) for j != i

    Uses InfoNCE (cross-entropy on temperature-scaled cosine similarities).
    Always computes -- NEVER disabled for small batches.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        early_repr: torch.Tensor,   # [B, D]
        late_repr: torch.Tensor,    # [B, D]
    ) -> torch.Tensor:
        """
        InfoNCE loss. Diagonal of the similarity matrix is the positive pair.
        """
        B = early_repr.shape[0]

        e_norm = F.normalize(early_repr, dim=-1)   # [B, D]
        l_norm = F.normalize(late_repr, dim=-1)    # [B, D]

        # sim[i,j] = cosine(early_i, late_j) / temperature
        sim = torch.matmul(e_norm, l_norm.T) / self.temperature  # [B, B]

        # Diagonal entries are the positive pairs
        labels = torch.arange(B, device=early_repr.device)       # [B]
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# PSEUDO-SESSION GENERATOR (catalog-only training)
# ---------------------------------------------------------------------------

class PseudoSessionGenerator:
    """
    Generates synthetic interaction sessions from the product catalog.
    All sessions live in memory only -- nothing written to disk.

    Rules:
        - Category-consistent transitions (all products share category_1,
          with occasional cross-category jumps for variety)
        - Price-aware negatives (negative targets from different price tier)
        - Minimum sequence length = MIN_SEQ_LEN (hard enforced)
    """

    def __init__(self, catalog: CatalogFeatureBuilder) -> None:
        self.catalog = catalog

    def generate_session(
        self,
        seq_len: int = 10,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Generate one pseudo-session of exactly seq_len events.
        Raises ValueError if seq_len < MIN_SEQ_LEN.
        Returns (product_ids_list, action_strings_list).
        """
        enforce_min_seq_len(seq_len, "PseudoSessionGenerator.generate_session")

        if rng is None:
            rng = random.Random()

        # Eligible categories have enough products
        eligible = [
            cat
            for cat, pids in self.catalog.cat1_groups.items()
            if len(pids) >= seq_len
        ]

        if not eligible:
            # Full catalog fallback: still enforces seq_len
            all_pids = list(range(1, self.catalog.n_products + 1))
            rng.shuffle(all_pids)
            chosen_pids = all_pids[:seq_len]
        else:
            cat = rng.choice(eligible)
            pool = self.catalog.cat1_groups[cat].copy()
            rng.shuffle(pool)
            chosen_pids = pool[:seq_len]

        # Realistic action distribution: view-heavy, wishlist-rare
        actions = rng.choices(
            ACTION_VOCAB,
            weights=[0.40, 0.30, 0.20, 0.10],
            k=seq_len,
        )
        return chosen_pids, actions

    def price_aware_negative(
        self,
        positive_pid: int,
        rng: Optional[random.Random] = None,
    ) -> int:
        """
        Sample a negative product from a DIFFERENT price tier.
        """
        if rng is None:
            rng = random.Random()

        idx = self.catalog.pid_to_idx[positive_pid]
        pos_tier = self.catalog._price_tier(self.catalog.price_raw[idx])

        other_tiers = [
            t
            for t in range(4)
            if t != pos_tier and self.catalog.price_tier_groups[t]
        ]

        neg_pool = (
            self.catalog.price_tier_groups[rng.choice(other_tiers)]
            if other_tiers
            else self.catalog.price_tier_groups[pos_tier]
        )

        neg_pid = rng.choice(neg_pool)
        attempts = 0
        while neg_pid == positive_pid and len(neg_pool) > 1 and attempts < 20:
            neg_pid = rng.choice(neg_pool)
            attempts += 1
        return neg_pid

    def generate_batch(
        self,
        batch_size: int = 16,
        seq_len: int = 12,
        seed: Optional[int] = None,
    ) -> List[Tuple[List[int], List[str]]]:
        """
        Generate a batch of pseudo-sessions.
        Raises ValueError if seq_len < MIN_SEQ_LEN.
        """
        enforce_min_seq_len(seq_len, "PseudoSessionGenerator.generate_batch")
        rng = random.Random(seed)
        return [self.generate_session(seq_len, rng) for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# MAIN MODEL: EnhancedSASRec
# ---------------------------------------------------------------------------

class EnhancedSASRec(nn.Module):
    """
    Enhanced Self-Attentive Sequential Recommendation.

    Architecture (per forward pass):
        product_ids [B,T] -> ProductFeatureEncoder -> prod_embs [B,T,D]
        action_ids  [B,T] -> ActionEmbeddingLayer  -> act_embs  [B,T,D]
                                                      | add (before intent routing)
                                                combined [B,T,D]
                                                      | + pos_emb + layer_norm
                                              CausalAttentionBlocks
                                                      |
                                              seq_repr [B,T,D]
                                                      |
                                  IntentGatingLayer (conditioned on last prod + cats)
                                                      |
                                              session_repr [B,D]
                                                      |
                                              score_proj -> scores [B,N]
    """

    def __init__(
        self,
        catalog: CatalogFeatureBuilder,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_intents: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cat_dim: int = 16,
        title_dim: int = 64,
        contrastive_temp: float = 0.07,
        lambda_contrastive: float = 0.1,
        lambda_entropy: float = 0.01,  # Entropy regularization weight (training only)
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.hidden_dim = hidden_dim
        self.num_intents = num_intents
        self.lambda_contrastive = lambda_contrastive
        self.lambda_entropy = lambda_entropy  # Encourages diverse intent usage

        # Product feature encoder (title + categories + numeric)
        self.product_enc = ProductFeatureEncoder(
            num_cat1=catalog.num_cat1,
            num_cat2=catalog.num_cat2,
            num_cat3=catalog.num_cat3,
            hidden_dim=hidden_dim,
            title_dim=title_dim,
            cat_dim=cat_dim,
            num_feat_dim=16,
        )

        # Action embeddings (runtime-injected, NOT from CSV)
        self.action_emb = ActionEmbeddingLayer(hidden_dim)

        # Positional encoding (learned, supports up to 512 positions)
        self.pos_emb = nn.Embedding(512, hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Input normalisation + dropout
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_drop = nn.Dropout(dropout)

        # Causal attention blocks
        self.blocks = nn.ModuleList([
            CausalAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Intent gating (replaces ALL pooling)
        self.intent_gate = IntentGatingLayer(
            hidden_dim=hidden_dim,
            num_intents=num_intents,
            cat_dim=cat_dim,
            num_cat1=catalog.num_cat1,
            num_cat2=catalog.num_cat2,
            num_cat3=catalog.num_cat3,
        )

        # Contrastive loss module
        self.contrastive_loss_fn = ContrastiveIntentLoss(temperature=contrastive_temp)

        # Output projection for item scoring
        self.score_proj = nn.Linear(hidden_dim, hidden_dim)

        # Product embedding cache (invalidated in training mode)
        self._prod_emb_cache: Optional[torch.Tensor] = None

    # -- internal helpers ---

    def _encode_products(
        self,
        product_ids: torch.Tensor,   # any shape ending with product ids
    ) -> torch.Tensor:
        """
        Encode product_ids through ProductFeatureEncoder.
        Preserves all leading dimensions.
        [B, T] -> [B, T, D],   [B] -> [B, D],   [N] -> [N, D]
        """
        orig_shape = product_ids.shape
        flat_ids: List[int] = product_ids.reshape(-1).tolist()
        feats = self.catalog.get_feature_tensors(flat_ids, product_ids.device)
        embs = self.product_enc(
            feats["word_ids"], feats["cat1_ids"],
            feats["cat2_ids"], feats["cat3_ids"], feats["numerics"]
        )  # [flat_N, D]
        return embs.reshape(*orig_shape, self.hidden_dim)

    def _run_transformer(
        self,
        product_ids: torch.Tensor,  # [B, T']
        action_ids: torch.Tensor,   # [B, T']
    ) -> torch.Tensor:
        """
        Shared forward through product encoder -> action add -> transformer.
        Does NOT enforce MIN_SEQ_LEN (used internally for contrastive halves).
        Returns seq_repr [B, T', D].
        """
        B, T = product_ids.shape
        prod_embs = self._encode_products(product_ids)   # [B, T, D]
        act_embs = self.action_emb(action_ids)            # [B, T, D]

        # Action embedding added to product embedding BEFORE intent routing
        x = prod_embs + act_embs                          # [B, T, D]

        positions = torch.arange(T, device=product_ids.device).unsqueeze(0)  # [1, T]
        x = x + self.pos_emb(positions)                   # [B, T, D]
        x = self.input_drop(self.input_norm(x))           # [B, T, D]

        for block in self.blocks:
            x = block(x)                                  # [B, T, D]
        return self.output_norm(x)                        # [B, T, D]

    def _encode_subsequence_contrastive(
        self,
        product_ids: torch.Tensor,  # [B, T']  T' may be < MIN_SEQ_LEN
        action_ids: torch.Tensor,   # [B, T']
    ) -> torch.Tensor:
        """
        Encode a subsequence for contrastive loss ONLY.
        Does NOT enforce MIN_SEQ_LEN (contrastive halves are allowed short).
        Returns last-position hidden state: [B, D].
        """
        seq_repr = self._run_transformer(product_ids, action_ids)  # [B, T, D]
        return seq_repr[:, -1, :]                                   # [B, D]

    def encode_session(
        self,
        product_ids: torch.Tensor,  # [B, T]
        action_ids: torch.Tensor,   # [B, T]
    ) -> Dict[str, torch.Tensor]:
        """
        Full session encoding. HARD ERROR if T < MIN_SEQ_LEN.

        Returns:
            session_repr         : [B, D]
            intent_vectors       : [B, K, D]
            alpha_k              : [B, K]
            assign_weights       : [B, T, K]
            seq_repr             : [B, T, D]
            prod_embs            : [B, T, D]
            per_product_contrib  : [B, T]
        """
        B, T = product_ids.shape
        enforce_min_seq_len(T, "EnhancedSASRec.encode_session")

        prod_embs = self._encode_products(product_ids)   # [B, T, D]
        act_embs = self.action_emb(action_ids)            # [B, T, D]

        x = prod_embs + act_embs                          # [B, T, D]
        positions = torch.arange(T, device=product_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions)                   # [B, T, D]
        x = self.input_drop(self.input_norm(x))

        seq_repr = x
        for block in self.blocks:
            seq_repr = block(seq_repr)                    # [B, T, D]
        seq_repr = self.output_norm(seq_repr)             # [B, T, D]

        # Intent gating conditioned on sequence summary (mean-pooled) + categories
        # Gates on full sequence context, not just last item
        seq_summary = seq_repr.mean(dim=1)                # [B, D]
        lc1, lc2, lc3 = self.catalog.get_last_product_cats(product_ids)

        session_repr, intent_vectors, alpha_k, assign_weights = self.intent_gate(
            seq_repr, seq_summary, lc1, lc2, lc3
        )
        # session_repr   : [B, D]
        # intent_vectors : [B, K, D]
        # alpha_k        : [B, K]
        # assign_weights : [B, T, K]

        # per_product_contrib[b,t] = sum_k alpha_k[b,k] * assign_weights[b,t,k]
        per_product_contrib = torch.einsum(
            "bk,btk->bt", alpha_k, assign_weights
        )  # [B, T]

        return {
            "session_repr": session_repr,
            "intent_vectors": intent_vectors,
            "alpha_k": alpha_k,
            "assign_weights": assign_weights,
            "seq_repr": seq_repr,
            "prod_embs": prod_embs,
            "per_product_contrib": per_product_contrib,
        }

    def _get_all_product_embs(self) -> torch.Tensor:
        """
        Return embedding matrix for all catalog products: [N, D].
        Cache is invalidated when model is in training mode.
        """
        if self.training:
            self._prod_emb_cache = None

        if self._prod_emb_cache is not None:
            return self._prod_emb_cache

        dev = next(self.parameters()).device
        all_pids = list(range(1, self.catalog.n_products + 1))
        feats = self.catalog.get_feature_tensors(all_pids, dev)
        with torch.no_grad():
            embs = self.product_enc(
                feats["word_ids"], feats["cat1_ids"],
                feats["cat2_ids"], feats["cat3_ids"], feats["numerics"]
            )  # [N, D]
        self._prod_emb_cache = embs
        return embs   # [N, D]

    # -- training forward ---

    def forward(
        self,
        product_ids: torch.Tensor,                        # [B, T]
        action_ids: torch.Tensor,                         # [B, T]
        target_pids: Optional[torch.Tensor] = None,       # [B]
        neg_pids: Optional[torch.Tensor] = None,          # [B]
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward. HARD ERROR if T < MIN_SEQ_LEN.

        Computes:
          L_total = L_recommendation + lambda * L_contrastive

        Returns all intermediate tensors and losses.
        """
        B, T = product_ids.shape
        enforce_min_seq_len(T, "EnhancedSASRec.forward")

        enc = self.encode_session(product_ids, action_ids)
        session_repr = enc["session_repr"]   # [B, D]

        # BPR recommendation loss
        if target_pids is not None and neg_pids is not None:
            pos_emb = self._encode_products(target_pids)            # [B, D]
            neg_emb = self._encode_products(neg_pids)               # [B, D]
            s = F.normalize(self.score_proj(session_repr), dim=-1)  # [B, D]
            pos_scores = (s * F.normalize(pos_emb, dim=-1)).sum(-1) # [B]
            neg_scores = (s * F.normalize(neg_emb, dim=-1)).sum(-1) # [B]
            L_rec = -F.logsigmoid(pos_scores - neg_scores).mean()
        else:
            L_rec = torch.tensor(0.0, device=product_ids.device)

        # Contrastive loss on early / late halves (allowed to be < MIN_SEQ_LEN)
        mid = T // 2
        early_repr = self._encode_subsequence_contrastive(
            product_ids[:, :mid], action_ids[:, :mid]
        )   # [B, D]
        late_repr = self._encode_subsequence_contrastive(
            product_ids[:, mid:], action_ids[:, mid:]
        )   # [B, D]
        L_contrast = self.contrastive_loss_fn(early_repr, late_repr)

        # Entropy regularization (training only) - encourages diverse intent usage
        # Maximize entropy of alpha_k (intent gating) to prevent collapse to single intent
        # H(alpha_k) = -sum(alpha_k * log(alpha_k))
        alpha_k = enc["alpha_k"]  # [B, K]
        eps = 1e-8
        alpha_entropy = -(alpha_k * torch.log(alpha_k + eps)).sum(dim=-1).mean()  # scalar
        # We want to maximize entropy, so subtract it from loss (or add negative)
        L_entropy = -alpha_entropy  # Negative because we maximize entropy

        L_total = L_rec + self.lambda_contrastive * L_contrast + self.lambda_entropy * L_entropy

        return {
            **enc,
            "loss_recommendation": L_rec,
            "loss_contrastive": L_contrast,
            "loss_entropy": L_entropy,
            "alpha_entropy": alpha_entropy,  # For monitoring
            "loss_total": L_total,
        }

    # -- inference ---

    def predict(
        self,
        product_ids: torch.Tensor,  # [B, T] or [T] single session
        action_ids: torch.Tensor,   # [B, T] or [T]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next-item scores for all catalog products.
        HARD ERROR if T < MIN_SEQ_LEN.

        Returns:
            scores               : [B, N]     cosine similarity to all products
            session_repr         : [B, D]
            intent_vectors       : [B, K, D]
            alpha_k              : [B, K]     intent gating weights
            assign_weights       : [B, T, K]  per-position assignment
            per_product_contrib  : [B, T]     contribution score per step
            ranked_pids          : [B, N]     product ids sorted desc by score
        """
        if product_ids.dim() == 1:
            product_ids = product_ids.unsqueeze(0)   # [1, T]
            action_ids = action_ids.unsqueeze(0)

        B, T = product_ids.shape
        enforce_min_seq_len(T, "EnhancedSASRec.predict")

        enc = self.encode_session(product_ids, action_ids)
        session_repr = enc["session_repr"]              # [B, D]

        all_embs = self._get_all_product_embs()         # [N, D]
        all_norm = F.normalize(all_embs, dim=-1)        # [N, D]
        s = F.normalize(self.score_proj(session_repr), dim=-1)  # [B, D]
        scores = torch.matmul(s, all_norm.T)            # [B, N]

        # Rank all products by score descending, return 1-indexed pids
        ranked_indices = torch.argsort(scores, dim=-1, descending=True)  # [B, N]
        ranked_pids = ranked_indices + 1                                  # [B, N]

        return {
            **enc,
            "scores": scores,          # [B, N]
            "ranked_pids": ranked_pids, # [B, N]
        }


# ---------------------------------------------------------------------------
# JOINT LOSS WRAPPER
# ---------------------------------------------------------------------------

class JointLoss(nn.Module):
    """
    L_total = L_recommendation + lambda * L_contrastive

    Accepts the output dict from EnhancedSASRec.forward() and summarises losses.
    """

    def __init__(self, lambda_contrastive: float = 0.1) -> None:
        super().__init__()
        self.lambda_contrastive = lambda_contrastive

    def forward(self, model_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        L_rec = model_out["loss_recommendation"]
        L_con = model_out["loss_contrastive"]
        L_total = L_rec + self.lambda_contrastive * L_con
        return {
            "loss_recommendation": L_rec,
            "loss_contrastive": L_con,
            "loss_total": L_total,
        }


# Alias for backward compatibility
JointTrainingLoss = JointLoss


# ---------------------------------------------------------------------------
# SEQUENCE PADDING UTILITY
# ---------------------------------------------------------------------------

def pad_sequence(seq: List[int], max_len: int) -> List[int]:
    """
    Pad or truncate a sequence to max_len.
    Left-pads with zeros if too short, truncates from left if too long.
    Returns a list of length max_len.
    
    WARNING: This does NOT bypass MIN_SEQ_LEN enforcement.
    Model methods still enforce MIN_SEQ_LEN independently.
    """
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def create_training_batch(
    sessions: List[Tuple[List[int], List[str]]],
    catalog: CatalogFeatureBuilder,
    device: torch.device,
    session_generator: PseudoSessionGenerator,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Alias for build_training_batch for backward compatibility.
    """
    return build_training_batch(sessions, catalog, device, session_generator, seed)


# ---------------------------------------------------------------------------
# TRAINING UTILITIES
# ---------------------------------------------------------------------------

def build_training_batch(
    sessions: List[Tuple[List[int], List[str]]],
    catalog: CatalogFeatureBuilder,
    device: torch.device,
    session_generator: PseudoSessionGenerator,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of (product_ids, action_strings) into training tensors.
    Input sequence = all but last event.  Target = last product.
    Generates price-aware negative for each sample.

    HARD ERROR if any session has < MIN_SEQ_LEN events, or if the resulting
    input subsequence has < MIN_SEQ_LEN events.

    Returns:
        product_ids : [B, T-1]  long
        action_ids  : [B, T-1]  long
        target_pids : [B]       long  positive target
        neg_pids    : [B]       long  price-aware negative
    """
    rng = random.Random(seed)
    all_pids, all_aids, all_targets, all_negs = [], [], [], []

    for pids, acts in sessions:
        enforce_min_seq_len(len(pids), "build_training_batch: full session")
        input_pids = pids[:-1]
        enforce_min_seq_len(len(input_pids), "build_training_batch: input subsequence")
        target = pids[-1]
        input_acts = acts[:-1]
        neg = session_generator.price_aware_negative(target, rng)

        all_pids.append(input_pids)
        all_aids.append(ActionEmbeddingLayer.encode_actions(input_acts))
        all_targets.append(target)
        all_negs.append(neg)

    product_tensor = torch.tensor(all_pids, dtype=torch.long, device=device)    # [B, T-1]
    action_tensor = torch.tensor(all_aids, dtype=torch.long, device=device)     # [B, T-1]
    target_tensor = torch.tensor(all_targets, dtype=torch.long, device=device)  # [B]
    neg_tensor = torch.tensor(all_negs, dtype=torch.long, device=device)        # [B]

    return {
        "product_ids": product_tensor,
        "action_ids": action_tensor,
        "target_pids": target_tensor,
        "neg_pids": neg_tensor,
    }


def run_training_epoch(
    model: EnhancedSASRec,
    optimizer: torch.optim.Optimizer,
    session_generator: PseudoSessionGenerator,
    catalog: CatalogFeatureBuilder,
    device: torch.device,
    batch_size: int = 16,
    seq_len: int = 12,
    num_batches: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    One epoch of catalog-only pseudo-session training.

    seq_len must be > MIN_SEQ_LEN so the input subsequence (seq_len - 1)
    is also >= MIN_SEQ_LEN.  HARD ERROR otherwise.
    """
    if seq_len <= MIN_SEQ_LEN:
        raise ValueError(
            f"[HARD ERROR] seq_len={seq_len} must be > MIN_SEQ_LEN={MIN_SEQ_LEN} "
            f"because the input subsequence after removing the target must still "
            f"be >= {MIN_SEQ_LEN}."
        )

    model.train()
    model._prod_emb_cache = None   # Invalidate stale cache

    totals = {"loss_recommendation": 0.0, "loss_contrastive": 0.0, "loss_total": 0.0}
    rng_base = random.Random(seed)

    for _ in range(num_batches):
        batch_seed = rng_base.randint(0, 2**31)
        sessions = session_generator.generate_batch(
            batch_size=batch_size, seq_len=seq_len, seed=batch_seed
        )
        batch = build_training_batch(sessions, catalog, device, session_generator, seed=batch_seed)

        optimizer.zero_grad()
        out = model(
            batch["product_ids"],
            batch["action_ids"],
            batch["target_pids"],
            batch["neg_pids"],
        )
        out["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss_recommendation"] += out["loss_recommendation"].item()
        totals["loss_contrastive"] += out["loss_contrastive"].item()
        totals["loss_total"] += out["loss_total"].item()

    n = float(num_batches)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print("=" * 70)
    print("EnhancedSASRec -- self-test")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    df = pd.read_csv(csv_path).head(300)
    print(f"Loaded {len(df)} products for testing.")

    catalog = CatalogFeatureBuilder(df)
    print(
        f"Catalog: {catalog.n_products} products, "
        f"cat1={catalog.num_cat1}, cat2={catalog.num_cat2}, cat3={catalog.num_cat3}"
    )

    device = torch.device("cpu")
    model = EnhancedSASRec(
        catalog=catalog,
        hidden_dim=64,
        num_blocks=2,
        num_intents=4,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    gen = PseudoSessionGenerator(catalog)

    # Test 1: short session must raise hard error
    print("\n[1] Testing MIN_SEQ_LEN enforcement ...")
    raised = False
    try:
        gen.generate_session(seq_len=3)
    except ValueError as e:
        raised = True
        print(f"   OK: short generate_session raised: {str(e)[:80]}")
    assert raised, "Expected ValueError for seq_len=3"

    # Test 2: normal forward pass
    print("\n[2] Testing forward pass with seq_len=10 ...")
    sessions = gen.generate_batch(batch_size=4, seq_len=10, seed=1)
    batch = build_training_batch(sessions, catalog, device, gen, seed=1)
    out = model(
        batch["product_ids"],
        batch["action_ids"],
        batch["target_pids"],
        batch["neg_pids"],
    )
    print(f"   session_repr     : {out['session_repr'].shape}")
    print(f"   intent_vectors   : {out['intent_vectors'].shape}")
    print(f"   alpha_k          : {out['alpha_k'].shape}")
    print(f"   assign_weights   : {out['assign_weights'].shape}")
    print(f"   per_prod_contrib : {out['per_product_contrib'].shape}")
    print(f"   loss_total       : {out['loss_total'].item():.4f}")
    print(f"   loss_rec         : {out['loss_recommendation'].item():.4f}")
    print(f"   loss_contrast    : {out['loss_contrastive'].item():.4f}")

    # Test 3: predict
    print("\n[3] Testing predict ...")
    model.eval()
    pids_t = torch.tensor(sessions[0][0][:9], dtype=torch.long).unsqueeze(0)
    aids_t = torch.tensor(
        ActionEmbeddingLayer.encode_actions(sessions[0][1][:9]),
        dtype=torch.long,
    ).unsqueeze(0)
    preds = model.predict(pids_t, aids_t)
    print(f"   scores          : {preds['scores'].shape}")
    print(f"   ranked_pids[:5] : {preds['ranked_pids'][0, :5].tolist()}")

    # Test 4: predict with T=7 must hard-error
    print("\n[4] Testing predict hard-error on T=7 ...")
    raised = False
    try:
        model.predict(pids_t[:, :7], aids_t[:, :7])
    except ValueError as e:
        raised = True
        print(f"   OK: predict raised: {str(e)[:80]}")
    assert raised, "Expected ValueError for T=7"

    print("\nAll tests passed.")