"""
DisReq — Enhanced E-Commerce Recommendation System (Single File)
Combines: Flask backend + SASRec ML model + Price mapping + Popup UI
Files needed alongside this: enhanced_sasrec.py, dataset.csv
"""

from flask import Flask, render_template_string, jsonify, request, session, send_file
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time, random, hashlib, os
import requests as http_requests
from threading import Thread
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

from enhanced_sasrec import (
    EnhancedSASRec,
    JointLoss,
    JointTrainingLoss,
    CatalogFeatureBuilder,
    PseudoSessionGenerator,
    ActionEmbeddingLayer,
    pad_sequence,
    build_training_batch,
    run_training_epoch,
    MIN_SEQ_LEN,
    enforce_min_seq_len,
)

try:
    from laygen_pricemapping import (
        generate_product_poster_from_url,
        generate_content_aware_banner_from_url,
        set_vlm_analyzer,
        set_lm_reranker,
    )
    HAS_LAYGEN = True
except ImportError:
    HAS_LAYGEN = False
    generate_product_poster_from_url = None
    generate_content_aware_banner_from_url = None
    set_vlm_analyzer = None
    set_lm_reranker = None

app = Flask(__name__)
app.secret_key = 'disreq-secret-key-change-in-prod'


@app.before_request
def ensure_initialized():
    """Ensure product data is loaded before handling any request (WSGI-safe)."""
    global _app_initialized
    if not _app_initialized:
        _app_initialized = True
        try:
            load_product_data()
        except Exception as e:
            print(f"[ERROR] Failed to load dataset on first request: {e}")

# Minimal early declaration so startup initializer won't NameError
PRODUCTS_DF = None
_app_initialized = False

# ── Device Setup ───────────────────────────────────────────────────
# Force CPU to avoid CUDA initialization hangs (re-enable CUDA after testing)
device = torch.device('cpu')

# ── GitHub Models API config ───────────────────────────────────────
# Uses OpenAI-compatible inference endpoint on Azure via GitHub PAT
# Multiple tokens for fallback if one hits rate limits
_GITHUB_TOKENS = [
    os.getenv('GITHUB_TOKEN', ''),
    
]
_GITHUB_TOKENS = [t for t in _GITHUB_TOKENS if t and len(t) > 10]
GITHUB_MODELS_TOKEN = _GITHUB_TOKENS[0] if _GITHUB_TOKENS else ''
GITHUB_MODELS_ENDPOINT = 'https://models.inference.ai.azure.com/chat/completions'
# LLM for text generation (recommendations, explanations, ad copy)
GITHUB_LLM_MODEL = 'gpt-4o-mini'
# VLM for image content analysis (vision-capable model)
GITHUB_VLM_MODEL = 'gpt-4o'
GEMINI_ENABLED = bool(GITHUB_MODELS_TOKEN and len(GITHUB_MODELS_TOKEN) > 10)

def call_gemini_llm(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Call GitHub Models LLM with text prompt. Tries multiple tokens on failure."""
    global GITHUB_MODELS_TOKEN
    if not GEMINI_ENABLED:
        return ''
    for token in _GITHUB_TOKENS:
        try:
            payload = {
                'model': GITHUB_LLM_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
            print(f'[GitHub Models LLM] Sending request to {GITHUB_LLM_MODEL}')
            resp = http_requests.post(
                GITHUB_MODELS_ENDPOINT,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f'[GitHub Models LLM] Success - received {len(text)} chars')
            GITHUB_MODELS_TOKEN = token  # remember working token
            return text
        except Exception as e:
            print(f'[GitHub Models LLM] Error with token ...{token[-6:]}: {e}')
            continue
    return ''

def call_gemini_vlm(prompt: str, image_url: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
    """Call GitHub Models VLM with image and text prompt. Tries multiple tokens on failure."""
    global GITHUB_MODELS_TOKEN
    if not GEMINI_ENABLED:
        return ''
    for token in _GITHUB_TOKENS:
        try:
            payload = {
                'model': GITHUB_VLM_MODEL,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': image_url}},
                        {'type': 'text', 'text': prompt},
                    ],
                }],
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
            print(f'[GitHub Models VLM] Sending request to {GITHUB_VLM_MODEL}')
            resp = http_requests.post(
                GITHUB_MODELS_ENDPOINT,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            GITHUB_MODELS_TOKEN = token  # remember working token
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception as e:
            print(f'[GitHub Models VLM] Error with token ...{token[-6:]}: {e}')
            continue
    return ''

# ── Global State ───────────────────────────────────────────────────
PRODUCTS_DF = None
PRODUCTS_BY_ID = {}
PRODUCTS_CACHE = []
FEATURED_CACHE = []
USER_SESSIONS = {}
USER_CART = defaultdict(list)
USER_WISHLIST = defaultdict(list)
USER_PURCHASES = defaultdict(list)
MODEL = None
CATALOG = None  # CatalogFeatureBuilder instance
SESSION_GENERATOR = None  # PseudoSessionGenerator instance
POPULARITY_COUNTER = defaultdict(int)
TRAINING_PROGRESS = {'status':'idle','epoch':0,'total_epochs':0,'loss_rec':0.0,'loss_contrastive':0.0,'loss_total':0.0,'metrics':{},'batch_progress':0.0,'time_elapsed':0}
AUTO_TRAIN_LOCK = False
AUTO_TRAIN_THRESHOLD = 15
AUTO_TRAIN_MIN_HISTORY = MIN_SEQ_LEN  # Must have at least 8 events
PRODUCT_FIELDS = ['product_id','Product_Name','price','mrp','discount_pct','primary_image','combined_category','main_category']
ACTION_WEIGHTS = {'view':1.0,'cart':3.0,'wishlist':2.0,'buy':5.0}
ACTION_MAP = {'view': 'view', 'cart': 'click', 'wishlist': 'wishlist', 'buy': 'click'}  # Map app actions to model actions
import difflib

# ── Data Loading ───────────────────────────────────────────────────
def load_product_data():
    global PRODUCTS_DF, PRODUCTS_BY_ID, PRODUCTS_CACHE, FEATURED_CACHE, CATALOG, SESSION_GENERATOR
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    # Read all columns as strings to avoid type inference errors
    # Use default C engine (faster) with explicit encoding
    try:
        PRODUCTS_DF = pd.read_csv(csv_path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"  [WARN] UTF-8 read failed, trying latin-1: {e}")
        PRODUCTS_DF = pd.read_csv(csv_path, dtype=str, encoding='latin-1', on_bad_lines='skip')
    PRODUCTS_DF['product_id'] = range(1, len(PRODUCTS_DF)+1)
    
    # Build CatalogFeatureBuilder for the model
    CATALOG = CatalogFeatureBuilder(PRODUCTS_DF)
    SESSION_GENERATOR = PseudoSessionGenerator(CATALOG)
    print(f"  [OK] Built catalog: {CATALOG.n_products} products, cat1={CATALOG.num_cat1}, cat2={CATALOG.num_cat2}")
    
    # Map actual column names to expected names
    col_mapping = {
        'title': 'Product_Name',
        'category_1': 'main_category',
        'image_links': 'images'
    }
    for old, new in col_mapping.items():
        if old in PRODUCTS_DF.columns and new not in PRODUCTS_DF.columns:
            PRODUCTS_DF[new] = PRODUCTS_DF[old]
    
    # Parse price columns - strip currency symbols and commas
    def parse_price(x):
        if pd.isna(x) or x == '':
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        # Remove currency symbol ₹ and commas
        s = str(x).replace('₹', '').replace(',', '').strip()
        try:
            return float(s)
        except:
            return 0.0
    
    if 'selling_price' in PRODUCTS_DF.columns:
        PRODUCTS_DF['price'] = PRODUCTS_DF['selling_price'].apply(parse_price)
    elif 'price' not in PRODUCTS_DF.columns:
        PRODUCTS_DF['price'] = 0.0
    
    if 'mrp' in PRODUCTS_DF.columns:
        PRODUCTS_DF['mrp'] = PRODUCTS_DF['mrp'].apply(parse_price)
    else:
        PRODUCTS_DF['mrp'] = PRODUCTS_DF['price']
    
    # Build sub_category from category_2 and category_3 if present
    if 'sub_category' not in PRODUCTS_DF.columns:
        sub_parts = []
        for col in ['category_2', 'category_3']:
            if col in PRODUCTS_DF.columns:
                sub_parts.append(PRODUCTS_DF[col].fillna(''))
        if sub_parts:
            PRODUCTS_DF['sub_category'] = sub_parts[0]
            for part in sub_parts[1:]:
                PRODUCTS_DF['sub_category'] = PRODUCTS_DF['sub_category'] + ' > ' + part
        else:
            PRODUCTS_DF['sub_category'] = ''
    
    for col in ['Product_Name','main_category','sub_category']:
        if col in PRODUCTS_DF.columns: PRODUCTS_DF[col] = PRODUCTS_DF[col].fillna('')
    
    # Build combined category from category_1, category_2, category_3
    cat_parts = []
    for col in ['category_1', 'category_2', 'category_3']:
        if col in PRODUCTS_DF.columns:
            cat_parts.append(PRODUCTS_DF[col].fillna(''))
    if cat_parts:
        PRODUCTS_DF['combined_category'] = cat_parts[0]
        for part in cat_parts[1:]:
            PRODUCTS_DF['combined_category'] = PRODUCTS_DF['combined_category'] + ' > ' + part
    else:
        PRODUCTS_DF['combined_category'] = PRODUCTS_DF.get('main_category', '')
    
    # Ensure main_category is set
    if 'main_category' not in PRODUCTS_DF.columns or PRODUCTS_DF['main_category'].isna().all():
        if 'category_1' in PRODUCTS_DF.columns:
            PRODUCTS_DF['main_category'] = PRODUCTS_DF['category_1'].fillna('')
        else:
            PRODUCTS_DF['main_category'] = ''
    
    # Calculate discount percentage
    PRODUCTS_DF['discount_pct'] = 0
    mask = (PRODUCTS_DF['mrp'] > 0) & (PRODUCTS_DF['price'] > 0)
    PRODUCTS_DF.loc[mask, 'discount_pct'] = ((PRODUCTS_DF.loc[mask, 'mrp'] - PRODUCTS_DF.loc[mask, 'price']) / PRODUCTS_DF.loc[mask, 'mrp'] * 100).astype(int).clip(0, 100)
    
    if 'images' in PRODUCTS_DF.columns:
        def first_img(x):
            if pd.isna(x): return ''
            if isinstance(x,str):
                for c in ['[','"',"'"]: x=x.replace(c,'')
                x=x.replace(']','')
                return x.split(',')[0].strip()
            return ''
        PRODUCTS_DF['primary_image'] = PRODUCTS_DF['images'].apply(first_img)
    else:
        PRODUCTS_DF['primary_image'] = ''
    
    PRODUCTS_CACHE = PRODUCTS_DF[PRODUCT_FIELDS].to_dict('records')
    PRODUCTS_BY_ID = {p['product_id']:p for p in PRODUCTS_CACHE}
    FEATURED_CACHE = PRODUCTS_CACHE.copy()
    random.shuffle(FEATURED_CACHE)
    print(f"  [OK] Loaded {len(PRODUCTS_DF)} products")

# ── Recommendation Engine ──────────────────────────────────────────
def build_session_from_history(user_history):
    """
    Build a session (product_ids, actions) from user history.
    Returns (product_ids_list, action_strings_list, metadata).
    DOES NOT pad or enforce length - caller must handle MIN_SEQ_LEN.
    """
    product_ids = []
    action_strings = []
    categories = []
    main_categories = set()
    names = []
    price_range = []
    action_counts = {'view': 0, 'cart': 0, 'wishlist': 0, 'buy': 0}
    
    for h in user_history:
        if isinstance(h, dict):
            pid, act = h.get('pid'), h.get('action', 'view')
        else:
            try:
                pid, act = h
            except:
                continue
        
        if isinstance(pid, int) and pid in PRODUCTS_BY_ID:
            # Map app action to model action vocab
            model_action = ACTION_MAP.get(act, 'view')
            product_ids.append(pid)
            action_strings.append(model_action)
            action_counts[act] = action_counts.get(act, 0) + 1
            
            prod = PRODUCTS_BY_ID[pid]
            categories.append(prod.get('combined_category', ''))
            main_cat = prod.get('main_category', '')
            if main_cat:
                main_categories.add(main_cat)
            names.append(prod.get('Product_Name', ''))
            price = prod.get('price', 0)
            if price > 0:
                price_range.append(price)
    
    metadata = {
        'categories': categories,
        'main_categories': main_categories,
        'names': names,
        'price_range': price_range,
        'action_counts': action_counts,
        'viewed_pids': set(product_ids),
    }
    return product_ids, action_strings, metadata


def get_recommendations_for_user(user_history, top_k=6):
    """
    Generate personalized recommendations with multi-intent diversity.
    
    REQUIRES at least MIN_SEQ_LEN (8) events in user_history for model-based recs.
    Falls back to category-based if insufficient history or no model.
    """
    # Handle empty history
    if not user_history:
        featured_sample = FEATURED_CACHE[:top_k*3] if FEATURED_CACHE else []
        if featured_sample:
            random.shuffle(featured_sample)
            return [{'product_id': p['product_id'], 'product': p, 'reason': 'Featured for you', 'score': 50.0} for p in featured_sample[:top_k]]
        return []
    
    # Build session from history
    product_ids, action_strings, meta = build_session_from_history(user_history)
    
    if not product_ids:
        return [{'product_id': p['product_id'], 'product': p, 'reason': 'You might like this', 'score': 40.0} 
                for p in FEATURED_CACHE[:top_k]]
    
    categories = meta['categories']
    main_categories = meta['main_categories']
    names = meta['names']
    price_range = meta['price_range']
    viewed = meta['viewed_pids']
    
    avg_price = np.mean(price_range) if price_range else 0
    price_std = np.std(price_range) if len(price_range) > 1 else avg_price * 0.5
    
    # ─── FALLBACK: No trained model OR insufficient history ───
    # Model requires MIN_SEQ_LEN (8) events
    if MODEL is None or len(product_ids) < MIN_SEQ_LEN:
        recs = []
        
        # Strategy 1: Category-based recommendations
        category_recs = []
        for pid, prod in PRODUCTS_BY_ID.items():
            if pid in viewed:
                continue
            prod_cat = prod.get('main_category', '')
            prod_combined_cat = prod.get('combined_category', '')
            
            # Score based on category match and price similarity
            cat_score = 0
            if prod_cat in main_categories:
                cat_score = 40
            elif any(prod_combined_cat.startswith(c) for c in categories):
                cat_score = 30
            
            # Price similarity
            prod_price = prod.get('price', 0)
            price_score = 0
            if avg_price > 0 and prod_price > 0:
                price_diff = abs(prod_price - avg_price)
                if price_diff < price_std:
                    price_score = 20
                elif price_diff < price_std * 2:
                    price_score = 10
            
            # Popularity boost
            pop_score = min(20, POPULARITY_COUNTER.get(pid, 0) * 2)
            
            total_score = cat_score + price_score + pop_score
            if total_score > 0:
                reason_parts = []
                if cat_score >= 30:
                    reason_parts.append(f"Similar to {prod_cat}")
                if price_score > 0:
                    reason_parts.append("In your price range")
                if pop_score > 10:
                    reason_parts.append("Popular choice")
                
                # Add intermediate reasoning for fallback mode
                category_recs.append({
                    'product_id': pid,
                    'product': prod,
                    'reason': ' | '.join(reason_parts) if reason_parts else 'Recommended',
                    'score': total_score,
                    '_reasoning': {
                        'mode': 'fallback',
                        'min_seq_required': MIN_SEQ_LEN,
                        'current_seq_len': len(product_ids),
                        'product_sequence': [PRODUCTS_BY_ID.get(p, {}).get('Product_Name', '')[:30] for p in product_ids[-8:]],
                        'action_sequence': action_strings[-8:],
                    }
                })
        
        # Sort by score and add diversity
        category_recs.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure category diversity
        seen_main_cats = set()
        for rec in category_recs:
            main_cat = rec['product'].get('main_category', '')
            if len(recs) >= top_k:
                break
            if main_cat not in seen_main_cats or len(recs) < top_k // 2:
                recs.append(rec)
                seen_main_cats.add(main_cat)
        
        # Fill remaining slots with featured products
        while len(recs) < top_k and FEATURED_CACHE:
            for prod in FEATURED_CACHE:
                pid = prod['product_id']
                if pid not in viewed and not any(r['product_id'] == pid for r in recs):
                    recs.append({
                        'product_id': pid,
                        'product': prod,
                        'reason': 'You might also like',
                        'score': 30.0,
                        '_reasoning': {
                            'mode': 'fallback',
                            'min_seq_required': MIN_SEQ_LEN,
                            'current_seq_len': len(product_ids),
                        }
                    })
                    if len(recs) >= top_k:
                        break
            break
        
        return recs[:top_k]

    # ─── MODEL-BASED RECOMMENDATIONS ───
    # We have at least MIN_SEQ_LEN events, use the model
    try:
        MODEL.eval()
        with torch.no_grad():
            # Prepare tensors for model
            pids_tensor = torch.tensor([product_ids], dtype=torch.long, device=device)  # [1, T]
            aids_list = ActionEmbeddingLayer.encode_actions(action_strings)
            aids_tensor = torch.tensor([aids_list], dtype=torch.long, device=device)    # [1, T]
            
            # Get predictions with full intermediate outputs
            preds = MODEL.predict(pids_tensor, aids_tensor)
            scores = preds['scores'][0]              # [N]
            session_repr = preds['session_repr'][0]  # [D]
            intent_vectors = preds['intent_vectors'][0]  # [K, D]
            alpha_k = preds['alpha_k'][0]                # [K]
            assign_weights = preds['assign_weights'][0]  # [T, K]
            per_product_contrib = preds['per_product_contrib'][0]  # [T]
            
            K = intent_vectors.shape[0]
            T = len(product_ids)
            
            # Build intent labels from dominant categories
            intent_labels = []
            intent_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#c0392b']
            for k in range(K):
                wk = assign_weights[:, k].cpu().numpy()  # [T]
                top_idx = int(np.argmax(wk))
                if top_idx < len(product_ids):
                    top_pid = product_ids[top_idx]
                    if top_pid in PRODUCTS_BY_ID:
                        cat = PRODUCTS_BY_ID[top_pid].get('combined_category', '')
                        intent_labels.append(cat.split(' > ')[0] if ' > ' in cat else cat or f'Interest {k+1}')
                    else:
                        intent_labels.append(f'Interest {k+1}')
                else:
                    intent_labels.append(f'Interest {k+1}')

            # Get all product embeddings for per-intent scoring
            all_embs = MODEL._get_all_product_embs()  # [N, D]
            intent_norm = F.normalize(intent_vectors, dim=-1)  # [K, D]
            all_norm = F.normalize(all_embs, dim=-1)           # [N, D]
            per_intent_scores = torch.matmul(intent_norm, all_norm.T)  # [K, N]
            
            # Get top candidates
            num_candidates = min(top_k * 25, scores.shape[0])
            topk_scores, topk_indices = torch.topk(scores, num_candidates)
            candidates = []
            
            for score, idx in zip(topk_scores.cpu().numpy(), topk_indices.cpu().numpy()):
                idx = int(idx)
                pid = idx + 1
                if pid in viewed or pid not in PRODUCTS_BY_ID:
                    continue
                
                prod = PRODUCTS_BY_ID[pid]
                
                # Multi-factor scoring
                model_score = 50 * score.item() / (scores.max().item() + 1e-8)
                
                # Category match bonus
                cat_match = prod.get('combined_category', '') in categories
                main_cat_match = prod.get('main_category', '') in main_categories
                cat_score = 25 if cat_match else (15 if main_cat_match else 0)
                
                # Name similarity
                name_sim = max([difflib.SequenceMatcher(None, prod.get('Product_Name', ''), n).ratio() 
                               for n in names] + [0]) if names else 0
                name_score = 10 * name_sim
                
                # Price range fit
                prod_price = prod.get('price', 0)
                price_score = 0
                if avg_price > 0 and prod_price > 0:
                    price_diff = abs(prod_price - avg_price)
                    if price_diff < price_std:
                        price_score = 10
                    elif price_diff < price_std * 2:
                        price_score = 5
                
                # Discount bonus
                discount = prod.get('discount_pct', 0)
                discount_score = min(5, discount / 10)
                
                final_score = model_score + cat_score + name_score + price_score + discount_score
                
                # Intent analysis
                ips = per_intent_scores[:, idx].cpu().numpy()  # [K]
                ipcts = np.exp(ips) / (np.exp(ips).sum() + 1e-8)
                breakdown = sorted([
                    {'label': intent_labels[k], 'pct': round(float(ipcts[k]) * 100), 'color': intent_colors[k % len(intent_colors)]}
                    for k in range(K) if ipcts[k] > 0.12
                ], key=lambda x: x['pct'], reverse=True)
                dominant_k = int(np.argmax(ipcts))
                
                # Build reason string
                reason_parts = []
                if cat_match:
                    reason_parts.append("Category match")
                if name_sim > 0.3:
                    reason_parts.append("Similar product")
                if price_score > 0:
                    reason_parts.append("In budget")
                if discount >= 20:
                    reason_parts.append(f"{discount}% off")
                reason = ' | '.join(reason_parts) if reason_parts else f"Based on {intent_labels[dominant_k]}"
                
                # Build intermediate reasoning data (ALWAYS visible, never hidden)
                candidates.append({
                    'product_id': pid,
                    'product': prod,
                    'reason': reason,
                    'score': final_score,
                    'intent': dominant_k,
                    'intent_label': intent_labels[dominant_k],
                    'intents': breakdown,
                    '_reasoning': {
                        'mode': 'model',
                        'seq_len': T,
                        'product_sequence': [PRODUCTS_BY_ID.get(p, {}).get('Product_Name', '')[:30] for p in product_ids],
                        'action_sequence': action_strings,
                        'intent_vectors': intent_vectors.cpu().tolist(),
                        'intent_gating_weights': alpha_k.cpu().tolist(),
                        'per_product_contrib': per_product_contrib.cpu().tolist(),
                        'model_score': float(score),
                        'intent_labels': intent_labels,
                    }
                })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # ─── Multi-intent diversity selection ───
            if len(candidates) > top_k:
                selected = []
                seen_intents = set()
                seen_main_cats = set()
                
                # Phase 1: Pick best from each intent (diversity)
                for c in candidates:
                    if len(selected) >= top_k:
                        break
                    intent_key = c['intent']
                    main_cat = c['product'].get('main_category', '')
                    
                    # Prefer items from unseen intents and categories
                    if intent_key not in seen_intents:
                        selected.append(c)
                        seen_intents.add(intent_key)
                        seen_main_cats.add(main_cat)
                
                # Phase 2: Fill with best remaining (quality)
                for c in candidates:
                    if len(selected) >= top_k:
                        break
                    if c not in selected:
                        # Prefer different main categories
                        main_cat = c['product'].get('main_category', '')
                        if main_cat not in seen_main_cats or len(selected) < top_k - 1:
                            selected.append(c)
                            seen_main_cats.add(main_cat)
                
                # Phase 3: Just fill if still short
                for c in candidates:
                    if len(selected) >= top_k:
                        break
                    if c not in selected:
                        selected.append(c)
                
                candidates = selected

            # Build context for explanations
            recent_actions = []
            if user_history and isinstance(user_history[0], dict):
                for h in user_history[-8:]:
                    pid = h.get('pid')
                    act = h.get('action', 'view')
                    pname = PRODUCTS_BY_ID.get(pid, {}).get('Product_Name', '') if pid else ''
                    if pname:
                        recent_actions.append(f"{act}: {pname[:60]}")

            for rec in candidates[:top_k]:
                rec['_context'] = {
                    'recent_items': names[-5:],
                    'recent_categories': list(set(categories[-5:])),
                    'recent_actions': recent_actions,
                    'intent_label': rec.get('intent_label', ''),
                    'cat_match': 'Category match' in rec.get('reason', ''),
                    'score': rec.get('score', 0)
                }
            
            return candidates[:top_k]
            
    except Exception as e:
        print(f"[Recommendations] Model inference failed: {e}")
        # Fallback to category-based
        recs = []
        for prod in FEATURED_CACHE:
            pid = prod['product_id']
            if pid in viewed:
                continue
            if prod.get('main_category', '') in main_categories:
                recs.append({'product_id': pid, 'product': prod, 'reason': 'Similar category', 'score': 45.0})
            else:
                recs.append({'product_id': pid, 'product': prod, 'reason': 'You might like', 'score': 35.0})
            if len(recs) >= top_k:
                break
        return recs

# ── Training ───────────────────────────────────────────────────────
# SequenceDataset and collate_fn removed — we use PseudoSessionGenerator instead
# Training now uses run_training_epoch() from enhanced_sasrec.py

def evaluate_model(model, num_samples=200, k=10, eval_seed=None):
    """
    Evaluate model using pseudo-sessions from CATALOG.
    Sessions from generate_batch are List[Tuple[List[int], List[str]]].

    Metrics:
      - HR@K   (Hit Rate, a.k.a. Recall@K for single-item ground truth)
      - NDCG@K (Normalised Discounted Cumulative Gain)
      - MRR    (Mean Reciprocal Rank — full ranking, NO cutoff)
      - Intent Entropy (diversity of gating weights)
    """
    if CATALOG is None or SESSION_GENERATOR is None:
        return {'recall@10': 0.0, 'ndcg@10': 0.0, 'mrr': 0.0, 'intent_entropy': 0.0}

    was_training = model.training
    model.eval()
    hr_vals, ndcg_vals, mrr_vals, entropy_vals = [], [], [], []

    seq_len = MIN_SEQ_LEN + 2  # Must be strictly > MIN_SEQ_LEN so input subseq >= MIN_SEQ_LEN
    # Use a varied seed so different evaluation calls don't always produce identical results
    seed = eval_seed if eval_seed is not None else int(time.time()) % 100000

    with torch.no_grad():
        eval_sessions = SESSION_GENERATOR.generate_batch(
            batch_size=num_samples,
            seq_len=seq_len,
            seed=seed
        )

        for pids, acts in eval_sessions:
            # pids: List[int], acts: List[str]
            if len(pids) < MIN_SEQ_LEN + 1:
                continue

            input_pids = pids[:-1]     # all but last item
            target_pid = pids[-1]      # ground truth (1-indexed)
            input_acts = acts[:-1]

            if len(input_pids) < MIN_SEQ_LEN:
                continue

            pids_tensor = torch.tensor([input_pids], dtype=torch.long, device=device)
            # ActionEmbeddingLayer.encode_actions converts action strings to int indices
            aids = ActionEmbeddingLayer.encode_actions(input_acts)
            aids_tensor = torch.tensor([aids], dtype=torch.long, device=device)

            out = model.predict(pids_tensor, aids_tensor)
            scores = out['scores'][0].cpu().numpy()  # [N]

            # Compute intent entropy from gating weights
            alpha = out['alpha_k'][0].cpu().numpy()  # [K]
            eps = 1e-8
            entropy = float(-np.sum(alpha * np.log(alpha + eps)))
            entropy_vals.append(entropy)

            # target_pid is 1-indexed; scores are indexed 0..N-1 (catalog index = pid - 1)
            target_idx = target_pid - 1
            if target_idx < 0 or target_idx >= len(scores):
                continue  # out-of-range safety
            all_sorted = np.argsort(-scores)  # descending
            rank_arr = np.where(all_sorted == target_idx)[0]
            rank = int(rank_arr[0]) + 1 if len(rank_arr) > 0 else len(scores) + 1

            # HR@K (Hit Rate) — 1 if target in top-K, else 0
            hr_vals.append(1.0 if rank <= k else 0.0)

            # MRR — reciprocal rank over the FULL ranking (no K cutoff)
            mrr_vals.append(1.0 / rank)

            # NDCG@K — single relevant item: DCG/IDCG = (1/log2(rank+1)) / (1/log2(2))
            if rank <= k:
                ndcg_vals.append(1.0 / np.log2(rank + 1))
            else:
                ndcg_vals.append(0.0)

    # Restore original mode so training can continue unaffected
    if was_training:
        model.train()

    return {
        'recall@10': float(np.mean(hr_vals)) if hr_vals else 0.0,
        'ndcg@10': float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        'mrr': float(np.mean(mrr_vals)) if mrr_vals else 0.0,
        'intent_entropy': float(np.mean(entropy_vals)) if entropy_vals else 0.0,
    }


def train_model_background(epochs):
    """
    Background training using pseudo-sessions from catalog.
    Uses run_training_epoch() from enhanced_sasrec.
    """
    global MODEL, TRAINING_PROGRESS, AUTO_TRAIN_LOCK, CATALOG, SESSION_GENERATOR
    start = time.time()
    
    try:
        # Ensure CATALOG and SESSION_GENERATOR are initialized
        if CATALOG is None or SESSION_GENERATOR is None:
            TRAINING_PROGRESS['status'] = 'error: CATALOG not initialized'
            AUTO_TRAIN_LOCK = False
            return
        
        # Create new model instance
        model = EnhancedSASRec(
            catalog=CATALOG,
            hidden_dim=128,
            num_blocks=2,
            num_intents=4
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        best_recall = 0.0
        seq_len = MIN_SEQ_LEN + 4  # Must be > MIN_SEQ_LEN (8), so use 12
        
        for epoch in range(epochs):
            TRAINING_PROGRESS.update({
                'status': 'training',
                'epoch': epoch + 1,
                'total_epochs': epochs
            })
            
            # Run one training epoch using run_training_epoch from enhanced_sasrec
            losses = run_training_epoch(
                model=model,
                optimizer=optimizer,
                session_generator=SESSION_GENERATOR,
                catalog=CATALOG,
                device=device,
                batch_size=16,
                seq_len=seq_len,
                num_batches=50,
                seed=epoch * 1000
            )
            
            TRAINING_PROGRESS.update({
                'loss_recommendation': losses['loss_recommendation'],
                'loss_contrastive': losses['loss_contrastive'],
                'loss_total': losses['loss_total'],
                # Aliases for admin UI (JS reads loss_s2i, loss_s2s)
                'loss_s2i': losses['loss_recommendation'],
                'loss_s2s': losses['loss_contrastive'],
                'batch_progress': ((epoch + 1) / epochs) * 100,
                'time_elapsed': time.time() - start
            })
            
            # Evaluate every 2 epochs
            if (epoch + 1) % 2 == 0:
                m = evaluate_model(model)
                TRAINING_PROGRESS['metrics'] = m
                if m['recall@10'] > best_recall:
                    best_recall = m['recall@10']
                    torch.save(model.state_dict(), 'best_model.pth')
        
        # Final evaluation and save
        m = evaluate_model(model)
        TRAINING_PROGRESS.update({'metrics': m, 'status': 'completed'})
        MODEL = model
        AUTO_TRAIN_LOCK = False
        torch.save(model.state_dict(), 'final_model.pth')
        print("Training complete!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        TRAINING_PROGRESS['status'] = f'error: {str(e)}'
        AUTO_TRAIN_LOCK = False

# ── Session & Interaction ──────────────────────────────────────────
def get_session_user_id():
    if 'user_id' not in session:
        session['user_id'] = hashlib.sha256(f"{time.time()}-{random.random()}".encode()).hexdigest()[:16]
        session.modified=True
    return session['user_id']

def record_interaction(user_id, product_id, action='view'):
    if product_id is None: return
    pid=int(product_id)
    USER_SESSIONS.setdefault(user_id,[]).append({'pid':pid,'action':action,'ts':time.time()})
    POPULARITY_COUNTER[pid]+=1
    if action=='cart' and pid not in USER_CART[user_id]: USER_CART[user_id].append(pid)
    elif action=='wishlist' and pid not in USER_WISHLIST[user_id]: USER_WISHLIST[user_id].append(pid)
    elif action=='buy': USER_PURCHASES[user_id].append(pid); USER_CART[user_id] = [x for x in USER_CART[user_id] if x!=pid]
    global AUTO_TRAIN_LOCK, TRAINING_PROGRESS
    if (len(USER_SESSIONS.get(user_id,[]))>=AUTO_TRAIN_THRESHOLD and
            TRAINING_PROGRESS.get('status','idle') in ['idle','completed'] and not AUTO_TRAIN_LOCK):
        AUTO_TRAIN_LOCK=True
        TRAINING_PROGRESS={'status':'training','epoch':0,'total_epochs':10,'loss_s2i':0.0,'loss_s2s':0.0,'loss_entropy':0.0,'loss_total':0.0,'metrics':{},'batch_progress':0.0,'time_elapsed':0}
        Thread(target=train_model_background,args=(10,),daemon=True).start()

# ── LLM helpers ────────────────────────────────────────────────────
def llm_rerank_candidates(recs, user_history):
    """Rerank recommendations using LLM with product-specific context."""
    if len(recs) <= 3:
        return recs
    
    frozen = recs[:2]  # Keep top 2 from ML model
    to_rerank = recs[2:]
    
    # Analyze user behavior
    viewed_items = []
    carted_items = []
    bought_items = []
    categories = []
    price_points = []
    
    if user_history and isinstance(user_history[0], dict):
        for h in user_history[-12:]:
            pid = h.get('pid')
            act = h.get('action', 'view')
            if pid and pid in PRODUCTS_BY_ID:
                prod = PRODUCTS_BY_ID[pid]
                name = prod.get('Product_Name', '')[:45]
                cat = prod.get('main_category', '')
                price = prod.get('price', 0)
                
                if act == 'buy':
                    bought_items.append(name)
                elif act == 'cart':
                    carted_items.append(name)
                elif act == 'view':
                    viewed_items.append(name)
                
                if cat and cat not in categories:
                    categories.append(cat)
                if price > 0:
                    price_points.append(price)
    
    avg_price = int(np.mean(price_points)) if price_points else 0
    
    # Build context string
    context_parts = []
    if bought_items:
        context_parts.append(f"PURCHASED: {', '.join(bought_items[-3:])}")
    if carted_items:
        context_parts.append(f"IN CART: {', '.join(carted_items[-3:])}")
    if viewed_items:
        context_parts.append(f"VIEWED: {', '.join(viewed_items[-4:])}")
    if categories:
        context_parts.append(f"INTERESTS: {', '.join(categories[:4])}")
    if avg_price:
        context_parts.append(f"BUDGET: ~₹{avg_price:,}")
    
    # Build candidate list with details
    candidate_lines = []
    for i, rec in enumerate(to_rerank):
        p = rec['product']
        name = p.get('Product_Name', 'Unknown')[:50]
        cat = p.get('main_category', '')
        price = p.get('price', 0)
        disc = p.get('discount_pct', 0)
        reason = rec.get('reason', '')[:30]
        
        line = f"{i}: {name} | {cat} | ₹{int(price):,}"
        if disc >= 15:
            line += f" ({disc}% OFF)"
        candidate_lines.append(line)
    
    prompt = f"""You are an e-commerce recommendation optimizer. Reorder candidates to maximize purchase likelihood.

USER PROFILE:
{chr(10).join(context_parts)}

CANDIDATES TO REORDER:
{chr(10).join(candidate_lines)}

RANKING CRITERIA (priority order):
1. Match to purchased/carted items (complementary or similar)
2. Match to browsing interests
3. Price appropriateness for budget
4. Deal quality (discounts)

OUTPUT: Return ONLY a Python list of indices in best-to-worst order.
Example: [2, 0, 4, 1, 3]"""

    try:
        raw = call_gemini_llm(prompt, max_tokens=80, temperature=0.2)
        if not raw:
            return recs
        import re, ast
        match = re.search(r'\[[\d\s,]+\]', raw)
        if match:
            order = ast.literal_eval(match.group())
            n = len(to_rerank)
            # Validate the order list
            if len(order) == n and all(isinstance(i, int) and 0 <= i < n for i in order) and len(set(order)) == n:
                reranked = [to_rerank[i] for i in order]
                return frozen + reranked
    except Exception as e:
        print(f"[Rerank] LLM reranking failed: {e}")
    
    return recs


def extract_product_features(name, category):
    """Extract brand, features, and key attributes from product name."""
    import re
    words = name.split()
    # Common brand patterns (first capitalized word or first 2 words)
    brand = ""
    features = []
    colors = ['Black', 'White', 'Red', 'Blue', 'Green', 'Gold', 'Silver', 'Pink', 'Grey', 'Brown', 'Yellow', 'Purple', 'Orange', 'Navy', 'Beige']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', 'Small', 'Medium', 'Large', 'Free Size']
    materials = ['Cotton', 'Silk', 'Leather', 'Denim', 'Polyester', 'Wool', 'Linen', 'Velvet', 'Satin', 'Chiffon', 'Nylon', 'Rayon']
    
    # Extract brand (usually first word if capitalized)
    if words and words[0][0].isupper() and len(words[0]) > 2:
        brand = words[0]
        # Check for two-word brands
        if len(words) > 1 and words[1][0].isupper() and words[1].lower() not in ['for', 'with', 'and', 'the']:
            brand = f"{words[0]} {words[1]}"
    
    # Extract colors
    for w in words:
        if w.title() in colors:
            features.append(w.title())
            break
    
    # Extract materials
    for w in words:
        if w.title() in materials:
            features.append(w.title())
            break
    
    # Extract numeric specs (like "128GB", "6.5 inch", "2000W")
    specs = re.findall(r'\d+(?:\.\d+)?\s*(?:GB|TB|MP|mAh|W|inch|cm|mm|kg|g|L|ml)\b', name, re.I)
    features.extend(specs[:2])
    
    # Extract key adjectives
    adjectives = ['Premium', 'Luxury', 'Professional', 'Wireless', 'Smart', 'Portable', 'Organic', 'Natural', 'Ultra', 'Pro', 'Max', 'Plus', 'Slim', 'Lite', 'Classic', 'Modern', 'Vintage', 'Designer']
    for w in words:
        if w.title() in adjectives:
            features.append(w.title())
            break
    
    return {
        'brand': brand,
        'features': features[:3],
        'short_name': ' '.join(words[:4]) if len(words) > 4 else name
    }


# ── Robust JSON parser ────────────────────────────────────────────
def _attempt_parse_json(text: str):
    """
    Try to parse *text* as JSON. Handles common LLM quirks:
      - trailing commas before } or ]
      - single quotes instead of double quotes
      - unquoted keys
    Returns the parsed object on success, or None on failure.
    """
    import json, re
    if not text or not text.strip():
        return None
    s = text.strip()
    # 1. direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 2. remove trailing commas  ( ,} or ,] )
    cleaned = re.sub(r',\s*([}\]])', r'\1', s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 3. replace single quotes with double quotes (rough)
    try:
        return json.loads(cleaned.replace("'", '"'))
    except json.JSONDecodeError:
        pass
    # 4. try auto-closing unclosed braces/brackets
    opens = cleaned.count('{') - cleaned.count('}')
    opens_b = cleaned.count('[') - cleaned.count(']')
    if opens > 0 or opens_b > 0:
        patched = cleaned + '}' * max(opens, 0) + ']' * max(opens_b, 0)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass
    return None


# ── VLM Image Analysis ─────────────────────────────────────────────
def analyze_image_with_vlm(image_url: str, analysis_type: str = "content") -> dict:
    """
    Analyze product image using Vision Language Model (VLM).
    
    Uses Gemini 1.5 Flash for visual understanding.
    
    Args:
        image_url: URL of the product image
        analysis_type: Type of analysis:
            - "content": Analyze what's in the image (product, background, etc.)
            - "layout": Analyze where free space exists for text placement
            - "saliency": Identify the most visually important regions
    
    Returns:
        dict with analysis results
    """
    if not GEMINI_ENABLED or not image_url:
        return {"error": "VLM not available", "fallback": True}
    
    prompts = {
        "content": """Analyze this product image and provide a structured analysis:

1. PRODUCT: What is the main product? Describe its shape, orientation, and position in the image.
2. PRODUCT_BOUNDS: Estimate where the product is located (e.g., "centered", "left side", "upper half").
3. BACKGROUND: Describe the background (plain, gradient, textured, busy).
4. FREE_SPACE: Where is there empty/calm space suitable for text overlay? List regions like "top-left", "bottom", "right side", etc.
5. DOMINANT_COLORS: List the 2-3 main colors in the image.
6. TEXT_CONTRAST: Suggest whether light or dark text would be more readable.

Respond in JSON format only.""",
        
        "layout": """For this product image, analyze the layout and identify:

1. PRODUCT_REGION: Where is the product? Give approximate percentages (e.g., "from 20% to 80% horizontally, 10% to 75% vertically")
2. SAFE_ZONES: List regions where text can be placed without covering the product:
   - TOP_BAND: Is there space at the top? (yes/no, how much %)
   - BOTTOM_BAND: Is there space at the bottom? (yes/no, how much %)
   - LEFT_SIDE: Is there space on the left? (yes/no, how much %)
   - RIGHT_SIDE: Is there space on the right? (yes/no, how much %)
3. BEST_TEXT_POSITION: Where would text look best? (e.g., "bottom-band", "left-side", "top-overlay")
4. OVERLAY_POSSIBLE: Can text be overlaid on part of the image with good readability? (yes/no, where)

Respond in JSON format only.""",

        "saliency": """Analyze the visual saliency of this product image:

1. FOCAL_POINT: Where does the eye go first? (coordinates as percentage)
2. ATTENTION_FLOW: How does attention move through the image?
3. HIGH_DETAIL_REGIONS: Where are the most detailed/important parts? (list regions)
4. LOW_DETAIL_REGIONS: Where are calm/empty regions? (list regions)
5. TEXT_SAFE_ZONES: Rank the safest zones for text placement (best to worst)

Respond in JSON format only."""
    }
    
    prompt = prompts.get(analysis_type, prompts["content"])
    
    try:
        content = call_gemini_vlm(prompt, image_url, max_tokens=800, temperature=0.3)
        
        if not content:
            return {"error": "Empty VLM response", "fallback": True}
        
        # Try to parse as JSON (use robust helper to recover noisy model output)
        try:
            # Extract JSON from response (might be wrapped in markdown code blocks)
            json_match = content
            if '```json' in content:
                json_match = content.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in content:
                json_match = content.split('```', 1)[1].split('```', 1)[0].strip()

            parsed = _attempt_parse_json(json_match)
            if parsed is not None:
                return {"success": True, "analysis": parsed, "raw": content}
            else:
                return {"success": True, "analysis": None, "raw": content}
        except Exception:
            return {"success": True, "analysis": None, "raw": content}
            
    except Exception as e:
        print(f"[VLM] Analysis failed: {e}")
        return {"error": str(e), "fallback": True}


def rerank_placements_with_lm(
    product_name: str,
    category: str,
    candidate_regions: list,
    image_analysis: dict = None,
) -> list:
    """
    Use LM to rerank placement candidates based on e-commerce best practices.
    
    Uses Gemini 1.5 Flash for reasoning about optimal placement.
    
    Args:
        product_name: Name of the product
        category: Product category
        candidate_regions: List of candidate regions with scores
        image_analysis: Optional VLM analysis results
    
    Returns:
        Reranked list of regions with LM-adjusted scores
    """
    if not GEMINI_ENABLED or not candidate_regions:
        return candidate_regions
    
    # Format candidates for the prompt
    candidates_str = "\n".join([
        f"- {r['position']}: score={r['quality_score']:.2f}, saliency={r['avg_saliency']:.2f}, proximity={r['proximity_to_product']:.2f}"
        for r in candidate_regions[:8]  # Limit to top 8 for efficiency
    ])
    
    # Include image analysis if available
    image_context = ""
    if image_analysis and image_analysis.get("success"):
        analysis = image_analysis.get("analysis", {})
        if analysis:
            image_context = f"\nImage Analysis: {analysis}"
    
    prompt = f"""You are an e-commerce ad layout expert. Given a product and placement candidates, rerank them for optimal banner design.

Product: {product_name}
Category: {category}
{image_context}

Candidate Placement Regions:
{candidates_str}

RULES FOR E-COMMERCE BANNERS:
1. Product must remain the dominant focal element
2. Price should be highly visible and near the product (left, right, or below)
3. Avoid corner-only placements - product-adjacent is preferred
4. Text should not cover critical product details
5. Bottom bands work well for CTAs
6. Discount badges work well in top corners

Rerank these placements from best to worst. Output a JSON array of position names in order:
["best_position", "second_best", ...]"""

    try:
        content = call_gemini_llm(prompt, max_tokens=300, temperature=0.2)
        
        if not content:
            return candidate_regions
        
        # Parse the ranking
        try:
            # Extract JSON array from response
            if '[' in content and ']' in content:
                start = content.index('[')
                end = content.rindex(']') + 1
                ranking = _attempt_parse_json(content[start:end])
                if not isinstance(ranking, list):
                    ranking = None
                
                # Apply reranking boost
                position_to_region = {r['position']: r for r in candidate_regions}
                reranked = []
                boost = 0.3  # Boost for LM-preferred positions
                
                for i, pos in enumerate(ranking):
                    if pos in position_to_region:
                        region = position_to_region[pos].copy()
                        # Apply decreasing boost based on LM ranking
                        region['quality_score'] += boost * (1 - i * 0.1)
                        region['lm_rank'] = i + 1
                        reranked.append(region)
                
                # Add any regions not in LM ranking
                ranked_positions = set(ranking)
                for r in candidate_regions:
                    if r['position'] not in ranked_positions:
                        reranked.append(r)
                
                # Re-sort by adjusted score
                reranked.sort(key=lambda x: x['quality_score'], reverse=True)
                return reranked
        except (ValueError, Exception):
            pass
        
        return candidate_regions
        
    except Exception as e:
        print(f"[LM Rerank] Failed: {e}")
        return candidate_regions


# ── Wire up VLM/LM to laygen_pricemapping ──────────────────────────
# Connect the VLM and LM functions to the content-aware banner system
if HAS_LAYGEN and set_vlm_analyzer and set_lm_reranker:
    set_vlm_analyzer(analyze_image_with_vlm)
    set_lm_reranker(rerank_placements_with_lm)
    print("[OK] VLM and LM functions wired to laygen_pricemapping")


def generate_creative_ad_copy(prod):
    """Generate creative, product-specific ad copy using LLM or smart fallback."""
    name = prod.get('Product_Name', 'Product')[:100]
    category = prod.get('combined_category', '')
    price = prod.get('price', 0)
    mrp = prod.get('mrp', 0)
    discount = prod.get('discount_pct', 0)
    
    # Extract product-specific features
    extracted = extract_product_features(name, category)
    brand = extracted['brand']
    features = extracted['features']
    short_name = extracted['short_name']
    
    # Skip LLM if API key not configured
    if not GEMINI_ENABLED:
        return _generate_smart_ad_copy(name, category, brand, features, short_name, discount, price)
    
    # Build feature string for prompt
    feature_str = ""
    if brand:
        feature_str += f"Brand: {brand}\n"
    if features:
        feature_str += f"Key Features: {', '.join(features)}\n"
    
    prompt = f"""You are an expert advertising copywriter. Create SHORT, product-specific ad copy for this EXACT product.

PRODUCT: {name}
Category: {category}
{feature_str}Price: ₹{int(price):,}
{f'Was: ₹{int(mrp):,} | Save {discount}%' if mrp and mrp > price and discount > 0 else ''}

RULES:
1. TAGLINE must mention a SPECIFIC feature or benefit of THIS product (not generic)
2. HEADLINE should include the brand name or key product type
3. DESCRIPTION should be a catchy 1-liner that makes someone want to buy this product
4. CTA should create urgency specific to the deal/product

Generate exactly 4 lines (keep SHORT):
TAGLINE: [specific benefit/feature - max 6 words]
HEADLINE: [brand/product + compelling hook - max 6 words]
DESCRIPTION: [catchy reason to buy - max 10 words]
CTA: [urgent action - 3 words max]

BAD examples (too generic):
- "Elevate Your Style" ❌
- "Premium Quality" ❌

GOOD examples (product-specific):
- For "Nike Air Max Sneakers": DESCRIPTION: Run further, land softer — your feet will thank you
- For "Samsung 55 inch TV": DESCRIPTION: Movie night just got a massive, cinematic upgrade
- For "Lakme Face Cream": DESCRIPTION: Wake up glowing — the secret top models swear by

Now generate for the product above. NO quotes."""

    try:
        raw = call_gemini_llm(prompt, max_tokens=150, temperature=0.7)
        
        if not raw:
            return _generate_smart_ad_copy(name, category, brand, features, short_name, discount, price)
        
        import re
        # Parse the response
        tagline = headline = cta = description = ""
        for line in raw.split('\n'):
            line = line.strip()
            if line.upper().startswith('TAGLINE:'):
                tagline = line.split(':', 1)[1].strip().strip('"\'')
            elif line.upper().startswith('HEADLINE:'):
                headline = line.split(':', 1)[1].strip().strip('"\'')
            elif line.upper().startswith('DESCRIPTION:'):
                description = line.split(':', 1)[1].strip().strip('"\'')
            elif line.upper().startswith('CTA:'):
                cta = line.split(':', 1)[1].strip().strip('"\'')
        
        # Validate - if too generic, use fallback
        generic_phrases = ['premium quality', 'best choice', 'your perfect', 'elevate your', 'discover the']
        if tagline and any(g in tagline.lower() for g in generic_phrases):
            tagline = ""  # Force fallback
        
        return {
            'tagline': tagline or _generate_specific_tagline(name, category, features, discount),
            'headline': headline or (f"{brand} {short_name[:25]}" if brand else short_name[:30]),
            'description': description or _generate_catchy_description(name, category, features, discount, price),
            'cta': cta or ("Grab Deal!" if discount >= 20 else "Shop Now")
        }
    except Exception as e:
        print(f"[Ad Copy] LLM failed: {e}")
        return _generate_smart_ad_copy(name, category, brand, features, short_name, discount, price)


def _generate_specific_tagline(name, category, features, discount):
    """Generate a product-specific fallback tagline."""
    name_lower = name.lower()
    cat_lower = category.lower()
    
    # Product-type specific taglines
    if 'phone' in name_lower or 'mobile' in cat_lower:
        return "Smart. Fast. Always Connected."
    elif 'laptop' in name_lower or 'computer' in cat_lower:
        return "Power Meets Portability"
    elif 'watch' in name_lower:
        return "Time, Elevated"
    elif 'headphone' in name_lower or 'earphone' in name_lower or 'earbuds' in name_lower:
        return "Sound That Moves You"
    elif 'camera' in name_lower:
        return "Capture Every Moment"
    elif 'shirt' in name_lower or 'tshirt' in name_lower:
        return "Comfort Meets Style"
    elif 'dress' in name_lower:
        return "Turn Heads, Steal Hearts"
    elif 'shoe' in name_lower or 'sneaker' in name_lower or 'footwear' in cat_lower:
        return "Walk in Confidence"
    elif 'bag' in name_lower or 'backpack' in name_lower:
        return "Carry Your World"
    elif 'cream' in name_lower or 'lotion' in name_lower or 'beauty' in cat_lower:
        return "Reveal Your Glow"
    elif 'book' in cat_lower:
        return "Stories That Stay"
    elif 'kitchen' in cat_lower or 'appliance' in cat_lower:
        return "Kitchen Made Easy"
    elif features:
        return f"{features[0]} - Built for You"
    elif discount >= 30:
        return f"Save {discount}% Today Only!"
    else:
        return "Quality You Can Trust"


def _generate_smart_ad_copy(name, category, brand, features, short_name, discount, price):
    """Generate creative ad copy without LLM - uses product-specific templates and features."""
    name_lower = name.lower()
    cat_lower = category.lower()
    
    # Smart tagline based on product type and features
    tagline = _generate_specific_tagline(name, category, features, discount)
    
    # Smart headline with brand emphasis
    if brand:
        # Category-specific headline templates
        if 'phone' in name_lower or 'mobile' in cat_lower:
            headline = f"{brand} - Next-Gen Mobile"
        elif 'laptop' in name_lower or 'computer' in cat_lower:
            headline = f"{brand} - Work Smarter"
        elif 'watch' in name_lower:
            headline = f"{brand} - Time Perfected"
        elif 'headphone' in name_lower or 'earbuds' in name_lower:
            headline = f"{brand} - Pure Sound"
        elif 'camera' in name_lower:
            headline = f"{brand} - Capture Life"
        elif 'shoe' in name_lower or 'sneaker' in name_lower:
            headline = f"{brand} - Step Up"
        elif 'dress' in name_lower or 'shirt' in name_lower:
            headline = f"{brand} - Style Statement"
        elif 'bag' in name_lower or 'backpack' in name_lower:
            headline = f"{brand} - Pack Smart"
        elif 'cream' in name_lower or 'beauty' in cat_lower:
            headline = f"{brand} - Glow Up"
        elif 'appliance' in cat_lower or 'kitchen' in cat_lower:
            headline = f"{brand} - Home Upgrade"
        else:
            headline = f"{brand} - {short_name[:20]}"
    else:
        # No brand - use short descriptive headline
        words = short_name.split()[:4]
        headline = ' '.join(words) if len(words) > 1 else short_name[:30]
    
    # Smart CTA based on discount and price
    if discount >= 50:
        cta = "Steal This Deal!"
    elif discount >= 30:
        cta = f"Save {discount}%!"
    elif discount >= 20:
        cta = "Grab This Offer!"
    elif price and price < 500:
        cta = "Quick Buy!"
    elif price and price > 10000:
        cta = "Invest Now!"
    else:
        cta_options = ["Shop Now", "Get Yours", "Buy Today", "Add to Cart"]
        cta = random.choice(cta_options)
    
    return {
        'tagline': tagline,
        'headline': headline[:35],
        'description': _generate_catchy_description(name, category, features, discount, price),
        'cta': cta
    }


def _generate_catchy_description(name, category, features, discount, price):
    """Generate a catchy one-liner product description for the poster."""
    name_lower = name.lower()
    cat_lower = category.lower()
    
    if 'phone' in name_lower or 'mobile' in cat_lower:
        return "Lightning speed in the palm of your hand"
    elif 'laptop' in name_lower or 'computer' in cat_lower:
        return "Power up your hustle — work, play, create"
    elif 'watch' in name_lower:
        return "Track every heartbeat, own every second"
    elif 'headphone' in name_lower or 'earbuds' in name_lower or 'earphone' in name_lower:
        return "Drop the world, dive into pure sound"
    elif 'camera' in name_lower:
        return "Every click tells a story worth framing"
    elif 'shirt' in name_lower or 'tshirt' in name_lower:
        return "Dress sharp without even trying"
    elif 'dress' in name_lower:
        return "Walk in, own the room — effortlessly"
    elif 'shoe' in name_lower or 'sneaker' in name_lower or 'footwear' in cat_lower:
        return "Every step feels like you're walking on clouds"
    elif 'bag' in name_lower or 'backpack' in name_lower:
        return "Pack light, carry everything that matters"
    elif 'cream' in name_lower or 'beauty' in cat_lower or 'skin' in name_lower:
        return "Glow like you just got back from vacation"
    elif 'book' in cat_lower:
        return "A page-turner that stays with you forever"
    elif 'kitchen' in cat_lower or 'appliance' in cat_lower:
        return "Cook like a pro without breaking a sweat"
    elif 'toy' in cat_lower or 'game' in cat_lower:
        return "Fun that never gets old — game on!"
    elif 'fitness' in cat_lower or 'sport' in cat_lower:
        return "Push harder, go further — no excuses"
    elif discount and discount >= 40:
        return f"{int(discount)}% off — this deal won't last long"
    elif price and price < 500:
        return "Tiny price, massive value — grab it now"
    elif features:
        return f"Built with {features[0].lower()} — made for you"
    else:
        return "Your next favorite thing is one click away"


# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE CONTENT-AWARE AD LAYOUT GENERATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
# Inspired by "Content-Aware Ad Banner Layout Generation with Two-Stage
# Chain-of-Thought in Vision-Language Models".
#
# Stage 1 — PlacementPlanner   (text-only plan, NO coordinates)
# Stage 2 — LayoutRenderer     (plan → pixel coords + HTML)
# Orchestrator enforces strict sequential execution & data firewall.
# ══════════════════════════════════════════════════════════════════════════════

BANNER_DEBUG = True


# ── Element type constants ──────────────────────────────────────────────────
ELEMENT_TYPES = ["logo 0", "text 1", "text 2", "underlay 3"]


class PlacementPlanner:
    """
    STAGE 1 — Analyse detected objects, score image regions by
    importance, identify safe placement zones, and produce a
    text-only placement plan.  NO pixel coordinates are emitted.
    """

    # Importance weights for region scoring
    _IMPORTANCE_W = {'high': 1.0, 'medium': 0.5, 'low': 0.2}

    def __init__(self, image_width: int, image_height: int,
                 detected_objects: list):
        self.W = image_width
        self.H = image_height
        self.objects = detected_objects or []

        # Fixed 5×5 grid for region scoring
        self._GRID_COLS = 5
        self._GRID_ROWS = 5
        self._cell_w = self.W / self._GRID_COLS
        self._cell_h = self.H / self._GRID_ROWS

        # Precompute scores
        self._region_scores = self._score_regions()
        self._safe_zones = self._identify_safe_zones()

    # ── Region scoring ──────────────────────────────────────────────
    def _score_regions(self) -> list:
        """
        Divide the canvas into a GRID_ROWS × GRID_COLS grid and
        accumulate an importance score in each cell from every
        detected object whose bbox overlaps that cell.

        Higher score → more important content → worse for text.
        """
        scores = [[0.0] * self._GRID_COLS for _ in range(self._GRID_ROWS)]

        for obj in self.objects:
            weight = self._IMPORTANCE_W.get(
                obj.get('importance', 'low'), 0.2)
            bx, by, bw, bh = obj['bbox']

            # Which grid cells does this bbox intersect?
            c_min = max(0, int(bx / self._cell_w))
            c_max = min(self._GRID_COLS - 1,
                        int((bx + bw) / self._cell_w))
            r_min = max(0, int(by / self._cell_h))
            r_max = min(self._GRID_ROWS - 1,
                        int((by + bh) / self._cell_h))

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    # Overlap fraction ∈ (0, 1]
                    cx0 = c * self._cell_w
                    cy0 = r * self._cell_h
                    ox = max(0, min(bx + bw, cx0 + self._cell_w) -
                             max(bx, cx0))
                    oy = max(0, min(by + bh, cy0 + self._cell_h) -
                             max(by, cy0))
                    overlap = (ox * oy) / (self._cell_w * self._cell_h)
                    scores[r][c] += weight * overlap

        return scores

    # ── Safe-zone detection ─────────────────────────────────────────
    def _identify_safe_zones(self) -> list:
        """
        Return a list of safe zones described in spatial language.
        A zone is "safe" when its average importance score is below
        a threshold (≤ 0.15).
        """
        THRESH = 0.15
        zones = []

        # Named regions (row-slices × col-slices)
        named = {
            'top-left':     (range(0, 2), range(0, 2)),
            'top-center':   (range(0, 2), range(1, 4)),
            'top-right':    (range(0, 2), range(3, 5)),
            'center-left':  (range(1, 4), range(0, 2)),
            'center':       (range(1, 4), range(1, 4)),
            'center-right': (range(1, 4), range(3, 5)),
            'bottom-left':  (range(3, 5), range(0, 2)),
            'bottom-center':(range(3, 5), range(1, 4)),
            'bottom-right': (range(3, 5), range(3, 5)),
        }

        for name, (rows, cols) in named.items():
            total, count = 0.0, 0
            for r in rows:
                for c in cols:
                    total += self._region_scores[r][c]
                    count += 1
            avg = total / max(count, 1)
            if avg <= THRESH:
                zones.append({'name': name, 'avg_score': round(avg, 3)})

        # Guarantee at least one zone (pick the lowest scoring)
        if not zones:
            best_name, best_avg = 'bottom-center', 999.0
            for name, (rows, cols) in named.items():
                total, count = 0.0, 0
                for r in rows:
                    for c in cols:
                        total += self._region_scores[r][c]
                        count += 1
                avg = total / max(count, 1)
                if avg < best_avg:
                    best_avg = avg
                    best_name = name
            zones.append({'name': best_name, 'avg_score': round(best_avg, 3)})

        return zones

    # ── Public API ──────────────────────────────────────────────────
    def generate_plan(self, element_types: list = None) -> str:
        """
        Produce a textual placement plan.

        Output is descriptive spatial language ONLY — no pixel
        coordinates, no HTML.
        """
        if element_types is None:
            element_types = ELEMENT_TYPES

        safe = self._safe_zones
        safe_names = [z['name'] for z in safe]

        # Build a concise spatial summary of what to avoid
        avoids = []
        for obj in self.objects:
            if obj.get('importance') in ('high', 'medium'):
                bx, by, bw, bh = obj['bbox']
                cx = bx + bw / 2
                cy = by + bh / 2
                h_tag = ('left third' if cx < self.W / 3
                         else 'right third' if cx > 2 * self.W / 3
                         else 'horizontal center')
                v_tag = ('top third' if cy < self.H / 3
                         else 'bottom third' if cy > 2 * self.H / 3
                         else 'vertical center')
                avoids.append(
                    f"{obj['label']} ({obj['importance']}) in the {v_tag}, {h_tag}")

        avoid_str = ('; '.join(avoids) if avoids
                     else 'no high-importance objects detected')

        # Deterministic assignment: map each element type to a zone
        assignments = self._assign_elements_to_zones(element_types, safe)

        # Format the plan
        lines = ["Placement Plan:"]
        for elem, desc in assignments:
            lines.append(f"- {elem}: {desc}")

        plan_text = '\n'.join(lines)

        if BANNER_DEBUG:
            print(f"[PlacementPlanner] Safe zones: {safe_names}")
            print(f"[PlacementPlanner] Avoids: {avoid_str}")
            print(f"[PlacementPlanner] Plan:\n{plan_text}")

        return plan_text

    def _assign_elements_to_zones(self, element_types, safe_zones):
        """
        Deterministic (no randomness) element-to-zone assignment.

        Priority order:
          1. Logo    → prefer top-left or top-center (brand anchor)
          2. Text 1  → prefer center-left / top-center (headline)
          3. Text 2  → below Text 1, same horizontal region
          4. Underlay → behind the text elements
        """
        safe_names = [z['name'] for z in safe_zones]

        # Preference lists per element role
        prefs = {
            'logo 0':     ['top-left', 'top-center', 'top-right',
                           'bottom-left', 'center-left'],
            'text 1':     ['center-left', 'top-center', 'center',
                           'bottom-center', 'center-right'],
            'text 2':     ['center-left', 'bottom-center', 'center',
                           'center-right', 'bottom-left'],
            'underlay 3': [],  # special — derived from text positions
        }

        chosen = {}
        used = set()

        # Assign non-underlay elements first
        for elem in element_types:
            if elem.startswith('underlay'):
                continue
            pref_list = prefs.get(elem, safe_names)
            assigned = False
            for pname in pref_list:
                if pname in safe_names and pname not in used:
                    chosen[elem] = pname
                    used.add(pname)
                    assigned = True
                    break
            if not assigned:
                # Fall back to least-scored safe zone not yet used
                for z in safe_zones:
                    if z['name'] not in used:
                        chosen[elem] = z['name']
                        used.add(z['name'])
                        assigned = True
                        break
                if not assigned:
                    chosen[elem] = safe_names[0]

        # Build descriptive strings
        # Gather what high-importance objects each element avoids
        result = []
        for elem in element_types:
            zone = chosen.get(elem)
            if elem.startswith('underlay'):
                # Underlay covers the text elements
                text_elems = [e for e in element_types
                              if e.startswith('text')]
                covered_zones = [chosen[t] for t in text_elems if t in chosen]
                desc = (f"Place a filled rectangle behind the text elements "
                        f"({', '.join(covered_zones)}), extending with "
                        f"padding to ensure readability. Must stay behind "
                        f"all text and logo layers.")
            else:
                # Build avoidance note
                avoid_parts = []
                for obj in self.objects:
                    if obj.get('importance') == 'high':
                        avoid_parts.append(obj['label'])
                avoid_note = (f" Avoids {', '.join(avoid_parts)}."
                              if avoid_parts else '')
                if elem.startswith('logo'):
                    desc = (f"Place in the {zone} region of the canvas, "
                            f"aligned to the {zone.split('-')[0]} edge.{avoid_note}")
                elif elem.startswith('text'):
                    idx = elem.split()[-1]
                    align = ('left' if 'left' in zone else
                             'right' if 'right' in zone else 'center')
                    desc = (f"Place in the {zone} region, "
                            f"{align}-aligned, readable size.{avoid_note}")
                else:
                    desc = f"Place in the {zone} region.{avoid_note}"
            result.append((elem, desc))

        return result

    def get_detected_objects_summary(self) -> str:
        """Return a human-readable summary for debug / logging."""
        if not self.objects:
            return "No objects detected."
        lines = []
        for o in self.objects:
            lines.append(
                f"  {o['label']} importance={o['importance']} "
                f"bbox={o['bbox']}")
        return '\n'.join(lines)


class LayoutRenderer:
    """
    STAGE 2 — Parse the textual placement plan produced by Stage 1
    and convert it into pixel-accurate coordinates + valid HTML.

    This class NEVER receives raw image data or detected-object lists.
    It operates solely on the plan text and the canvas dimensions.
    """

    # Fixed sizing rules (deterministic, no randomness)
    _LOGO_W_FRAC = 0.18          # logo width = 18 % of canvas W
    _LOGO_H_FRAC = 0.07          # logo height = 7 % of canvas H
    _TEXT_W_FRAC  = 0.55          # text width  = 55 % of canvas W
    _TEXT_H_FRAC  = 0.06          # single text line height = 6 %
    _PAD          = 0.03          # padding = 3 % of smaller dimension
    _UNDERLAY_PAD = 0.015         # underlay extra bleed

    def __init__(self, image_width: int, image_height: int):
        self.W = image_width
        self.H = image_height
        self._pad_px = int(min(self.W, self.H) * self._PAD)
        self._elements: list = []   # populated by resolve step

    # ── Zone → anchor point mapping ─────────────────────────────────
    def _zone_anchor(self, zone_name: str) -> tuple:
        """
        Map a named zone (e.g. 'top-left') to an (x, y) anchor point
        in pixel space. The anchor is the TOP-LEFT corner of the zone.
        """
        col_map = {
            'left':   self._pad_px,
            'center': self.W // 2,
            'right':  self.W,
        }
        row_map = {
            'top':    self._pad_px,
            'center': self.H // 2,
            'bottom': self.H,
        }

        parts = zone_name.split('-')
        v = parts[0] if parts else 'center'
        h = parts[1] if len(parts) > 1 else 'center'

        x = col_map.get(h, self.W // 2)
        y = row_map.get(v, self.H // 2)
        return x, y

    # ── Plan parsing ────────────────────────────────────────────────
    def parse_plan(self, plan_text: str) -> list:
        """
        Extract (element_name, zone_name) pairs from the Stage 1
        placement plan text.

        Returns list of dicts:
          [{'element': 'logo 0', 'zone': 'top-left', 'role': 'logo'}, …]
        """
        import re
        parsed = []
        for line in plan_text.strip().splitlines():
            line = line.strip().lstrip('- ')
            # Match "Logo 0: Place in the top-left region …"
            m = re.match(
                r'(logo\s*\d+|text\s*\d+|underlay\s*\d+)\s*:\s*(.*)',
                line, re.IGNORECASE)
            if not m:
                continue
            elem = m.group(1).strip().lower()
            desc = m.group(2).strip()

            # Extract zone name from description
            zone_m = re.search(
                r'(?:in the\s+|behind.*?(?:elements?\s*\())'
                r'(top-left|top-center|top-right|'
                r'center-left|center|center-right|'
                r'bottom-left|bottom-center|bottom-right)',
                desc, re.IGNORECASE)
            zone = zone_m.group(1).lower() if zone_m else 'center'

            role = elem.split()[0]   # logo | text | underlay
            parsed.append({
                'element': elem,
                'zone': zone,
                'role': role,
                'raw_desc': desc,
            })

        self._parsed = parsed
        return parsed

    # ── Coordinate resolution ───────────────────────────────────────
    def resolve_coordinates(self, parsed: list = None) -> list:
        """
        Convert the parsed plan into concrete bounding boxes.

        Returns list of dicts with keys:
            element, role, left, top, width, height, z_index
        """
        if parsed is None:
            parsed = getattr(self, '_parsed', [])

        pad = self._pad_px
        elements = []
        text_boxes = []       # track text/logo boxes for underlay

        for item in parsed:
            role = item['role']
            zone = item['zone']
            ax, ay = self._zone_anchor(zone)

            if role == 'logo':
                w = int(self.W * self._LOGO_W_FRAC)
                h = int(self.H * self._LOGO_H_FRAC)
                x, y = self._align(ax, ay, w, h, zone)
                z = 3
            elif role == 'text':
                w = int(self.W * self._TEXT_W_FRAC)
                h = int(self.H * self._TEXT_H_FRAC)
                x, y = self._align(ax, ay, w, h, zone)
                z = 2
            elif role == 'underlay':
                # Compute after all others
                continue
            else:
                w = int(self.W * 0.20)
                h = int(self.H * 0.05)
                x, y = self._align(ax, ay, w, h, zone)
                z = 1

            # Clamp to canvas boundaries
            x = max(0, min(x, self.W - w))
            y = max(0, min(y, self.H - h))

            box = {
                'element': item['element'],
                'role': role,
                'left': x, 'top': y,
                'width': w, 'height': h,
                'z_index': z,
            }
            elements.append(box)
            if role in ('text', 'logo'):
                text_boxes.append(box)

        # Resolve overlap between non-underlay elements
        elements = self._resolve_overlaps(elements)

        # Now place underlays behind text/logo
        for item in parsed:
            if item['role'] != 'underlay':
                continue
            if not text_boxes:
                continue
            ux = min(b['left'] for b in text_boxes) - pad
            uy = min(b['top'] for b in text_boxes) - pad
            ux2 = max(b['left'] + b['width'] for b in text_boxes) + pad
            uy2 = max(b['top'] + b['height'] for b in text_boxes) + pad
            ux = max(0, ux)
            uy = max(0, uy)
            uw = min(ux2 - ux, self.W - ux)
            uh = min(uy2 - uy, self.H - uy)
            elements.append({
                'element': item['element'],
                'role': 'underlay',
                'left': ux, 'top': uy,
                'width': uw, 'height': uh,
                'z_index': 1,    # behind everything
            })

        # Sort by z_index so underlay is first (behind)
        elements.sort(key=lambda e: e['z_index'])
        self._elements = elements
        return elements

    # ── Alignment helper ────────────────────────────────────────────
    def _align(self, ax: int, ay: int, w: int, h: int,
               zone: str) -> tuple:
        """
        Compute (x, y) from an anchor point so the element sits
        naturally inside its zone.
        """
        pad = self._pad_px

        # Horizontal
        if 'left' in zone:
            x = pad
        elif 'right' in zone:
            x = self.W - w - pad
        else:   # center
            x = (self.W - w) // 2

        # Vertical
        if 'top' in zone:
            y = pad
        elif 'bottom' in zone:
            y = self.H - h - pad
        else:
            y = (self.H - h) // 2

        return x, y

    # ── Overlap resolution ──────────────────────────────────────────
    @staticmethod
    def _boxes_overlap(a: dict, b: dict) -> bool:
        return not (a['left'] + a['width'] <= b['left'] or
                    b['left'] + b['width'] <= a['left'] or
                    a['top'] + a['height'] <= b['top'] or
                    b['top'] + b['height'] <= a['top'])

    def _resolve_overlaps(self, elements: list) -> list:
        """
        Nudge elements downward to eliminate overlaps.
        Deterministic: iterate in list order, push later elements down.
        """
        for i in range(1, len(elements)):
            for j in range(i):
                if (elements[i]['role'] != 'underlay' and
                        elements[j]['role'] != 'underlay' and
                        self._boxes_overlap(elements[i], elements[j])):
                    # Push element i below element j
                    new_top = elements[j]['top'] + elements[j]['height'] + self._pad_px
                    elements[i]['top'] = min(new_top,
                                            self.H - elements[i]['height'])
        return elements

    # ── HTML generation ─────────────────────────────────────────────
    def generate_html(self, elements: list = None) -> str:
        """
        Produce strictly-formatted HTML layout from resolved elements.
        """
        if elements is None:
            elements = self._elements

        lines = ['<html>', '<body>',
                 f'  <div class="canvas" style="'
                 f'left:0px; top:0px; '
                 f'width:{self.W}px; height:{self.H}px"></div>']

        for el in elements:
            css_class = el['role']      # "logo" | "text" | "underlay"
            lines.append(
                f'  <div class="{css_class}" style="'
                f'left:{el["left"]}px; top:{el["top"]}px; '
                f'width:{el["width"]}px; height:{el["height"]}px">'
                f'</div>')

        lines += ['</body>', '</html>']
        html = '\n'.join(lines)

        if BANNER_DEBUG:
            print(f"[LayoutRenderer] Generated HTML ({len(elements)} elements)")
        return html


class AdLayoutOrchestrator:
    """
    Orchestrator — runs Stage 1 → Stage 2 sequentially and enforces
    the strict data-firewall between stages.

    • Stage 2 NEVER receives image data or detected objects.
    • Stage 2 ONLY receives the plan text and canvas dimensions.
    """

    def __init__(self):
        self._stage1_plan: str = ''
        self._stage2_elements: list = []
        self._stage2_html: str = ''

    def run(self,
            image_width: int,
            image_height: int,
            detected_objects: list,
            element_types: list = None) -> dict:
        """
        Execute the full two-stage pipeline.

        Returns:
            {
                'placement_plan': <str>,      # Stage 1 output
                'elements': [<boxes>],         # Stage 2 resolved coords
                'html': <str>,                 # Stage 2 HTML
            }
        """
        if element_types is None:
            element_types = ELEMENT_TYPES

        # ── STAGE 1 ─────────────────────────────────────────────────
        planner = PlacementPlanner(image_width, image_height,
                                   detected_objects)
        if BANNER_DEBUG:
            print(f"[Orchestrator] Stage 1 — PlacementPlanner "
                  f"({image_width}×{image_height}, "
                  f"{len(detected_objects)} objects)")

        self._stage1_plan = planner.generate_plan(element_types)

        # ── DATA FIREWALL ───────────────────────────────────────────
        # Stage 2 receives ONLY the plan text and canvas size.
        # No image data, no detected_objects.

        # ── STAGE 2 ─────────────────────────────────────────────────
        renderer = LayoutRenderer(image_width, image_height)
        if BANNER_DEBUG:
            print(f"[Orchestrator] Stage 2 — LayoutRenderer")

        parsed = renderer.parse_plan(self._stage1_plan)
        self._stage2_elements = renderer.resolve_coordinates(parsed)
        self._stage2_html = renderer.generate_html(self._stage2_elements)

        return {
            'placement_plan': self._stage1_plan,
            'elements': self._stage2_elements,
            'html': self._stage2_html,
        }


# ── Convenience entry-point (used by the existing banner pipeline) ──
def run_two_stage_ad_layout(image_width: int, image_height: int,
                            detected_objects: list,
                            element_types: list = None) -> dict:
    """
    Thin wrapper so the rest of the app can call the two-stage
    pipeline in one shot.
    """
    orch = AdLayoutOrchestrator()
    return orch.run(image_width, image_height,
                    detected_objects, element_types)


# ══════════════════════════════════════════════════════════════════════════════
# VISION-FIRST BANNER GENERATION PIPELINE (legacy helpers)
# ══════════════════════════════════════════════════════════════════════════════
# The functions below implement the image-fetching, VLM analysis, LLM
# placement, and PIL rendering used by generate_product_ad_image().
# They now integrate with the two-stage system above where applicable.
# ══════════════════════════════════════════════════════════════════════════════


def _vlm_analyze_image_for_banner(image_url: str) -> dict:
    """
    STAGE 1: Visual Analysis using VLM.
    
    Analyzes the product image to extract structured visual understanding.
    This is the ONLY place where the VLM sees the actual image.
    
    Returns:
        dict with keys:
            - product_bounds: {x_start, x_end, y_start, y_end} as percentages
            - critical_regions: list of regions that must NOT be covered
            - safe_zones: {top, bottom, left, right} with availability + size
            - background_type: 'plain' | 'gradient' | 'textured' | 'busy'
            - recommended_text_color: 'light' | 'dark'
            - visual_summary: brief description for LLM reasoning
    """
    if not GEMINI_ENABLED or not image_url:
        if BANNER_DEBUG:
            print("[Banner VLM] Gemini not enabled, using pixel analysis fallback")
        return _get_fallback_visual_analysis(image_url)
    
    # Structured prompt for visual analysis - JSON output required
    prompt = """Analyze this product image for ad banner text placement.

OUTPUT STRICT JSON WITH THESE EXACT KEYS:

{
  "product_bounds": {
    "x_start": <0-100 percentage from left where product starts>,
    "x_end": <0-100 percentage from left where product ends>,
    "y_start": <0-100 percentage from top where product starts>,
    "y_end": <0-100 percentage from top where product ends>
  },
  "critical_regions": [
    {"name": "<e.g., product face, logo, brand name>", "location": "<top-left|center|etc>"}
  ],
  "safe_zones": {
    "top_band": {"available": <true|false>, "height_pct": <0-40>},
    "bottom_band": {"available": <true|false>, "height_pct": <0-40>},
    "left_side": {"available": <true|false>, "width_pct": <0-40>},
    "right_side": {"available": <true|false>, "width_pct": <0-40>}
  },
  "background_type": "<plain|gradient|textured|busy>",
  "recommended_text_color": "<light|dark>",
  "visual_summary": "<One sentence describing the image layout>"
}

Be precise with percentages. If product is centered and fills most of the image, safe zones will be small."""

    try:
        if BANNER_DEBUG:
            print(f"[Banner VLM] Analyzing image: {image_url[:60]}...")
        
        raw_response = call_gemini_vlm(prompt, image_url, max_tokens=600, temperature=0.2)
        
        if not raw_response:
            if BANNER_DEBUG:
                print("[Banner VLM] Empty response, using pixel analysis fallback")
            return _get_fallback_visual_analysis(image_url)
        
        if BANNER_DEBUG:
            print(f"[Banner VLM] Raw response:\n{raw_response[:500]}")
        
        # Parse JSON from response - robust extraction
        import json
        try:
            clean = raw_response.strip()
            if '```json' in clean:
                clean = clean.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in clean:
                clean = clean.split('```', 1)[1].split('```', 1)[0].strip()
            brace_start = clean.find('{')
            if brace_start >= 0:
                depth, last_close = 0, -1
                for i, ch in enumerate(clean[brace_start:], brace_start):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            last_close = i
                            break
                if last_close >= 0:
                    clean = clean[brace_start:last_close+1]
                else:
                    open_count = clean[brace_start:].count('{') - clean[brace_start:].count('}')
                    clean = clean[brace_start:] + '}' * open_count
            analysis = _attempt_parse_json(clean)
            if analysis is not None:
                analysis = _validate_visual_analysis(analysis)
                if BANNER_DEBUG:
                    print(f"[Banner VLM] Parsed analysis: {json.dumps(analysis, indent=2)}")
                return {'success': True, 'analysis': analysis, 'raw': raw_response}
            else:
                if BANNER_DEBUG:
                    print(f"[Banner VLM] JSON parse failed, using pixel analysis fallback")
                return _get_fallback_visual_analysis(image_url)
        except Exception as e:
            if BANNER_DEBUG:
                print(f"[Banner VLM] Error during parse: {e}, using pixel analysis fallback")
            return _get_fallback_visual_analysis(image_url)
            
    except Exception as e:
        if BANNER_DEBUG:
            print(f"[Banner VLM] Error: {e}")
        return _get_fallback_visual_analysis(image_url)


def _validate_visual_analysis(analysis: dict) -> dict:
    """Validate and fill in missing fields in visual analysis."""
    # Default structure
    default = {
        'product_bounds': {'x_start': 20, 'x_end': 80, 'y_start': 10, 'y_end': 85},
        'critical_regions': [],
        'safe_zones': {
            'top_band': {'available': False, 'height_pct': 10},
            'bottom_band': {'available': True, 'height_pct': 20},
            'left_side': {'available': False, 'width_pct': 10},
            'right_side': {'available': False, 'width_pct': 10}
        },
        'background_type': 'plain',
        'recommended_text_color': 'light',
        'visual_summary': 'Product centered on neutral background'
    }
    
    # Merge with defaults
    result = default.copy()
    
    if 'product_bounds' in analysis and isinstance(analysis['product_bounds'], dict):
        for k in ['x_start', 'x_end', 'y_start', 'y_end']:
            if k in analysis['product_bounds']:
                try:
                    result['product_bounds'][k] = float(analysis['product_bounds'][k])
                except:
                    pass
    
    if 'critical_regions' in analysis and isinstance(analysis['critical_regions'], list):
        result['critical_regions'] = analysis['critical_regions']
    
    if 'safe_zones' in analysis and isinstance(analysis['safe_zones'], dict):
        for zone in ['top_band', 'bottom_band', 'left_side', 'right_side']:
            if zone in analysis['safe_zones'] and isinstance(analysis['safe_zones'][zone], dict):
                if 'available' in analysis['safe_zones'][zone]:
                    result['safe_zones'][zone]['available'] = bool(analysis['safe_zones'][zone]['available'])
                if 'height_pct' in analysis['safe_zones'][zone]:
                    try:
                        result['safe_zones'][zone]['height_pct'] = float(analysis['safe_zones'][zone]['height_pct'])
                    except:
                        pass
                if 'width_pct' in analysis['safe_zones'][zone]:
                    try:
                        result['safe_zones'][zone]['width_pct'] = float(analysis['safe_zones'][zone]['width_pct'])
                    except:
                        pass
    
    if 'background_type' in analysis:
        result['background_type'] = str(analysis['background_type']).lower()
    
    if 'recommended_text_color' in analysis:
        result['recommended_text_color'] = str(analysis['recommended_text_color']).lower()
    
    if 'visual_summary' in analysis:
        result['visual_summary'] = str(analysis['visual_summary'])
    
    return result


def _analyze_image_pixels(image_url: str) -> dict:
    """
    Pure PIL-based image analysis — works with no external APIs.
    Detects product bounding box and safe text zones by examining pixel content.

    Algorithm:
        1. Convert to greyscale, threshold to find non-background pixels
        2. Find the tightest bounding box around the product mass
        3. Compute free space in each of the four margins
        4. Recommend the layout that best uses the largest free zone
    """
    try:
        import numpy as np

        # ── Fetch & resize ──────────────────────────────────────────
        if image_url and str(image_url).startswith('http'):
            r = http_requests.get(image_url, timeout=8)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
        else:
            return None   # Caller will use static defaults

        AW, AH = 200, 200          # analysis resolution
        img_small = img.resize((AW, AH), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.float32)   # [H, W, 3]

        # ── Background colour: use corner median ────────────────────
        corners = np.concatenate([
            arr[:10, :10].reshape(-1, 3),
            arr[:10, -10:].reshape(-1, 3),
            arr[-10:, :10].reshape(-1, 3),
            arr[-10:, -10:].reshape(-1, 3),
        ])
        bg = np.median(corners, axis=0)               # [3]

        # ── Non-background mask ─────────────────────────────────────
        diff = np.linalg.norm(arr - bg, axis=2)       # [H, W]
        THRESH = 22.0
        mask = diff > THRESH                          # True = product pixel

        # Fallback if image is very uniform
        if mask.sum() < 100:
            diff2 = np.std(arr, axis=2)
            mask = diff2 > 8.0

        rows = np.any(mask, axis=1)     # [H]  rows that contain product
        cols = np.any(mask, axis=0)     # [W]

        if rows.any() and cols.any():
            # Use a density-based approach: find where the BULK of the
            # product mass is, ignoring sparse stray pixels at the edges.
            # This gives much more useful safe-zone measurements.
            row_density = np.sum(mask, axis=1).astype(float)  # pixels per row
            col_density = np.sum(mask, axis=0).astype(float)  # pixels per col
            total_mass = mask.sum()

            # Cumulative density — find 5th and 95th percentile rows/cols
            row_cum = np.cumsum(row_density) / max(total_mass, 1)
            col_cum = np.cumsum(col_density) / max(total_mass, 1)

            TRIM = 0.04   # ignore outermost 4% of product pixels
            r_min = int(np.searchsorted(row_cum, TRIM))
            r_max = int(np.searchsorted(row_cum, 1.0 - TRIM))
            c_min = int(np.searchsorted(col_cum, TRIM))
            c_max = int(np.searchsorted(col_cum, 1.0 - TRIM))

            # Clamp to valid range
            r_min = max(0, min(r_min, AH - 1))
            r_max = max(r_min + 1, min(r_max, AH - 1))
            c_min = max(0, min(c_min, AW - 1))
            c_max = max(c_min + 1, min(c_max, AW - 1))
        else:
            r_min, r_max, c_min, c_max = 10, AH-10, 10, AW-10

        # Convert to percentages
        y_start = round(r_min / AH * 100, 1)
        y_end   = round(r_max / AH * 100, 1)
        x_start = round(c_min / AW * 100, 1)
        x_end   = round(c_max / AW * 100, 1)

        # ── Safe zone sizes ─────────────────────────────────────────
        top_free    = y_start                        # % free above product
        bottom_free = 100.0 - y_end                  # % free below product
        left_free   = x_start                        # % free left of product
        right_free  = 100.0 - x_end                 # % free right of product

        # A zone is "available" if it has sufficient clearance for text
        MIN_BAND   = 8.0     # horizontal bands need less (gradient ensures readability)
        MIN_PANEL  = 12.0    # vertical panels need more width

        safe_zones = {
            'top_band':    {'available': top_free    >= MIN_BAND,  'height_pct': round(max(0, top_free), 1)},
            'bottom_band': {'available': bottom_free >= MIN_BAND,  'height_pct': round(max(0, bottom_free), 1)},
            'left_side':   {'available': left_free   >= MIN_PANEL, 'width_pct':  round(max(0, left_free), 1)},
            'right_side':  {'available': right_free  >= MIN_PANEL, 'width_pct':  round(max(0, right_free), 1)},
        }

        # ── Background brightness → text colour ─────────────────────
        brightness = float(np.mean(arr)) / 255.0
        text_color = 'light' if brightness < 0.55 else 'dark'

        # ── Background type ─────────────────────────────────────────
        std = float(np.std(arr[~mask.reshape(AH, AW)] if (~mask).sum() > 50 else arr)) / 255.0
        if std < 0.04:
            bg_type = 'plain'
        elif std < 0.12:
            bg_type = 'gradient'
        elif std < 0.22:
            bg_type = 'textured'
        else:
            bg_type = 'busy'

        # Build summary
        cx = (x_start + x_end) / 2
        cy = (y_start + y_end) / 2
        h_pos = 'left' if cx < 40 else ('right' if cx > 60 else 'center')
        v_pos = 'top' if cy < 35 else ('bottom' if cy > 65 else 'middle')
        summary = (f"Product occupies x=[{x_start:.0f}-{x_end:.0f}]% "
                   f"y=[{y_start:.0f}-{y_end:.0f}]%, "
                   f"positioned {h_pos}-{v_pos}, "
                   f"{bg_type} background, "
                   f"free space: top={top_free:.0f}% bottom={bottom_free:.0f}% "
                   f"left={left_free:.0f}% right={right_free:.0f}%")

        return {
            'success': True,
            'pixel_analysis': True,
            'analysis': {
                'product_bounds': {'x_start': x_start, 'x_end': x_end,
                                   'y_start': y_start, 'y_end': y_end},
                'critical_regions': [],
                'safe_zones': safe_zones,
                'background_type': bg_type,
                'recommended_text_color': text_color,
                'visual_summary': summary,
            }
        }

    except Exception as e:
        if BANNER_DEBUG:
            print(f"[Pixel Analysis] Failed: {e}")
        return None


def _get_fallback_visual_analysis(image_url: str = '') -> dict:
    """
    Try real pixel analysis first; only use static defaults if that fails too.
    """
    if image_url:
        result = _analyze_image_pixels(image_url)
        if result:
            if BANNER_DEBUG:
                print(f"[Pixel Analysis] {result['analysis']['visual_summary']}")
            return result

    # Last-resort static defaults
    return {
        'success': False,
        'fallback': True,
        'analysis': {
            'product_bounds': {'x_start': 15, 'x_end': 85, 'y_start': 5, 'y_end': 80},
            'critical_regions': [],
            'safe_zones': {
                'top_band':    {'available': False, 'height_pct': 8},
                'bottom_band': {'available': True,  'height_pct': 25},
                'left_side':   {'available': False, 'width_pct': 10},
                'right_side':  {'available': False, 'width_pct': 10},
            },
            'background_type': 'plain',
            'recommended_text_color': 'light',
            'visual_summary': 'Static fallback: bottom band',
        }
    }


def _llm_generate_placement_plan(visual_analysis: dict, ad_elements: dict) -> dict:
    """
    STAGE 2: Placement Planning using LLM.
    
    Takes the structured visual analysis and ad elements, then reasons about
    optimal placement. The LLM NEVER sees the raw image - only the analysis.
    
    Args:
        visual_analysis: Output from Stage 1 (VLM analysis)
        ad_elements: dict with keys: headline, tagline, price, mrp, discount_pct, cta
    
    Returns:
        dict with placement plan:
            - layout_type: 'bottom_band' | 'top_band' | 'left_panel' | 'right_panel' | 'overlay'
            - elements: list of {name, position, size, priority}
            - overlay_opacity: 0.0-1.0 if using overlay
            - reasoning: brief explanation of choices
    """
    analysis = visual_analysis.get('analysis', {})
    
    if not GEMINI_ENABLED:
        if BANNER_DEBUG:
            print("[Banner LLM] Gemini not enabled, using rule-based placement")
        return _get_rule_based_placement_plan(analysis, ad_elements)
    
    # Build prompt with visual analysis context - LLM never sees the image
    safe_zones = analysis.get('safe_zones', {})
    product_bounds = analysis.get('product_bounds', {})
    
    prompt = f"""You are an e-commerce ad layout expert. Plan text element placement for a product banner.

VISUAL ANALYSIS FROM IMAGE (you cannot see the image, only this analysis):
- Product location: x=[{product_bounds.get('x_start', 20)}-{product_bounds.get('x_end', 80)}]%, y=[{product_bounds.get('y_start', 10)}-{product_bounds.get('y_end', 85)}]%
- Background: {analysis.get('background_type', 'plain')}
- Safe zones:
  * Top band: {'✓ available' if safe_zones.get('top_band', {}).get('available') else '✗ blocked'}, height ~{safe_zones.get('top_band', {}).get('height_pct', 10)}%
  * Bottom band: {'✓ available' if safe_zones.get('bottom_band', {}).get('available') else '✗ blocked'}, height ~{safe_zones.get('bottom_band', {}).get('height_pct', 20)}%
  * Left side: {'✓ available' if safe_zones.get('left_side', {}).get('available') else '✗ blocked'}, width ~{safe_zones.get('left_side', {}).get('width_pct', 10)}%
  * Right side: {'✓ available' if safe_zones.get('right_side', {}).get('available') else '✗ blocked'}, width ~{safe_zones.get('right_side', {}).get('width_pct', 10)}%
- Text color suggestion: {analysis.get('recommended_text_color', 'light')} text
- Visual summary: {analysis.get('visual_summary', 'N/A')}

AD ELEMENTS TO PLACE:
- Headline: "{ad_elements.get('headline', '')[:40]}"
- Tagline: "{ad_elements.get('tagline', '')[:50] if ad_elements.get('tagline') else '(none)'}"
- Price: ₹{ad_elements.get('price', 0):,}
- MRP: ₹{ad_elements.get('mrp', 0):,} (strike-through if > price)
- Discount: {ad_elements.get('discount_pct', 0)}% off (badge if > 0)
- CTA: "{ad_elements.get('cta', 'Shop Now')}"

OUTPUT STRICT JSON:
{{
  "layout_type": "<top_band|bottom_band|left_panel|right_panel>",
  "band_size_pct": <20-40>,
  "overlay_opacity": <0.6-0.85>,
  "text_color": "<white|black>",
  "reasoning": "<One sentence explaining why this zone was chosen>"
}}

RULES:
1. Choose the layout_type that places text in the LARGEST available safe zone
2. If top band has the most free space → "top_band"
3. If bottom band has the most free space → "bottom_band"
4. If left side has the most free space → "left_panel"
5. If right side has the most free space → "right_panel"
6. Set band_size_pct to approximately match the safe zone size (minimum 20, maximum 40)
7. Pick text_color based on background brightness (light bg → black text, dark bg → white text)
8. ADAPT to each image — different products MUST get different layouts based on their unique composition
9. Never place text over the product center"""

    try:
        if BANNER_DEBUG:
            print("[Banner LLM] Generating placement plan...")
        
        raw_response = call_gemini_llm(prompt, max_tokens=500, temperature=0.3)
        
        if not raw_response:
            if BANNER_DEBUG:
                print("[Banner LLM] Empty response, using rule-based plan")
            return _get_rule_based_placement_plan(analysis, ad_elements)
        
        if BANNER_DEBUG:
            print(f"[Banner LLM] Raw response:\n{raw_response[:400]}")
        
        # Parse JSON - robust extraction handles truncated/markdown-wrapped responses
        import json, re
        try:
            # Strip markdown code fences
            clean = raw_response.strip()
            if '```json' in clean:
                clean = clean.split('```json', 1)[1]
                clean = clean.split('```', 1)[0].strip()
            elif '```' in clean:
                clean = clean.split('```', 1)[1]
                clean = clean.split('```', 1)[0].strip()

            # Find the outermost JSON object, even if response is truncated
            brace_start = clean.find('{')
            if brace_start >= 0:
                # Walk forward to find a balanced or best-effort JSON object
                depth, last_close = 0, -1
                for i, ch in enumerate(clean[brace_start:], brace_start):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            last_close = i
                            break
                if last_close >= 0:
                    clean = clean[brace_start:last_close+1]
                else:
                    # Truncated JSON - close open braces to recover partial data
                    open_count = clean[brace_start:].count('{') - clean[brace_start:].count('}')
                    clean = clean[brace_start:] + '}' * open_count

            plan = _attempt_parse_json(clean)
            if plan is not None:
                plan = _validate_placement_plan(plan, ad_elements)
                # Inject analysis data so the renderer knows product position
                plan['safe_zones_data'] = safe_zones
                plan['product_bounds'] = product_bounds
                if BANNER_DEBUG:
                    print(f"[Banner LLM] Placement plan: {json.dumps(plan, indent=2)}")
                return {'success': True, 'plan': plan, 'raw': raw_response}
            else:
                if BANNER_DEBUG:
                    print(f"[Banner LLM] JSON parse failed")
                return _get_rule_based_placement_plan(analysis, ad_elements)
        except Exception as e:
            if BANNER_DEBUG:
                print(f"[Banner LLM] Error during parse: {e}")
            return _get_rule_based_placement_plan(analysis, ad_elements)

    except Exception as e:
        if BANNER_DEBUG:
            print(f"[Banner LLM] Error: {e}")
        return _get_rule_based_placement_plan(analysis, ad_elements)


def _validate_placement_plan(plan: dict, ad_elements: dict) -> dict:
    """Validate and normalize the placement plan."""
    valid_layouts = ['bottom_band', 'top_band', 'left_panel', 'right_panel']
    
    result = {
        'layout_type': 'bottom_band',
        'band_size_pct': 25,
        'elements': [],
        'overlay_opacity': 0.7,
        'text_color': 'white',
        'reasoning': 'Default adaptive layout'
    }
    
    if 'layout_type' in plan and plan['layout_type'] in valid_layouts:
        result['layout_type'] = plan['layout_type']
    
    if 'band_size_pct' in plan:
        try:
            size = float(plan['band_size_pct'])
            result['band_size_pct'] = max(15, min(40, size))
        except:
            pass
    
    if 'elements' in plan and isinstance(plan['elements'], list):
        result['elements'] = plan['elements']
    else:
        # Default element arrangement
        result['elements'] = [
            {'name': 'discount_badge', 'zone': 'above_band_right', 'priority': 1},
            {'name': 'headline', 'zone': 'in_band_top', 'priority': 2},
            {'name': 'tagline', 'zone': 'in_band_below_headline', 'priority': 3},
            {'name': 'price_row', 'zone': 'in_band_left', 'priority': 4},
            {'name': 'cta_button', 'zone': 'in_band_right', 'priority': 5}
        ]
    
    if 'overlay_opacity' in plan:
        try:
            result['overlay_opacity'] = max(0.5, min(0.9, float(plan['overlay_opacity'])))
        except:
            pass
    
    if 'text_color' in plan:
        result['text_color'] = 'black' if 'black' in str(plan['text_color']).lower() else 'white'
    
    if 'reasoning' in plan:
        result['reasoning'] = str(plan['reasoning'])[:200]
    
    return result


def _get_rule_based_placement_plan(analysis: dict, ad_elements: dict) -> dict:
    """Adaptive layout selection based on per-image safe zone analysis.
    
    Scores all four zones (top/bottom/left/right) by their actual free
    space percentage and picks the one with the most room for text.
    This means every product image gets a DIFFERENT layout based on
    where its product is positioned and where the gaps are.
    """
    safe_zones = analysis.get('safe_zones', {})
    product_bounds = analysis.get('product_bounds', {})

    # ── Score each zone by actual free space ────────────────────────
    top_pct = safe_zones.get('top_band', {}).get('height_pct', 0)
    bot_pct = safe_zones.get('bottom_band', {}).get('height_pct', 0)
    left_pct = safe_zones.get('left_side', {}).get('width_pct', 0)
    right_pct = safe_zones.get('right_side', {}).get('width_pct', 0)

    # Boost score when zone is marked available; penalise otherwise
    zone_scores = {}
    zone_scores['top_band']    = top_pct   * (1.15 if safe_zones.get('top_band', {}).get('available') else 0.3)
    zone_scores['bottom_band'] = bot_pct   * (1.0  if safe_zones.get('bottom_band', {}).get('available') else 0.3)
    zone_scores['left_panel']  = left_pct  * (0.85 if safe_zones.get('left_side', {}).get('available') else 0.2)
    zone_scores['right_panel'] = right_pct * (0.85 if safe_zones.get('right_side', {}).get('available') else 0.2)

    best_layout = max(zone_scores, key=zone_scores.get)
    best_score  = zone_scores[best_layout]

    # If nothing has meaningful space, bottom_band is the safest fallback
    if best_score < 5:
        best_layout = 'bottom_band'

    # ── Band / panel size from real measurements ────────────────────
    if best_layout == 'top_band':
        band_size = max(22, min(40, top_pct + 5))
    elif best_layout == 'bottom_band':
        band_size = max(22, min(40, bot_pct + 5))
    elif best_layout == 'left_panel':
        band_size = max(30, min(48, left_pct + 5))
    else:  # right_panel
        band_size = max(30, min(48, right_pct + 5))

    text_color = 'white' if analysis.get('recommended_text_color', 'light') == 'light' else 'black'

    # Badge goes in the OPPOSITE quadrant from the text zone
    badge_zone = {
        'top_band': 'bottom_right', 'bottom_band': 'top_right',
        'left_panel': 'top_right',  'right_panel': 'top_left',
    }.get(best_layout, 'top_right')

    plan = {
        'layout_type': best_layout,
        'band_size_pct': band_size,
        'elements': [
            {'name': 'headline',       'zone': 'primary',   'priority': 1},
            {'name': 'tagline',        'zone': 'primary',   'priority': 2},
            {'name': 'description',    'zone': 'primary',   'priority': 3},
            {'name': 'price_row',      'zone': 'primary',   'priority': 4},
            {'name': 'cta_button',     'zone': 'primary',   'priority': 5},
            {'name': 'discount_badge', 'zone': badge_zone,  'priority': 6},
        ],
        'overlay_opacity': 0.75,
        'text_color': text_color,
        'safe_zones_data': safe_zones,
        'product_bounds': product_bounds,
        'zone_scores': zone_scores,
        'reasoning': (f'Adaptive: {best_layout} chosen (score={best_score:.1f}, '
                      f'T={top_pct:.0f}% B={bot_pct:.0f}% L={left_pct:.0f}% R={right_pct:.0f}%)')
    }

    if BANNER_DEBUG:
        print(f"[Rule-Based Plan] {plan['reasoning']}")

    return {'success': False, 'fallback': True, 'plan': plan}


def _extract_dominant_colors(image: Image.Image, n_colors=5):
    """Extract dominant colors from a product image using k-means-style quantization."""
    import numpy as np
    img_small = image.resize((80, 80), Image.LANCZOS).convert('RGB')
    arr = np.array(img_small).reshape(-1, 3).astype(float)

    # Simple mini-batch k-means (no sklearn needed)
    rng = np.random.RandomState(42)
    centers = arr[rng.choice(len(arr), n_colors, replace=False)]
    for _ in range(12):
        dists = np.linalg.norm(arr[:, None] - centers[None, :], axis=2)
        labels = dists.argmin(axis=1)
        for k in range(n_colors):
            members = arr[labels == k]
            if len(members) > 0:
                centers[k] = members.mean(axis=0)
    # Sort by cluster size (largest first)
    counts = np.bincount(labels, minlength=n_colors)
    order = np.argsort(-counts)
    return [tuple(int(c) for c in centers[i]) for i in order]


def _generate_ad_color_scheme(dominant_colors, category=''):
    """Generate a professional ad color scheme from dominant product colors.

    Returns dict with keys: bg_primary, bg_secondary, accent, text_primary,
    text_secondary, highlight, cta_bg, cta_text, badge_bg, badge_text.
    """
    import colorsys

    def rgb_to_hsv(r, g, b):
        return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    def hsv_to_rgb(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    # Pick the most vibrant dominant color (highest saturation)
    best = dominant_colors[0]
    best_sat = 0
    for c in dominant_colors[:4]:
        h, s, v = rgb_to_hsv(*c)
        if s * v > best_sat:
            best_sat = s * v
            best = c

    h, s, v = rgb_to_hsv(*best)

    # Build a rich, vibrant background based on the product's color
    bg_primary = hsv_to_rgb(h, min(0.85, s + 0.3), min(0.95, v + 0.15))
    bg_secondary = hsv_to_rgb((h + 0.08) % 1.0, min(0.9, s + 0.25), max(0.25, v - 0.25))

    # Determine if background is light or dark
    bg_lum = (0.299 * bg_primary[0] + 0.587 * bg_primary[1] + 0.114 * bg_primary[2]) / 255

    if bg_lum > 0.55:
        text_primary = (15, 15, 25)
        text_secondary = (50, 50, 70)
    else:
        text_primary = (255, 255, 255)
        text_secondary = (220, 220, 235)

    # Accent: complementary hue, fully saturated
    accent = hsv_to_rgb((h + 0.45) % 1.0, 0.9, 0.95)

    # Highlight for price — warm gold/yellow
    highlight = (255, 210, 50) if bg_lum < 0.5 else (200, 50, 20)

    # CTA: strong contrast button
    cta_h = (h + 0.55) % 1.0
    cta_bg = hsv_to_rgb(cta_h, 0.85, 0.92)
    cta_lum = (0.299 * cta_bg[0] + 0.587 * cta_bg[1] + 0.114 * cta_bg[2]) / 255
    cta_text = (255, 255, 255) if cta_lum < 0.5 else (15, 15, 25)

    # Badge
    badge_bg = hsv_to_rgb((h + 0.3) % 1.0, 0.8, 0.95)
    badge_lum = (0.299 * badge_bg[0] + 0.587 * badge_bg[1] + 0.114 * badge_bg[2]) / 255
    badge_text = (255, 255, 255) if badge_lum < 0.5 else (10, 10, 10)

    return {
        'bg_primary': bg_primary,
        'bg_secondary': bg_secondary,
        'accent': accent,
        'text_primary': text_primary,
        'text_secondary': text_secondary,
        'highlight': highlight,
        'cta_bg': cta_bg,
        'cta_text': cta_text,
        'badge_bg': badge_bg,
        'badge_text': badge_text,
    }


def _render_professional_ad(product_img: Image.Image, ad_elements: dict,
                            layout_type: str, width: int, height: int) -> Image.Image:
    """
    Render a professional, creative product ad poster.

    Creates a vibrant colored background derived from the product image,
    composites the product strategically, and draws large, bold, highly-
    visible typography using system display fonts (Impact, Bahnschrift, etc.).

    Each product gets a deterministically-different creative style through
    a hash-based style selector that varies gradient direction, decorative
    shapes, text placement, and accent patterns — so no two products look
    the same.

    Returns PIL Image (RGB).
    """
    import numpy as np
    import hashlib
    from PIL import ImageFilter

    # ── Deterministic style seed from product headline ────────────────
    seed_str = str(ad_elements.get('headline', '')) + str(ad_elements.get('price', ''))
    style_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    # ── STEP 1: Color extraction & scheme ─────────────────────────────
    dom_colors = _extract_dominant_colors(product_img)
    scheme = _generate_ad_color_scheme(dom_colors)

    bg1 = scheme['bg_primary']
    bg2 = scheme['bg_secondary']
    accent = scheme['accent']
    txt1 = scheme['text_primary']
    txt2 = scheme['text_secondary']
    highlight = scheme['highlight']
    cta_bg_color = scheme['cta_bg']
    cta_txt = scheme['cta_text']
    badge_bg_c = scheme['badge_bg']
    badge_txt_c = scheme['badge_text']

    # ── Creative style variations (hash-driven) ───────────────────────
    GRAD_STYLES = ['vertical', 'diagonal', 'radial', 'split_h', 'split_v']
    DECO_STYLES = ['circles', 'stripe', 'corner_block', 'dots', 'wave_bar', 'none']
    grad_style = GRAD_STYLES[style_hash % len(GRAD_STYLES)]
    deco_style = DECO_STYLES[(style_hash >> 4) % len(DECO_STYLES)]

    # Expand layout types: hash can override to more creative placements
    CREATIVE_LAYOUTS = [
        'top_hero',       # Big text top, product bottom-center
        'bottom_hero',    # Product top, big text bottom
        'left_panel',     # Text left 45%, product right
        'right_panel',    # Product left, text right
        'center_overlap', # Product center, text overlaps at bottom
        'diagonal_split', # Diagonal split: product one corner, text opposite
        'top_strip',      # Thin accent strip top, text middle, product bottom
        'full_center',    # Product large center, text overlay top + price bottom
    ]

    # Map the incoming layout_type to our expanded set using hash for variety
    layout_map = {
        'top_band': ['top_hero', 'top_strip', 'diagonal_split'],
        'bottom_band': ['bottom_hero', 'center_overlap', 'full_center'],
        'left_panel': ['left_panel', 'diagonal_split', 'center_overlap'],
        'right_panel': ['right_panel', 'full_center', 'top_hero'],
    }
    options = layout_map.get(layout_type, ['bottom_hero'])
    creative_layout = options[(style_hash >> 8) % len(options)]

    if BANNER_DEBUG:
        print(f"[Ad] Creative: layout={creative_layout}, grad={grad_style}, deco={deco_style}")

    # ── STEP 2: Gradient background ───────────────────────────────────
    bg1_arr = np.array(bg1, dtype=np.float32)
    bg2_arr = np.array(bg2, dtype=np.float32)

    if grad_style == 'vertical':
        frac = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        arr = bg1_arr * (1 - frac) + bg2_arr * frac
        arr = np.broadcast_to(arr, (height, width, 3)).copy()
    elif grad_style == 'diagonal':
        fy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        fx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        frac = ((fy + fx) / 2.0)[:, :, None]
        arr = bg1_arr * (1 - frac) + bg2_arr * frac
    elif grad_style == 'radial':
        cx_g, cy_g = width / 2, height / 2
        max_dist = (cx_g ** 2 + cy_g ** 2) ** 0.5
        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        dists = np.sqrt((xs - cx_g) ** 2 + (ys - cy_g) ** 2) / max_dist
        dists = np.clip(dists, 0, 1)[:, :, None]
        arr = bg1_arr * (1 - dists) + bg2_arr * dists
    elif grad_style == 'split_h':
        mid = width // 2
        blend_w = max(1, int(width * 0.15))
        cols = np.arange(width, dtype=np.float32)
        frac = np.clip((cols - (mid - blend_w / 2)) / max(blend_w, 1), 0, 1)[None, :, None]
        arr = np.broadcast_to(bg1_arr * (1 - frac) + bg2_arr * frac, (height, width, 3)).copy()
    else:  # split_v
        mid = height // 2
        blend_h = max(1, int(height * 0.15))
        rows = np.arange(height, dtype=np.float32)
        frac = np.clip((rows - (mid - blend_h / 2)) / max(blend_h, 1), 0, 1)[:, None, None]
        arr = np.broadcast_to(bg1_arr * (1 - frac) + bg2_arr * frac, (height, width, 3)).copy()

    canvas = Image.fromarray(arr.astype(np.uint8), 'RGB')
    canvas_rgba = canvas.convert('RGBA')
    draw = ImageDraw.Draw(canvas_rgba)

    # ── STEP 3: Decorative elements ───────────────────────────────────
    deco_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    deco_draw = ImageDraw.Draw(deco_layer)

    if deco_style == 'circles':
        r1 = int(min(width, height) * 0.30)
        cx1 = width // 2 + ((style_hash >> 12) % width // 3) - width // 6
        cy1 = height // 2 + ((style_hash >> 16) % height // 3) - height // 6
        deco_draw.ellipse((cx1 - r1, cy1 - r1, cx1 + r1, cy1 + r1),
                          fill=(*accent[:3], 40))
        r2 = int(r1 * 0.55)
        deco_draw.ellipse((cx1 + r1 // 2, cy1 - r1 // 2, cx1 + r1 // 2 + r2 * 2, cy1 - r1 // 2 + r2 * 2),
                          fill=(*bg1[:3], 50))
        # Tiny dot cluster
        for i in range(5):
            dx = (style_hash >> (20 + i)) % (width // 2)
            dy = (style_hash >> (25 + i)) % (height // 2)
            dr = 4 + i * 3
            deco_draw.ellipse((dx - dr, dy - dr, dx + dr, dy + dr),
                              fill=(*accent[:3], 25 + i * 5))

    elif deco_style == 'stripe':
        # Bold diagonal stripe accent
        stripe_w = int(width * 0.12)
        offset = (style_hash >> 12) % (width // 2)
        pts = [(offset, 0), (offset + stripe_w, 0),
               (offset + stripe_w - height // 3, height), (offset - height // 3, height)]
        deco_draw.polygon(pts, fill=(*accent[:3], 50))
        # Second thinner stripe
        pts2 = [(offset + stripe_w + 20, 0), (offset + stripe_w + 20 + stripe_w // 3, 0),
                (offset + stripe_w + 20 + stripe_w // 3 - height // 3, height),
                (offset + 20 - height // 3, height)]
        deco_draw.polygon(pts2, fill=(*bg1[:3], 35))

    elif deco_style == 'corner_block':
        # Large geometric block in one corner
        bw = int(width * 0.40)
        bh = int(height * 0.30)
        corner = (style_hash >> 12) % 4
        if corner == 0:
            deco_draw.rectangle((0, 0, bw, bh), fill=(*accent[:3], 45))
        elif corner == 1:
            deco_draw.rectangle((width - bw, 0, width, bh), fill=(*accent[:3], 45))
        elif corner == 2:
            deco_draw.rectangle((0, height - bh, bw, height), fill=(*accent[:3], 45))
        else:
            deco_draw.rectangle((width - bw, height - bh, width, height), fill=(*accent[:3], 45))

    elif deco_style == 'dots':
        # Grid of subtle dots
        spacing = int(width * 0.06)
        dot_r = max(2, int(width * 0.008))
        for row in range(0, height, spacing):
            for col in range(0, width, spacing):
                deco_draw.ellipse((col - dot_r, row - dot_r, col + dot_r, row + dot_r),
                                  fill=(*txt1[:3], 18))

    elif deco_style == 'wave_bar':
        # Horizontal accent bar across the poster
        bar_y = int(height * (0.3 + (style_hash >> 12) % 40 / 100.0))
        bar_h = int(height * 0.06)
        deco_draw.rectangle((0, bar_y, width, bar_y + bar_h), fill=(*accent[:3], 55))
        # Thin line above
        deco_draw.rectangle((0, bar_y - 3, width, bar_y), fill=(*txt1[:3], 30))

    canvas_rgba = Image.alpha_composite(canvas_rgba, deco_layer)
    draw = ImageDraw.Draw(canvas_rgba)

    # ── STEP 4: Load LARGE display fonts ──────────────────────────────
    # Use Windows system display fonts for maximum impact
    FONT_PATHS = {
        'impact': 'C:/Windows/Fonts/impact.ttf',
        'bahnschrift': 'C:/Windows/Fonts/bahnschrift.ttf',
        'arial_bold': 'C:/Windows/Fonts/arialbd.ttf',
        'segoe_bold': 'C:/Windows/Fonts/segoeuib.ttf',
        'calibri_bold': 'C:/Windows/Fonts/calibrib.ttf',
        'trebuchet_bold': 'C:/Windows/Fonts/trebucbd.ttf',
    }

    def _load_font(name, size, fallback_name='arial_bold'):
        """Load a font with fallback chain."""
        for attempt in [name, fallback_name, 'impact', 'arial_bold']:
            path = FONT_PATHS.get(attempt, '')
            if path:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        try:
            return ImageFont.truetype("arialbd.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except Exception:
                return ImageFont.load_default()

    # Font family selection varies per product (hash-driven)
    FONT_COMBOS = [
        ('impact', 'bahnschrift', 'segoe_bold'),       # Bold + modern
        ('bahnschrift', 'segoe_bold', 'calibri_bold'),  # Clean geometric
        ('impact', 'arial_bold', 'trebuchet_bold'),     # Strong + readable
        ('impact', 'trebuchet_bold', 'segoe_bold'),     # Display + editorial
        ('bahnschrift', 'arial_bold', 'segoe_bold'),    # Modern + clean
    ]
    combo = FONT_COMBOS[(style_hash >> 6) % len(FONT_COMBOS)]
    hero_font_name, body_font_name, accent_font_name = combo

    # Font sizes — MUCH bigger than before
    hero_size = max(42, int(height * 0.095))       # ~9.5% of height (was 6.5%)
    tagline_size = max(22, int(height * 0.040))    # ~4%
    desc_size = max(16, int(height * 0.028))       # ~2.8%
    price_size = max(46, int(height * 0.090))      # ~9% (was 7%)
    mrp_size = max(18, int(height * 0.032))        # ~3.2%
    cta_size = max(20, int(height * 0.040))        # ~4% (was 3.2%)
    badge_size = max(18, int(height * 0.034))      # ~3.4%

    font_hero = _load_font(hero_font_name, hero_size)
    font_tagline = _load_font(body_font_name, tagline_size)
    font_desc = _load_font(accent_font_name, desc_size)
    font_price = _load_font(hero_font_name, price_size)
    font_mrp = _load_font(body_font_name, mrp_size)
    font_cta = _load_font(body_font_name, cta_size)
    font_badge = _load_font(hero_font_name, badge_size)

    # ── STEP 5: Text helpers ──────────────────────────────────────────
    def text_size(text, font):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            return len(text) * 10, 20

    def wrap_text(text, font, max_w):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            tw, _ = text_size(t, font)
            if tw <= max_w:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [text[:20]]

    def draw_text_outlined(x, y, text, font, color, outline_color=None, thickness=3):
        """Draw text with thick outline for maximum readability."""
        if outline_color is None:
            # Auto-pick dark or light outline based on text color brightness
            lum = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255
            outline_color = (0, 0, 0, 180) if lum > 0.5 else (255, 255, 255, 60)
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx * dx + dy * dy <= thickness * thickness:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=color)

    def draw_text_shadow(x, y, text, font, color, offset=3, blur_alpha=100):
        """Draw text with drop shadow."""
        draw.text((x + offset, y + offset), text, font=font, fill=(0, 0, 0, blur_alpha))
        draw.text((x, y), text, font=font, fill=color)

    # ── STEP 6: Extract ad copy ───────────────────────────────────────
    headline = str(ad_elements.get('headline', ''))[:40]
    tagline = str(ad_elements.get('tagline', ''))[:60] if ad_elements.get('tagline') else ''
    description = str(ad_elements.get('description', ''))[:80] if ad_elements.get('description') else ''
    price = ad_elements.get('price', 0)
    mrp = ad_elements.get('mrp', 0)
    discount_pct = ad_elements.get('discount_pct', 0) or 0
    cta = str(ad_elements.get('cta', 'Shop Now'))

    pad = int(width * 0.055)

    # ── STEP 7: Product image compositing ─────────────────────────────
    prod_img = product_img.convert('RGBA')
    pw, ph = prod_img.size

    # Product sizing varies by layout
    if creative_layout in ('top_hero', 'bottom_hero'):
        max_pw = int(width * 0.80)
        max_ph = int(height * 0.48)
    elif creative_layout in ('left_panel', 'right_panel'):
        max_pw = int(width * 0.52)
        max_ph = int(height * 0.70)
    elif creative_layout == 'center_overlap':
        max_pw = int(width * 0.75)
        max_ph = int(height * 0.55)
    elif creative_layout == 'diagonal_split':
        max_pw = int(width * 0.58)
        max_ph = int(height * 0.55)
    elif creative_layout == 'top_strip':
        max_pw = int(width * 0.70)
        max_ph = int(height * 0.50)
    else:  # full_center
        max_pw = int(width * 0.70)
        max_ph = int(height * 0.58)

    sc = min(max_pw / max(pw, 1), max_ph / max(ph, 1))
    new_pw = int(pw * sc)
    new_ph = int(ph * sc)
    prod_resized = prod_img.resize((new_pw, new_ph), Image.LANCZOS)

    # Product position per creative layout
    if creative_layout == 'top_hero':
        prod_x = (width - new_pw) // 2
        prod_y = height - new_ph - int(height * 0.03)
    elif creative_layout == 'bottom_hero':
        prod_x = (width - new_pw) // 2
        prod_y = int(height * 0.03)
    elif creative_layout == 'left_panel':
        prod_x = width - new_pw - int(width * 0.03)
        prod_y = (height - new_ph) // 2
    elif creative_layout == 'right_panel':
        prod_x = int(width * 0.03)
        prod_y = (height - new_ph) // 2
    elif creative_layout == 'center_overlap':
        prod_x = (width - new_pw) // 2
        prod_y = int(height * 0.10)
    elif creative_layout == 'diagonal_split':
        prod_x = width - new_pw - int(width * 0.05)
        prod_y = int(height * 0.04)
    elif creative_layout == 'top_strip':
        prod_x = (width - new_pw) // 2
        prod_y = height - new_ph - int(height * 0.06)
    else:  # full_center
        prod_x = (width - new_pw) // 2
        prod_y = (height - new_ph) // 2 - int(height * 0.04)

    # Drop shadow behind product
    shadow_pad = 24
    shadow = Image.new('RGBA', (new_pw + shadow_pad * 2, new_ph + shadow_pad * 2), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    try:
        shadow_draw.rounded_rectangle(
            (shadow_pad // 2, shadow_pad // 2, new_pw + shadow_pad + shadow_pad // 2, new_ph + shadow_pad + shadow_pad // 2),
            radius=20, fill=(0, 0, 0, 65)
        )
    except Exception:
        shadow_draw.rectangle(
            (shadow_pad // 2, shadow_pad // 2, new_pw + shadow_pad + shadow_pad // 2, new_ph + shadow_pad + shadow_pad // 2),
            fill=(0, 0, 0, 65)
        )
    try:
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    except Exception:
        pass
    canvas_rgba.paste(shadow, (prod_x - shadow_pad, prod_y - shadow_pad), shadow)

    # Paste product
    canvas_rgba.paste(prod_resized, (prod_x, prod_y), prod_resized)
    draw = ImageDraw.Draw(canvas_rgba)

    # ── STEP 8: Draw text by creative layout ──────────────────────────
    # Each layout has its own text zone positions so text can be near
    # or overlapping the product, creating more dynamic compositions.

    price_str = f"\u20b9{int(price):,}" if price else ""
    mrp_str = f"\u20b9{int(mrp):,}" if mrp and mrp > price else ""

    if creative_layout == 'top_hero':
        # ── BIG headline at top, product at bottom ────────────────
        tx, ty = pad, pad
        tmw = width - pad * 2
        y = ty
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:3]:
                tw, th = text_size(line, font_hero)
                draw_text_shadow(tx, y, line, font_hero, txt1, offset=3, blur_alpha=110)
                y += th + int(height * 0.006)
        # Accent bar
        y += int(height * 0.010)
        draw.rectangle((tx, y, tx + int(tmw * 0.40), y + max(5, int(height * 0.008))), fill=accent)
        y += int(height * 0.022)
        # Tagline
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:2]:
                tw, th = text_size(line, font_tagline)
                draw.text((tx, y), line, font=font_tagline, fill=accent)
                y += th + 4
            y += int(height * 0.010)
        # Price row
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, y, price_str, font_price, highlight, offset=3, blur_alpha=110)
            if mrp_str:
                mx = tx + pw_s + 14
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.015)
        # CTA
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.05), int(height * 0.014)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(y, int(height * 0.47))
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    elif creative_layout == 'bottom_hero':
        # ── Product at top, BIG text at bottom ────────────────────
        tx = pad
        tmw = width - pad * 2
        y = height - int(height * 0.44)
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:3]:
                tw, th = text_size(line, font_hero)
                draw_text_shadow(tx, y, line, font_hero, txt1, offset=3, blur_alpha=110)
                y += th + int(height * 0.006)
        y += int(height * 0.008)
        draw.rectangle((tx, y, tx + int(tmw * 0.35), y + max(5, int(height * 0.008))), fill=accent)
        y += int(height * 0.020)
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:2]:
                draw.text((tx, y), line, font=font_tagline, fill=accent)
                _, th = text_size(line, font_tagline)
                y += th + 4
            y += int(height * 0.008)
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, y, price_str, font_price, highlight, offset=3)
            if mrp_str:
                mx = tx + pw_s + 14
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.012)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.05), int(height * 0.014)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(y, height - cbh - pad)
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    elif creative_layout in ('left_panel', 'right_panel'):
        # ── Text on one side (48%), product on the other ──────────
        if creative_layout == 'left_panel':
            tx = pad
        else:
            tx = width - int(width * 0.48)
        tmw = int(width * 0.44)
        y = int(height * 0.08)

        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:4]:
                tw, th = text_size(line, font_hero)
                draw_text_shadow(tx, y, line, font_hero, txt1, offset=3, blur_alpha=110)
                y += th + int(height * 0.005)
        y += int(height * 0.012)
        draw.rectangle((tx, y, tx + int(tmw * 0.40), y + max(5, int(height * 0.007))), fill=accent)
        y += int(height * 0.022)
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:2]:
                draw.text((tx, y), line, font=font_tagline, fill=accent)
                _, th = text_size(line, font_tagline)
                y += th + 3
            y += int(height * 0.010)
        if description:
            for line in wrap_text(description, font_desc, tmw)[:3]:
                draw.text((tx, y), line, font=font_desc, fill=txt2)
                _, th = text_size(line, font_desc)
                y += th + 2
            y += int(height * 0.018)
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, y, price_str, font_price, highlight, offset=3)
            if mrp_str:
                mx = tx + pw_s + 10
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.018)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.04), int(height * 0.012)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(y, height - cbh - pad * 2)
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    elif creative_layout == 'center_overlap':
        # ── Product center-top, text OVERLAPS at bottom ───────────
        # Semi-transparent overlay behind text for readability
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_top = int(height * 0.52)
        ov_draw.rectangle((0, ov_top, width, height), fill=(*bg2[:3], 180))
        canvas_rgba = Image.alpha_composite(canvas_rgba, overlay)
        draw = ImageDraw.Draw(canvas_rgba)

        tx = pad
        tmw = width - pad * 2
        y = ov_top + int(height * 0.03)
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:2]:
                tw, th = text_size(line, font_hero)
                draw_text_outlined(tx, y, line, font_hero, txt1, thickness=2)
                y += th + int(height * 0.005)
        y += int(height * 0.008)
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:2]:
                draw.text((tx, y), line, font=font_tagline, fill=accent)
                _, th = text_size(line, font_tagline)
                y += th + 3
            y += int(height * 0.008)
        # Price prominently
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_outlined(tx, y, price_str, font_price, highlight, thickness=2)
            if mrp_str:
                mx = tx + pw_s + 14
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.012)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.05), int(height * 0.013)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(y, height - cbh - pad)
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    elif creative_layout == 'diagonal_split':
        # ── Product upper-right, text lower-left (diagonal feel) ──
        tx = pad
        tmw = int(width * 0.60)
        y = int(height * 0.52)
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:3]:
                tw, th = text_size(line, font_hero)
                draw_text_shadow(tx, y, line, font_hero, txt1, offset=3)
                y += th + int(height * 0.005)
        y += int(height * 0.010)
        draw.rectangle((tx, y, tx + int(tmw * 0.45), y + max(5, int(height * 0.008))), fill=accent)
        y += int(height * 0.020)
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:2]:
                draw.text((tx, y), line, font=font_tagline, fill=accent)
                _, th = text_size(line, font_tagline)
                y += th + 3
            y += int(height * 0.008)
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, y, price_str, font_price, highlight, offset=3)
            if mrp_str:
                mx = tx + pw_s + 14
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.012)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.045), int(height * 0.013)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(y, height - cbh - pad)
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    elif creative_layout == 'top_strip':
        # ── Thin accent strip at top, headline middle, product bottom ─
        strip_h = int(height * 0.05)
        draw.rectangle((0, 0, width, strip_h), fill=accent)
        # Small brand text in strip
        if tagline:
            stw, sth = text_size(tagline[:30].upper(), font_desc)
            strip_lum = (0.299*accent[0]+0.587*accent[1]+0.114*accent[2])/255
            strip_txt_c = (255,255,255) if strip_lum < 0.5 else (10,10,10)
            draw.text(((width - stw) // 2, (strip_h - sth) // 2),
                      tagline[:30].upper(), font=font_desc, fill=strip_txt_c)

        tx = pad
        tmw = width - pad * 2
        y = strip_h + int(height * 0.04)
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:3]:
                tw, th = text_size(line, font_hero)
                draw_text_shadow(tx, y, line, font_hero, txt1, offset=3, blur_alpha=110)
                y += th + int(height * 0.005)
        y += int(height * 0.010)
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, y, price_str, font_price, highlight, offset=3)
            if mrp_str:
                mx = tx + pw_s + 14
                my = y + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            y += ph_s + int(height * 0.012)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.05), int(height * 0.013)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            try:
                draw.rounded_rectangle((tx, y, tx + cbw, y + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, y, tx + cbw, y + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, y + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    else:  # full_center
        # ── Product large center, headline overlaid top, price bottom ─
        # Top: headline overlaps product slightly
        tx = pad
        tmw = width - pad * 2
        y = pad
        if headline:
            for line in wrap_text(headline.upper(), font_hero, tmw)[:2]:
                tw, th = text_size(line, font_hero)
                draw_text_outlined(tx, y, line, font_hero, txt1, thickness=3)
                y += th + int(height * 0.004)

        # Bottom zone: price + CTA (may overlap product bottom)
        bot_y = height - int(height * 0.22)
        # Semi-transparent panel behind bottom text
        bot_panel = Image.new('RGBA', (width, height - bot_y), (*bg2[:3], 160))
        canvas_rgba.paste(bot_panel, (0, bot_y), bot_panel)
        draw = ImageDraw.Draw(canvas_rgba)

        by = bot_y + int(height * 0.02)
        if tagline:
            for line in wrap_text(tagline, font_tagline, tmw)[:1]:
                draw.text((tx, by), line, font=font_tagline, fill=accent)
                _, th = text_size(line, font_tagline)
                by += th + 4
        if price_str:
            pw_s, ph_s = text_size(price_str, font_price)
            draw_text_shadow(tx, by, price_str, font_price, highlight, offset=3)
            if mrp_str:
                mx = tx + pw_s + 14
                my = by + int(ph_s * 0.25)
                draw.text((mx, my), mrp_str, font=font_mrp, fill=txt2)
                try:
                    bb = draw.textbbox((mx, my), mrp_str, font=font_mrp)
                    draw.line((bb[0], (bb[1]+bb[3])//2, bb[2], (bb[1]+bb[3])//2), fill=txt2, width=2)
                except Exception: pass
            by += ph_s + int(height * 0.012)
        if cta:
            ct_w, ct_h = text_size(cta.upper(), font_cta)
            cpx, cpy = int(width * 0.05), int(height * 0.013)
            cbw, cbh = ct_w + cpx * 2, ct_h + cpy * 2
            cy_pos = min(by, height - cbh - pad)
            try:
                draw.rounded_rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh),
                                       radius=int(cbh * 0.45), fill=cta_bg_color)
            except Exception:
                draw.rectangle((tx, cy_pos, tx + cbw, cy_pos + cbh), fill=cta_bg_color)
            draw.text((tx + cpx, cy_pos + cpy), cta.upper(), font=font_cta, fill=cta_txt)

    # ── STEP 9: Discount badge (all layouts) ──────────────────────────
    if discount_pct and discount_pct > 0:
        badge_text = f" {int(discount_pct)}% OFF "
        btw, bth = text_size(badge_text, font_badge)
        bw = btw + 20
        bh = bth + 14
        # Place in a corner away from main text
        if creative_layout in ('top_hero', 'top_strip', 'diagonal_split'):
            bx, by_b = width - bw - pad, height - bh - pad - new_ph // 4
            by_b = max(by_b, height // 2)
        elif creative_layout in ('bottom_hero', 'center_overlap'):
            bx, by_b = width - bw - pad, pad
        elif creative_layout == 'left_panel':
            bx, by_b = width - bw - pad, pad
        elif creative_layout == 'right_panel':
            bx, by_b = pad, pad
        else:
            bx, by_b = width - bw - pad, pad

        try:
            draw.rounded_rectangle((bx, by_b, bx + bw, by_b + bh),
                                   radius=int(bh * 0.45), fill=badge_bg_c)
        except Exception:
            draw.rectangle((bx, by_b, bx + bw, by_b + bh), fill=badge_bg_c)
        draw.text((bx + 10, by_b + 7), badge_text, font=font_badge, fill=badge_txt_c)

    # ── STEP 10: Final output ─────────────────────────────────────────
    result = Image.new('RGB', (width, height), (255, 255, 255))
    result.paste(canvas_rgba, mask=canvas_rgba.split()[3])
    return result


def _render_adaptive_zone(draw, image, img_w, img_h, band_x, band_y, band_w, band_h, pad,
                          headline, tagline, description, price, mrp, discount_pct, cta,
                          font_headline, font_tagline, font_price, font_small, font_cta,
                          text_color, muted_color, price_color, cta_bg, discount_bg,
                          layout_type, overlay_opacity):
    """
    ADAPTIVE ZONE RENDERER — places ALL text in the single best zone.

    Works for any layout_type: top_band, bottom_band, left_panel, right_panel.
    The gradient overlay is already drawn by _render_banner_from_plan.
    Text placement adapts per-image: different products get different zones
    based on where their gaps and free space are.

    Layout examples:
      top_band:     [HEADLINE / tagline / desc / price / CTA] at top, badge bottom-right
      bottom_band:  badge top-right, [HEADLINE / tagline / desc / price / CTA] at bottom
      left_panel:   [HEADLINE / tagline / desc / price / CTA] on left, badge top-right
      right_panel:  badge top-left, [HEADLINE / tagline / desc / price / CTA] on right
    """
    element_boxes = []
    is_horizontal = layout_type in ['top_band', 'bottom_band']

    # ── Helper: word-wrap text ───────────────────────────────────────
    def wrap_text(text, font, max_width):
        words = text.split()
        lines, current = [], ""
        for w in words:
            test = f"{current} {w}".strip()
            try:
                bbox = draw.textbbox((0, 0), test, font=font)
                tw = bbox[2] - bbox[0]
            except:
                tw = len(test) * 8
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines or [text[:20]]

    # ── Helper: draw text with shadow for readability ────────────────
    def draw_shadow_text(x, y, text, font, color):
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=color)

    # ── Helper: measure text ─────────────────────────────────────────
    def text_size(text, font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            return len(text) * 8, 16

    # ── Drawing area boundaries ──────────────────────────────────────
    text_max_w = band_w - pad * 3
    x_start = band_x + pad
    y_cursor = band_y + pad

    if not is_horizontal:
        text_max_w = band_w - pad * 2

    # ── HEADLINE ─────────────────────────────────────────────────────
    if headline:
        lines = wrap_text(headline.upper(), font_headline, text_max_w)
        for line in lines[:2]:
            tw, th = text_size(line, font_headline)
            lx = x_start + (text_max_w - tw) // 2 if is_horizontal else x_start
            draw_shadow_text(lx, y_cursor, line, font_headline, text_color)
            element_boxes.append({'name': 'headline', 'bbox': (lx, y_cursor, lx + tw, y_cursor + th)})
            y_cursor += th + 4
        y_cursor += 2

    # ── TAGLINE ──────────────────────────────────────────────────────
    if tagline:
        lines = wrap_text(tagline, font_tagline, text_max_w)
        for line in lines[:2]:
            tw, th = text_size(line, font_tagline)
            lx = x_start + (text_max_w - tw) // 2 if is_horizontal else x_start
            draw_shadow_text(lx, y_cursor, line, font_tagline, muted_color)
            element_boxes.append({'name': 'tagline', 'bbox': (lx, y_cursor, lx + tw, y_cursor + th)})
            y_cursor += th + 3
        y_cursor += 4

    # ── DESCRIPTION (catchy one-liner) ───────────────────────────────
    if description:
        lines = wrap_text(description, font_small, text_max_w)
        for line in lines[:2]:
            tw, th = text_size(line, font_small)
            lx = x_start + (text_max_w - tw) // 2 if is_horizontal else x_start
            draw_shadow_text(lx, y_cursor, line, font_small, text_color)
            element_boxes.append({'name': 'description', 'bbox': (lx, y_cursor, lx + tw, y_cursor + th)})
            y_cursor += th + 3
        y_cursor += 6

    # ── PRICE + MRP row ──────────────────────────────────────────────
    price_str = f"\u20b9{int(price):,}" if price else ""
    if price_str:
        draw_shadow_text(x_start, y_cursor, price_str, font_price, price_color)
        pw, ph = text_size(price_str, font_price)
        element_boxes.append({'name': 'price', 'bbox': (x_start, y_cursor, x_start + pw, y_cursor + ph)})

        if mrp and mrp > price:
            mrp_str = f"\u20b9{int(mrp):,}"
            mrp_x = x_start + pw + 10
            draw.text((mrp_x, y_cursor + 5), mrp_str, font=font_small, fill=muted_color)
            try:
                mrp_bbox = draw.textbbox((mrp_x, y_cursor + 5), mrp_str, font=font_small)
                draw.line((mrp_bbox[0], (mrp_bbox[1] + mrp_bbox[3]) // 2,
                           mrp_bbox[2], (mrp_bbox[1] + mrp_bbox[3]) // 2),
                          fill=muted_color, width=1)
            except:
                pass
        y_cursor += ph + 8

    # ── CTA BUTTON ───────────────────────────────────────────────────
    if cta:
        cta_tw, cta_th = text_size(cta, font_cta)
        cta_pad_x, cta_pad_y = 16, 8
        cta_w = cta_tw + cta_pad_x * 2
        cta_h = cta_th + cta_pad_y * 2
        cta_x = x_start
        # Ensure CTA stays inside the band
        cta_y = min(y_cursor, band_y + band_h - cta_h - pad)
        try:
            draw.rounded_rectangle((cta_x, cta_y, cta_x + cta_w, cta_y + cta_h), radius=10, fill=cta_bg)
        except:
            draw.rectangle((cta_x, cta_y, cta_x + cta_w, cta_y + cta_h), fill=cta_bg)
        draw.text((cta_x + cta_pad_x, cta_y + cta_pad_y), cta, font=font_cta, fill=(255, 255, 255))
        element_boxes.append({'name': 'cta', 'bbox': (cta_x, cta_y, cta_x + cta_w, cta_y + cta_h)})

    # ── DISCOUNT BADGE — positioned in the OPPOSITE corner ───────────
    if discount_pct and discount_pct > 0:
        pill_text = f" -{int(discount_pct)}% OFF "
        ppw, pph = text_size(pill_text, font_small)
        pill_w = ppw + 14
        pill_h = pph + 10

        # Place badge in corner opposite to the text zone
        if layout_type == 'bottom_band':
            px1, py1 = img_w - pill_w - pad, pad
        elif layout_type == 'top_band':
            px1, py1 = img_w - pill_w - pad, img_h - pill_h - pad
        elif layout_type == 'left_panel':
            px1, py1 = img_w - pill_w - pad, pad
        else:  # right_panel
            px1, py1 = pad, pad

        try:
            draw.rounded_rectangle((px1, py1, px1 + pill_w, py1 + pill_h), radius=8, fill=discount_bg)
        except:
            draw.rectangle((px1, py1, px1 + pill_w, py1 + pill_h), fill=discount_bg)
        draw.text((px1 + 7, py1 + 5), pill_text, font=font_small, fill=(0, 0, 0))
        element_boxes.append({'name': 'discount_badge', 'bbox': (px1, py1, px1 + pill_w, py1 + pill_h)})

    return element_boxes


def _render_horizontal_band(draw, image, band_x, band_y, band_w, band_h, pad,
                            headline, tagline, price, mrp, discount_pct, cta,
                            font_headline, font_tagline, font_price, font_small, font_cta,
                            text_color, muted_color, price_color, cta_bg, discount_bg,
                            layout_type):
    """Render elements in a horizontal band (top or bottom)."""
    element_boxes = []
    
    y_cursor = band_y + pad
    
    # Headline
    if headline:
        draw.text((band_x + pad, y_cursor), headline, font=font_headline, fill=text_color)
        try:
            bbox = draw.textbbox((band_x + pad, y_cursor), headline, font=font_headline)
            element_boxes.append({'name': 'headline', 'bbox': bbox})
        except:
            pass
        y_cursor += int(band_h * 0.28)
    
    # Tagline
    if tagline:
        draw.text((band_x + pad, y_cursor), tagline, font=font_tagline, fill=muted_color)
        try:
            bbox = draw.textbbox((band_x + pad, y_cursor), tagline, font=font_tagline)
            element_boxes.append({'name': 'tagline', 'bbox': bbox})
        except:
            pass
        y_cursor += int(band_h * 0.22)
    
    # Price row
    price_str = f"₹{int(price):,}" if price else ""
    if price_str:
        draw.text((band_x + pad, y_cursor), price_str, font=font_price, fill=price_color)
        try:
            price_bbox = draw.textbbox((band_x + pad, y_cursor), price_str, font=font_price)
            element_boxes.append({'name': 'price', 'bbox': price_bbox})
            
            # MRP with strikethrough
            if mrp and mrp > price:
                mrp_str = f"₹{int(mrp):,}"
                mrp_x = price_bbox[2] + 15
                draw.text((mrp_x, y_cursor + 5), mrp_str, font=font_small, fill=muted_color)
                mrp_bbox = draw.textbbox((mrp_x, y_cursor + 5), mrp_str, font=font_small)
                draw.line((mrp_bbox[0], (mrp_bbox[1]+mrp_bbox[3])//2, mrp_bbox[2], (mrp_bbox[1]+mrp_bbox[3])//2), 
                         fill=muted_color, width=1)
        except:
            pass
    
    # CTA button on the right
    cta_w = int(band_w * 0.25)
    cta_h = int(band_h * 0.32)
    cx1 = band_x + band_w - cta_w - pad
    cy1 = band_y + (band_h - cta_h) // 2
    
    try:
        draw.rounded_rectangle((cx1, cy1, cx1 + cta_w, cy1 + cta_h), radius=8, fill=cta_bg)
    except:
        draw.rectangle((cx1, cy1, cx1 + cta_w, cy1 + cta_h), fill=cta_bg)
    
    try:
        bbox = draw.textbbox((0, 0), cta, font=font_cta)
        txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        txt_w, txt_h = len(cta) * 8, 14
    draw.text((cx1 + (cta_w - txt_w) // 2, cy1 + (cta_h - txt_h) // 2), cta, font=font_cta, fill=(255, 255, 255))
    element_boxes.append({'name': 'cta', 'bbox': (cx1, cy1, cx1 + cta_w, cy1 + cta_h)})
    
    # Discount badge
    if discount_pct > 0:
        pill_text = f"-{int(discount_pct)}%"
        pill_w = int(band_w * 0.15)
        pill_h = int(band_h * 0.24)
        
        # Position above band for bottom layout, below for top
        if layout_type == 'bottom_band':
            px1 = band_x + band_w - pill_w - pad
            py1 = band_y - pill_h - 8
        else:
            px1 = band_x + band_w - pill_w - pad
            py1 = band_y + band_h + 8
        
        try:
            draw.rounded_rectangle((px1, py1, px1 + pill_w, py1 + pill_h), radius=6, fill=discount_bg)
        except:
            draw.rectangle((px1, py1, px1 + pill_w, py1 + pill_h), fill=discount_bg)
        
        try:
            bbox = draw.textbbox((0, 0), pill_text, font=font_small)
            txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            txt_w, txt_h = len(pill_text) * 7, 12
        draw.text((px1 + (pill_w - txt_w) // 2, py1 + (pill_h - txt_h) // 2), pill_text, font=font_small, fill=(0, 0, 0))
        element_boxes.append({'name': 'discount_badge', 'bbox': (px1, py1, px1 + pill_w, py1 + pill_h)})
    
    return element_boxes


def _render_vertical_panel(draw, image, panel_x, panel_y, panel_w, panel_h, pad,
                           headline, tagline, price, mrp, discount_pct, cta,
                           font_headline, font_tagline, font_price, font_small, font_cta,
                           text_color, muted_color, price_color, cta_bg, discount_bg,
                           layout_type):
    """Render elements in a vertical side panel."""
    element_boxes = []
    
    y_cursor = panel_y + pad * 2
    x_start = panel_x + pad
    max_w = panel_w - pad * 2
    
    # Discount badge at top
    if discount_pct > 0:
        pill_text = f"-{int(discount_pct)}%"
        pill_w = int(panel_w * 0.5)
        pill_h = int(panel_h * 0.05)
        
        try:
            draw.rounded_rectangle((x_start, y_cursor, x_start + pill_w, y_cursor + pill_h), 
                                   radius=6, fill=discount_bg)
        except:
            draw.rectangle((x_start, y_cursor, x_start + pill_w, y_cursor + pill_h), fill=discount_bg)
        
        try:
            bbox = draw.textbbox((0, 0), pill_text, font=font_small)
            txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            txt_w, txt_h = len(pill_text) * 7, 12
        draw.text((x_start + (pill_w - txt_w) // 2, y_cursor + (pill_h - txt_h) // 2), 
                 pill_text, font=font_small, fill=(0, 0, 0))
        element_boxes.append({'name': 'discount_badge', 'bbox': (x_start, y_cursor, x_start + pill_w, y_cursor + pill_h)})
        y_cursor += pill_h + pad
    
    # Headline with word wrap
    if headline:
        words = headline.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font_headline)
                w = bbox[2] - bbox[0]
            except:
                w = len(test_line) * 10
            if w <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for line in lines[:3]:
            draw.text((x_start, y_cursor), line, font=font_headline, fill=text_color)
            y_cursor += int(panel_h * 0.05)
        y_cursor += pad
    
    # Price
    price_str = f"₹{int(price):,}" if price else ""
    if price_str:
        draw.text((x_start, y_cursor), price_str, font=font_price, fill=price_color)
        try:
            element_boxes.append({'name': 'price', 'bbox': draw.textbbox((x_start, y_cursor), price_str, font=font_price)})
        except:
            pass
        y_cursor += int(panel_h * 0.06)
        
        # MRP
        if mrp and mrp > price:
            mrp_str = f"₹{int(mrp):,}"
            draw.text((x_start, y_cursor), mrp_str, font=font_small, fill=muted_color)
            try:
                mrp_bbox = draw.textbbox((x_start, y_cursor), mrp_str, font=font_small)
                draw.line((mrp_bbox[0], (mrp_bbox[1]+mrp_bbox[3])//2, mrp_bbox[2], (mrp_bbox[1]+mrp_bbox[3])//2), 
                         fill=muted_color, width=1)
            except:
                pass
            y_cursor += int(panel_h * 0.04)
    
    # CTA at bottom
    cta_w = panel_w - pad * 2
    cta_h = int(panel_h * 0.06)
    cy1 = panel_y + panel_h - cta_h - pad * 2
    
    try:
        draw.rounded_rectangle((x_start, cy1, x_start + cta_w, cy1 + cta_h), radius=8, fill=cta_bg)
    except:
        draw.rectangle((x_start, cy1, x_start + cta_w, cy1 + cta_h), fill=cta_bg)
    
    try:
        bbox = draw.textbbox((0, 0), cta, font=font_cta)
        txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        txt_w, txt_h = len(cta) * 8, 14
    draw.text((x_start + (cta_w - txt_w) // 2, cy1 + (cta_h - txt_h) // 2), cta, font=font_cta, fill=(255, 255, 255))
    element_boxes.append({'name': 'cta', 'bbox': (x_start, cy1, x_start + cta_w, cy1 + cta_h)})
    
    return element_boxes


# ── Ad / Image helpers ─────────────────────────────────────────────
def generate_product_ad_image(prod, width=640, use_content_aware=True):
    """
    Generate a product ad/poster image using the two-stage content-aware
    layout pipeline + vision-first rendering.

    Pipeline:
        STAGE 1  — PlacementPlanner   (text-only plan, no coords)
        STAGE 2  — LayoutRenderer     (plan → coords + HTML)
        RENDER   — PIL draws elements on the product image
    """
    if prod is None:
        return None
    
    # Generate creative ad copy using LLM (independent of visual pipeline)
    ad_copy = generate_creative_ad_copy(prod)
    
    img_url = prod.get('primary_image') or ''
    price = prod.get('price', 0)
    mrp = prod.get('mrp', 0)
    discount_pct = prod.get('discount_pct', 0)
    category = prod.get('combined_category', '')
    
    # Try content-aware banner generator first (preferred - uses laygen module)
    if use_content_aware and HAS_LAYGEN and generate_content_aware_banner_from_url:
        try:
            bio = generate_content_aware_banner_from_url(
                image_source=img_url,
                product_name=ad_copy['headline'],
                tagline=ad_copy['tagline'],
                price=price,
                mrp=mrp,
                discount_pct=discount_pct,
                category=category,
                cta=ad_copy['cta'],
                output_size=(width, int(width * 4 / 3)),  # 3:4 portrait poster
                use_vlm=True,
            )
            if bio:
                return bio
        except Exception as e:
            if BANNER_DEBUG:
                print(f"[Ad] Content-aware banner failed: {e}, using vision-first pipeline")
    
    # ══════════════════════════════════════════════════════════════════
    # PROFESSIONAL AD PIPELINE
    # ══════════════════════════════════════════════════════════════════
    
    img_url = prod.get('primary_image') or (prod.get('images') and prod.get('images')[0])
    
    # STAGE 1: Load product image (keep original proportions — NOT stretched)
    try:
        if img_url and str(img_url).lower().startswith('http'):
            r = http_requests.get(img_url, timeout=8)
            r.raise_for_status()
            product_img = Image.open(BytesIO(r.content)).convert('RGBA')
        else:
            img_path = os.path.join(os.path.dirname(__file__), img_url or '')
            product_img = Image.open(img_path).convert('RGBA')
    except Exception as e:
        if BANNER_DEBUG:
            print(f"[Ad] Image load failed: {e}, creating placeholder")
        product_img = Image.new('RGBA', (300, 300), color=(200, 200, 200, 255))

    target_h = int(width * 4 / 3)
    
    if BANNER_DEBUG:
        print(f"[Ad] Product image loaded: {product_img.size}, poster={width}x{target_h}")
    
    # STAGE 2: Pixel analysis to decide adaptive layout
    visual_analysis = _vlm_analyze_image_for_banner(img_url)
    analysis_data = visual_analysis.get('analysis', {})
    
    # Build ad elements
    ad_elements = {
        'headline': ad_copy.get('headline', prod.get('Product_Name', '')[:40]),
        'tagline': ad_copy.get('tagline', ''),
        'description': ad_copy.get('description', ''),
        'price': price,
        'mrp': mrp,
        'discount_pct': discount_pct,
        'cta': ad_copy.get('cta', 'Shop Now')
    }
    
    # Get adaptive layout from rule-based plan (uses safe zone analysis)
    placement_plan = _llm_generate_placement_plan(visual_analysis, ad_elements)
    plan = placement_plan.get('plan', {})
    layout_type = plan.get('layout_type', 'bottom_band')
    
    if BANNER_DEBUG:
        print(f"[Ad] Layout chosen: {layout_type}")
    
    # STAGE 3: Professional ad rendering (vibrant background + product compositing)
    rendered_image = _render_professional_ad(
        product_img, ad_elements, layout_type, width, target_h
    )

    # Output as BytesIO
    bio = BytesIO()
    rendered_image.save(bio, format='PNG')
    bio.seek(0)
    return bio


# ── Two-stage layout API endpoint ──────────────────────────────────
@app.route('/api/ad_layout')
def ad_layout_api():
    """
    Return the two-stage layout as JSON + HTML for a product.

    Query params:
      - product_id (required)
      - width  (default 480)
      - height (default 640)

    Response JSON:
      {
        placement_plan: <text>,
        elements: [{element, role, left, top, width, height, z_index}, …],
        html: <string>
      }
    """
    pid = request.args.get('product_id')
    w = request.args.get('width', 480, type=int)
    h = request.args.get('height', 640, type=int)

    try:
        pid = int(pid)
    except Exception:
        return jsonify({'error': 'invalid product_id'}), 400

    prod = PRODUCTS_BY_ID.get(pid)
    if not prod:
        return jsonify({'error': 'product not found'}), 404

    # Build detected_objects from VLM analysis
    img_url = prod.get('primary_image', '')
    visual_analysis = _vlm_analyze_image_for_banner(img_url)
    analysis_data = visual_analysis.get('analysis', {})
    detected_objects = []
    pb = analysis_data.get('product_bounds', {})
    if pb:
        detected_objects.append({
            'label': 'product',
            'bbox': [int(pb.get('x_start', 20) / 100 * w),
                     int(pb.get('y_start', 10) / 100 * h),
                     int((pb.get('x_end', 80) - pb.get('x_start', 20)) / 100 * w),
                     int((pb.get('y_end', 85) - pb.get('y_start', 10)) / 100 * h)],
            'importance': 'high'
        })

    result = run_two_stage_ad_layout(w, h, detected_objects)
    return jsonify(result)


# ── Flask routes ───────────────────────────────────────────────────
@app.route('/') 
def index(): return render_template_string(MAIN_PAGE)

@app.route('/admin')
def admin(): return render_template_string(ADMIN_PAGE)

@app.route('/api/products/featured')
def get_featured():
    o=int(request.args.get('offset',0)); l=int(request.args.get('limit',24))
    return jsonify(FEATURED_CACHE[o:o+l])


@app.route('/api/product_ad')
def product_ad():
  pid = request.args.get('product_id')
  width = request.args.get('width', 640, type=int)
  try:
    pid = int(pid)
  except Exception:
    return jsonify({'error':'invalid product_id'}), 400
  prod = PRODUCTS_BY_ID.get(pid)
  if not prod:
    return jsonify({'error':'product not found'}), 404
  bio = generate_product_ad_image(prod, width=width)
  if bio is None:
    return jsonify({'error':'could not generate image'}), 500
  return send_file(bio, mimetype='image/png', as_attachment=False, download_name=f'product_{pid}_ad.png')


@app.route('/api/content_aware_banner')
def content_aware_banner():
    """
    Generate a content-aware e-commerce banner.
    
    Query params:
      - product_id: (required) Product ID
      - width: Output width (default: 720)
      - height: Output height (default: 900)
    
    This endpoint uses sophisticated content-aware placement:
      - Analyzes product image to detect focal regions
      - Places text elements in optimal positions based on image content
      - Prefers product-adjacent layouts over corner placement
      - Creates professional e-commerce style banners
    """
    pid = request.args.get('product_id')
    width = request.args.get('width', 720, type=int)
    height = request.args.get('height', 900, type=int)
    
    try:
        pid = int(pid)
    except Exception:
        return jsonify({'error': 'invalid product_id'}), 400
    
    prod = PRODUCTS_BY_ID.get(pid)
    if not prod:
        return jsonify({'error': 'product not found'}), 404
    
    # Check if content-aware generation is available
    if not HAS_LAYGEN or generate_content_aware_banner_from_url is None:
        return jsonify({'error': 'content-aware banner generation not available'}), 501
    
    # Generate creative ad copy
    ad_copy = generate_creative_ad_copy(prod)
    
    try:
        bio = generate_content_aware_banner_from_url(
            image_source=prod.get('primary_image', ''),
            product_name=ad_copy['headline'],
            tagline=ad_copy['tagline'],
            price=prod.get('price', 0),
            mrp=prod.get('mrp', 0),
            discount_pct=prod.get('discount_pct', 0),
            category=prod.get('combined_category', ''),
            cta=ad_copy['cta'],
            output_size=(width, height),
        )
        
        if bio is None:
            return jsonify({'error': 'failed to generate banner'}), 500
        
        return send_file(
            bio, 
            mimetype='image/png', 
            as_attachment=False, 
            download_name=f'product_{pid}_banner_{width}x{height}.png'
        )
    except Exception as e:
        print(f"[API] Content-aware banner error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_image_placement')
def analyze_image_placement():
    """
    Analyze a product image and return optimal overlay placement data.
    
    Query params:
      - image_url: (required) URL of the product image
      - card_width: Card width in pixels (default: 222)
      - card_height: Card image height in pixels (default: 158)
    
    Returns JSON with placement data for each UI element:
      - rank_badge: position for #1, #2, #3 badge
      - discount_badge: position for "X% OFF" badge
      - price_zone: position for price display
      - cta_zone: position for CTA button
      - debug: image analysis details
    """
    image_url = request.args.get('image_url', '')
    card_width = request.args.get('card_width', 222, type=int)
    card_height = request.args.get('card_height', 158, type=int)
    
    if not image_url:
        return jsonify({'error': 'image_url required'}), 400
    
    try:
        # Import analyzer from laygen module
        from laygen_pricemapping import ProductRegionAnalyzer, SaliencyDetector
        
        # Fetch and analyze image
        img_resp = http_requests.get(image_url, timeout=10)
        img_resp.raise_for_status()
        img = Image.open(BytesIO(img_resp.content)).convert('RGB')
        
        # Resize to card dimensions for analysis
        img_resized = img.resize((card_width, card_height), Image.LANCZOS)
        
        # Analyze product region and saliency
        analyzer = ProductRegionAnalyzer(grid_resolution=8)
        product_region = analyzer.detect_product_region(img_resized)
        saliency_detector = SaliencyDetector(grid_size=8)
        saliency_map = saliency_detector.compute_saliency_map(img_resized)
        
        # Get product bounding box
        px1, py1, px2, py2 = product_region['bbox']
        prod_cx, prod_cy = product_region['center']
        orientation = product_region['orientation']
        coverage = product_region['product_coverage']
        
        # Analyze saliency in each corner/zone
        g = 8
        zones = {
            'top_left': saliency_map[0:g//3, 0:g//3].mean(),
            'top_right': saliency_map[0:g//3, 2*g//3:g].mean(),
            'bottom_left': saliency_map[2*g//3:g, 0:g//3].mean(),
            'bottom_right': saliency_map[2*g//3:g, 2*g//3:g].mean(),
            'top_center': saliency_map[0:g//4, g//3:2*g//3].mean(),
            'bottom_center': saliency_map[3*g//4:g, g//3:2*g//3].mean(),
        }
        
        # Find safest zones (lowest saliency = safest for overlays)
        safe_zones = sorted(zones.items(), key=lambda x: x[1])
        
        # Determine best positions for each element based on image analysis
        placement = compute_dynamic_placement(
            product_region, zones, safe_zones, 
            card_width, card_height, orientation
        )
        
        # Add debug info
        placement['debug'] = {
            'product_region': {
                'bbox_normalized': [float(px1), float(py1), float(px2), float(py2)],
                'center': [float(prod_cx), float(prod_cy)],
                'orientation': orientation,
                'coverage': float(coverage),
            },
            'zone_saliency': {k: float(v) for k, v in zones.items()},
            'safest_zones': [z[0] for z in safe_zones[:3]],
            'image_analyzed': True,
        }
        
        return jsonify(placement)
        
    except Exception as e:
        print(f"[API] Image analysis error: {e}")
        # Return default placements on error
        return jsonify({
            'rank_badge': {'position': 'top-left', 'top': '8px', 'left': '8px', 'right': 'auto', 'bottom': 'auto'},
            'discount_badge': {'position': 'top-right', 'top': '8px', 'right': '8px', 'left': 'auto', 'bottom': 'auto'},
            'price_zone': {'position': 'bottom-left', 'bottom': '8px', 'left': '8px', 'right': 'auto', 'top': 'auto'},
            'cta_zone': {'position': 'bottom-right', 'bottom': '8px', 'right': '8px', 'left': 'auto', 'top': 'auto'},
            'debug': {'image_analyzed': False, 'error': str(e)},
        })


def compute_dynamic_placement(product_region, zones, safe_zones, card_w, card_h, orientation):
    """
    Compute dynamic overlay placement based on image analysis.
    
    Rules:
    1. Never place overlays over high-saliency (product) regions
    2. Prefer placement adjacent to product, not just corners
    3. Different product shapes get different layouts
    4. Maintain visual hierarchy: Discount > Price > CTA > Rank
    """
    px1, py1, px2, py2 = product_region['bbox']
    prod_cx, prod_cy = product_region['center']
    
    # Convert normalized coords to pixels
    prod_left = int(px1 * card_w)
    prod_right = int(px2 * card_w)
    prod_top = int(py1 * card_h)
    prod_bottom = int(py2 * card_h)
    
    placements = {}
    
    # 1. RANK BADGE - goes in safest corner that doesn't overlap product
    rank_zone = safe_zones[0][0] if safe_zones else 'top_left'
    # If product is centered/left, prefer right side; if right, prefer left
    if prod_cx > 0.55:  # Product on right
        if zones['top_left'] < zones['top_right']:
            rank_zone = 'top_left'
        elif zones['bottom_left'] < zones['bottom_right']:
            rank_zone = 'bottom_left'
        else:
            rank_zone = 'top_left'
    elif prod_cx < 0.45:  # Product on left
        if zones['top_right'] < zones['top_left']:
            rank_zone = 'top_right'
        elif zones['bottom_right'] < zones['bottom_left']:
            rank_zone = 'bottom_right'
        else:
            rank_zone = 'top_right'
    else:  # Product centered
        # For centered product, use corner with lowest saliency
        rank_zone = safe_zones[0][0] if safe_zones else 'top_left'
    
    placements['rank_badge'] = zone_to_css(rank_zone, 'badge', card_w, card_h)
    
    # 2. DISCOUNT BADGE - opposite corner from rank, or adjacent to product
    discount_candidates = ['top_right', 'top_left', 'top_center']
    if rank_zone in discount_candidates:
        discount_candidates.remove(rank_zone)
    
    # Prefer the candidate with lowest saliency
    discount_zone = min(discount_candidates, key=lambda z: zones.get(z, 0.5))
    
    # If product is in bottom half, discount can go above it
    if prod_cy > 0.5 and zones['top_center'] < 0.4:
        discount_zone = 'top_center'
    
    placements['discount_badge'] = zone_to_css(discount_zone, 'badge', card_w, card_h)
    
    # 3. PRICE ZONE - prefer product-adjacent placement
    price_candidates = ['bottom_left', 'bottom_right', 'bottom_center']
    # Remove any zone that overlaps heavily with product
    price_candidates = [z for z in price_candidates if zones.get(z, 0.5) < 0.6]
    if not price_candidates:
        price_candidates = ['bottom_left', 'bottom_center']
    
    # Prefer side opposite to product center
    if prod_cx > 0.55:
        price_zone = 'bottom_left' if 'bottom_left' in price_candidates else price_candidates[0]
    elif prod_cx < 0.45:
        price_zone = 'bottom_right' if 'bottom_right' in price_candidates else price_candidates[0]
    else:
        price_zone = 'bottom_center' if 'bottom_center' in price_candidates else price_candidates[0]
    
    placements['price_zone'] = zone_to_css(price_zone, 'price', card_w, card_h)
    
    # 4. CTA ZONE - opposite side of price, or below product
    if 'left' in price_zone:
        cta_zone = 'bottom_right'
    elif 'right' in price_zone:
        cta_zone = 'bottom_left'
    else:
        cta_zone = 'bottom_right' if zones['bottom_right'] < zones['bottom_left'] else 'bottom_left'
    
    placements['cta_zone'] = zone_to_css(cta_zone, 'cta', card_w, card_h)
    
    # 5. Add overall layout type for CSS class
    if orientation == 'portrait':
        placements['layout_type'] = 'portrait'
    elif orientation == 'landscape':
        placements['layout_type'] = 'landscape'
    else:
        placements['layout_type'] = 'square'
    
    return placements


def zone_to_css(zone_name, element_type, card_w, card_h):
    """Convert a zone name to CSS positioning values."""
    # Base margins
    m = 8  # margin from edge
    
    # Badge sizes
    badge_h = 22
    price_h = 24
    cta_h = 28
    
    css = {
        'position': zone_name,
        'top': 'auto',
        'right': 'auto',
        'bottom': 'auto',
        'left': 'auto',
    }
    
    if 'top' in zone_name:
        css['top'] = f'{m}px'
    if 'bottom' in zone_name:
        css['bottom'] = f'{m}px'
    if 'left' in zone_name and 'center' not in zone_name:
        css['left'] = f'{m}px'
    if 'right' in zone_name and 'center' not in zone_name:
        css['right'] = f'{m}px'
    if 'center' in zone_name:
        css['left'] = '50%'
        css['transform'] = 'translateX(-50%)'
    
    return css


@app.route('/api/search')
def search():
    q=request.args.get('q',''); o=int(request.args.get('offset',0)); l=int(request.args.get('limit',24))
    mask=PRODUCTS_DF['Product_Name'].str.contains(q,case=False,na=False)
    filtered=PRODUCTS_DF[mask][PRODUCT_FIELDS]
    return jsonify({'items':filtered.iloc[o:o+l].to_dict('records'),'total':len(filtered)})

@app.route('/api/track_view',methods=['POST'])
def track_view(): user_id=get_session_user_id(); record_interaction(user_id,request.json.get('product_id'),'view'); return jsonify({'status':'ok','user_id':user_id})

@app.route('/api/track_cart',methods=['POST'])
def track_cart(): record_interaction(get_session_user_id(),request.json.get('product_id'),'cart'); return jsonify({'status':'ok'})

@app.route('/api/track_wishlist',methods=['POST'])
def track_wishlist(): record_interaction(get_session_user_id(),request.json.get('product_id'),'wishlist'); return jsonify({'status':'ok'})

@app.route('/api/track_buy',methods=['POST'])
def track_buy(): record_interaction(get_session_user_id(),request.json.get('product_id'),'buy'); return jsonify({'status':'ok'})

@app.route('/api/recommendations',methods=['POST'])
def get_recommendations():
    history=request.json.get('history',[]); use_llm=request.json.get('llm_rerank',True)
    recs=get_recommendations_for_user(history,top_k=12 if use_llm else 6)
    if use_llm and len(recs)>3 and MODEL is not None: recs=llm_rerank_candidates(recs,history)
    return jsonify(recs[:6])

@app.route('/api/prediction_debug',methods=['POST'])
def prediction_debug():
    """Get detailed intermediate results for understanding predictions."""
    history = request.json.get('history', [])
    user_id = get_session_user_id()
    
    debug_info = {
        'user_id': user_id[:8] + '...',
        'history_length': len(history),
        'model_active': MODEL is not None,
        'stages': [],
        'intents': [],
        'scoring_breakdown': [],
        'final_recommendations': [],
        'pipeline_trace': []  # Detailed step-by-step tensor trace
    }
    
    # Parse history
    seq = []  # List of product IDs
    action_strings = []  # List of action strings (mapped for model)
    categories = []
    price_range = []
    action_counts = {'view': 0, 'cart': 0, 'wishlist': 0, 'buy': 0}
    history_items = []
    
    for h in history[-20:]:  # Last 20 items
        if isinstance(h, dict):
            pid, act = h.get('pid'), h.get('action', 'view')
        else:
            try: pid, act = h
            except: continue
        
        if isinstance(pid, int) and pid in PRODUCTS_BY_ID:
            model_action = ACTION_MAP.get(act, 'view')  # Map app action to model action
            # No duplication - each event is recorded once with its action type
            seq.append(pid)
            action_strings.append(model_action)
            action_counts[act] = action_counts.get(act, 0) + 1
            
            prod = PRODUCTS_BY_ID[pid]
            categories.append(prod.get('main_category', ''))
            price = prod.get('price', 0)
            if price > 0:
                price_range.append(price)
            history_items.append({
                'name': prod.get('Product_Name', '')[:40],
                'category': prod.get('main_category', ''),
                'action': act
            })
    
    # Stage 1: History Analysis
    avg_price = np.mean(price_range) if price_range else 0
    category_counts = {}
    for c in categories:
        category_counts[c] = category_counts.get(c, 0) + 1
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    debug_info['stages'].append({
        'name': '1. History Analysis',
        'icon': '📊',
        'details': [
            {'label': 'Total Actions', 'value': sum(action_counts.values())},
            {'label': 'Views', 'value': action_counts['view']},
            {'label': 'Cart Adds', 'value': action_counts['cart']},
            {'label': 'Wishlist', 'value': action_counts['wishlist']},
            {'label': 'Purchases', 'value': action_counts['buy']},
            {'label': 'Avg Price', 'value': f'₹{int(avg_price):,}' if avg_price else 'N/A'},
            {'label': 'Top Categories', 'value': ', '.join([c[0][:15] for c in top_categories[:3]]) if top_categories else 'None'}
        ],
        'items': history_items[-8:]
    })
    
    if MODEL is None:
        debug_info['stages'].append({
            'name': '2. Fallback Mode',
            'icon': '⚠️',
            'details': [
                {'label': 'Status', 'value': 'Model not trained'},
                {'label': 'Method', 'value': 'Category + Price matching'}
            ]
        })
    elif len(seq) < MIN_SEQ_LEN:
        debug_info['stages'].append({
            'name': '2. Insufficient History',
            'icon': '⚠️',
            'details': [
                {'label': 'Status', 'value': f'Need {MIN_SEQ_LEN} events, have {len(seq)}'},
                {'label': 'Method', 'value': 'Category + Price matching fallback'}
            ]
        })
    else:
        # Stage 2: Model Inference
        try:
            MODEL.eval()
            with torch.no_grad():
                # Ensure minimum sequence length
                if len(seq) < MIN_SEQ_LEN:
                    raise ValueError(f"Sequence length {len(seq)} below minimum {MIN_SEQ_LEN}")
                
                # Build product IDs tensor
                pids_tensor = torch.tensor([seq[-50:]], dtype=torch.long, device=device)  # [1, T]
                
                # Build action IDs tensor using ActionEmbeddingLayer
                aids_list = ActionEmbeddingLayer.encode_actions(action_strings[-50:])
                aids_tensor = torch.tensor([aids_list], dtype=torch.long, device=device)  # [1, T]
                
                # Call model with both tensors
                outputs = MODEL.encode_session(pids_tensor, aids_tensor)
                
                assignment_weights = outputs['assign_weights'][0].cpu().numpy()  # [T, K]
                intent_vectors = outputs['intent_vectors'][0].cpu().numpy()  # [K, D]
                alpha_k = outputs['alpha_k'][0].cpu().numpy()  # [K] - intent gating weights
                recency_weights = outputs.get('recency_weights', torch.zeros(len(seq)))[0].cpu().numpy() if 'recency_weights' in outputs else None
                
                K = intent_vectors.shape[0]
                T = pids_tensor.shape[1]
                D = intent_vectors.shape[1]
                valid_mask = (pids_tensor[0] != 0).cpu().numpy()
                padded = seq[-50:]  # For compatibility with existing code
                
                # Build detailed pipeline trace
                pipeline_trace = []
                
                # Step 1: Input Encoding
                pipeline_trace.append({
                    'step': 1,
                    'name': 'Action String → Action ID',
                    'input': f'Action strings: {action_strings[-8:]}',
                    'operation': 'ACTION_TO_IDX mapping: view→0, click→1, hover→2, wishlist→3',
                    'output': f'Action IDs: {aids_list[-8:]}',
                    'shape': f'[{len(aids_list)}]'
                })
                
                # Step 2: Tensor Building
                pipeline_trace.append({
                    'step': 2,
                    'name': 'Build Input Tensors',
                    'input': f'Product IDs: {seq[-8:]}',
                    'operation': 'torch.tensor() with batch dim',
                    'output': f'pids_tensor shape, aids_tensor shape',
                    'shape': f'[1, {T}] each'
                })
                
                # Step 3: Product Feature Extraction
                prod_embs = outputs['prod_embs'][0].cpu().numpy()  # [T, D]
                pipeline_trace.append({
                    'step': 3,
                    'name': 'Product Feature Extraction',
                    'input': f'Flat product IDs [{T}]',
                    'operation': 'CatalogFeatureBuilder.get_feature_tensors()',
                    'output': f'word_ids, cat1/2/3_ids, numerics',
                    'shape': f'word_ids [{T}, 16], cats [{T}], numerics [{T}, 2]'
                })
                
                # Step 4-7: Feature Fusion
                pipeline_trace.append({
                    'step': 4,
                    'name': 'Feature Fusion (ProductFeatureEncoder)',
                    'input': 'title_emb + cat1/2/3_emb + numeric_proj',
                    'operation': 'Concat → Linear(128→128) → LayerNorm → GELU → Linear',
                    'output': f'prod_embs: mean={prod_embs.mean():.4f}, std={prod_embs.std():.4f}',
                    'shape': f'[{T}, {D}]'
                })
                
                # Step 5: Action Embedding
                pipeline_trace.append({
                    'step': 5,
                    'name': 'Action Embedding + Addition',
                    'input': f'action_ids [{T}]',
                    'operation': 'Embedding(4, {D}) → Add to prod_embs',
                    'output': 'Combined product+action embeddings',
                    'shape': f'[{T}, {D}]'
                })
                
                # Step 6: Positional + Transformer
                seq_repr = outputs['seq_repr'][0].cpu().numpy()  # [T, D]
                pipeline_trace.append({
                    'step': 6,
                    'name': 'Positional Encoding + Transformer',
                    'input': f'x [{T}, {D}]',
                    'operation': '+ pos_emb → LayerNorm → 2x CausalAttentionBlock',
                    'output': f'seq_repr: mean={seq_repr.mean():.4f}, std={seq_repr.std():.4f}',
                    'shape': f'[{T}, {D}]'
                })
                
                # Step 7: Intent Gating
                alpha_str = ', '.join([f'{a:.2f}' for a in alpha_k])
                pipeline_trace.append({
                    'step': 7,
                    'name': 'Intent Gating (alpha_k)',
                    'input': f'seq_summary [{D}] + last_cat embeddings',
                    'operation': 'gating_net() → softmax(logits/0.5)',
                    'output': f'alpha_k: [{alpha_str}]',
                    'shape': f'[{K}]'
                })
                
                # Step 8: Per-Position Assignment
                assign_str = ', '.join([f'{w:.2f}' for w in assignment_weights[-1]])
                pipeline_trace.append({
                    'step': 8,
                    'name': 'Per-Position Intent Assignment',
                    'input': f'seq_repr [{T}, {D}]',
                    'operation': f'Linear({D}→{K}) → softmax over T',
                    'output': f'assign_weights: last row = [{assign_str}]',
                    'shape': f'[{T}, {K}]'
                })
                
                # Step 9: Intent Vector Building
                norms_str = ', '.join([f'{np.linalg.norm(intent_vectors[k]):.2f}' for k in range(K)])
                pipeline_trace.append({
                    'step': 9,
                    'name': 'Build Intent Vectors',
                    'input': 'assign_weights + seq_repr',
                    'operation': 'einsum("bt,btd→bd") → intent_transform per k',
                    'output': f'intent_vectors: norms = [{norms_str}]',
                    'shape': f'[{K}, {D}]'
                })
                
                # Step 10: Session Representation
                session_repr = outputs['session_repr'][0].cpu().numpy()  # [D]
                pipeline_trace.append({
                    'step': 10,
                    'name': 'Session Representation (NO pooling)',
                    'input': 'alpha_k × intent_vectors',
                    'operation': f'Σ_k (alpha_k * intent_k)',
                    'output': f'session_repr: norm={np.linalg.norm(session_repr):.3f}, mean={session_repr.mean():.4f}',
                    'shape': f'[{D}]'
                })
                
                # Step 11: Per-Product Contribution
                per_product_contrib = outputs['per_product_contrib'][0].cpu().numpy()  # [T]
                top_contrib_idx = np.argsort(per_product_contrib)[-3:][::-1]
                top_contrib_items = []
                for idx in top_contrib_idx:
                    if idx < len(padded) and padded[idx] in PRODUCTS_BY_ID:
                        top_contrib_items.append(f'{PRODUCTS_BY_ID[padded[idx]].get("Product_Name", "")[:20]} ({per_product_contrib[idx]:.2f})')
                contrib_str = '; '.join(top_contrib_items[:3])
                pipeline_trace.append({
                    'step': 11,
                    'name': 'Per-Product Contribution',
                    'input': 'alpha_k, assign_weights',
                    'operation': 'einsum("bk,btk→bt")',
                    'output': f'Top contributors: {contrib_str}',
                    'shape': f'[{T}]'
                })
                
                # Step 12: Scoring
                preds = MODEL.predict(pids_tensor, aids_tensor)
                scores = preds['scores'][0].cpu().numpy()
                pipeline_trace.append({
                    'step': 12,
                    'name': 'Final Scoring',
                    'input': 'session_repr → score_proj → normalize',
                    'operation': 'cosine_sim(s, all_product_embs)',
                    'output': f'scores: max={scores.max():.3f}, min={scores.min():.3f}, top_idx={int(np.argmax(scores))+1}',
                    'shape': f'[{len(scores)}]'
                })
                
                debug_info['pipeline_trace'] = pipeline_trace
                
                # Intent analysis - use alpha_k for intent percentages
                intent_data = []
                for k in range(K):
                    # Use alpha_k for the intent weight percentage
                    intent_weight_pct = float(alpha_k[k]) * 100
                    
                    # Find dominant items for this intent using assignment_weights
                    if valid_mask.sum() > 0:
                        weights_k = assignment_weights[valid_mask, k]
                        top_idx = np.argsort(weights_k)[-3:][::-1]
                        valid_pids = [padded[i] for i in np.where(valid_mask)[0]]
                        top_items = []
                        for idx in top_idx:
                            if idx < len(valid_pids):
                                pid = valid_pids[idx]
                                if pid in PRODUCTS_BY_ID:
                                    top_items.append(PRODUCTS_BY_ID[pid].get('Product_Name', '')[:25])
                    else:
                        top_items = []
                    
                    intent_data.append({
                        'intent_id': k + 1,
                        'avg_weight': round(intent_weight_pct, 1),  # Use alpha_k percentage
                        'vector_norm': round(float(np.linalg.norm(intent_vectors[k])), 2),
                        'top_items': top_items[:2]
                    })
                
                debug_info['intents'] = intent_data
                
                debug_info['stages'].append({
                    'name': '2. Multi-Intent Extraction',
                    'icon': '🧠',
                    'details': [
                        {'label': 'Encoder', 'value': 'Transformer Self-Attention'},
                        {'label': 'Num Intents (K)', 'value': K},
                        {'label': 'Hidden Dim', 'value': MODEL.hidden_dim},
                        {'label': 'Sequence Length', 'value': int(valid_mask.sum())},
                        {'label': 'Routing Method', 'value': 'Gumbel-Softmax' if getattr(MODEL, 'use_gumbel', False) else 'Softmax'}
                    ]
                })
                
                # Stage 3: Recency Weighting (optional - may not exist in model)
                recency_scale = getattr(MODEL, 'recency_scale', None)
                if recency_weights is not None and recency_scale is not None:
                    recency_info = []
                    valid_recency = recency_weights[valid_mask] if valid_mask.sum() > 0 else []
                    if len(valid_recency) > 0:
                        for i, w in enumerate(valid_recency[-5:]):
                            recency_info.append({'pos': f'T-{5-i}', 'weight': round(float(w) * 100, 1)})
                    
                    debug_info['stages'].append({
                        'name': '3. Recency Weighting',
                        'icon': '⏱️',
                        'details': [
                            {'label': 'Method', 'value': 'Exponential Decay'},
                            {'label': 'Scale Factor', 'value': round(float(recency_scale.item()), 3)}
                        ],
                        'weights': recency_info
                    })
                
                # Stage 4: Scoring
                preds = MODEL.predict(pids_tensor, aids_tensor)
                scores = preds['scores'][0].cpu().numpy()
                top_indices = np.argsort(scores)[-10:][::-1]
                
                scoring_breakdown = []
                for idx in top_indices:
                    pid = idx + 1
                    if pid in PRODUCTS_BY_ID:
                        prod = PRODUCTS_BY_ID[pid]
                        scoring_breakdown.append({
                            'product': prod.get('Product_Name', '')[:35],
                            'category': prod.get('main_category', ''),
                            'score': round(float(scores[idx]), 3),
                            'in_history': pid in seq
                        })
                
                debug_info['scoring_breakdown'] = scoring_breakdown[:8]
                
                debug_info['stages'].append({
                    'name': '4. Item Scoring',
                    'icon': '🎯',
                    'details': [
                        {'label': 'Method', 'value': 'Intent-Weighted Dot Product'},
                        {'label': 'Normalization', 'value': 'L2 Normalized'},
                        {'label': 'Top Score', 'value': round(float(scores.max()), 3)},
                        {'label': 'Score Range', 'value': f'{round(float(scores.min()), 2)} to {round(float(scores.max()), 2)}'}
                    ]
                })
                
                # Stage 5: Prototype Clustering Analysis (optional - method may not exist)
                if hasattr(MODEL, 'get_prototype_analysis'):
                    try:
                        proto_analysis = MODEL.get_prototype_analysis()
                        cos_sims = outputs.get('cosine_similarities', None)
                        clustering_loss = outputs.get('clustering_loss', torch.tensor(0.0))
                        
                        cluster_details = [
                            {'label': 'Num Prototypes', 'value': proto_analysis.get('num_intents', K)},
                            {'label': 'Diversity Score', 'value': f"{proto_analysis.get('diversity_score', 0)*100:.1f}%"},
                            {'label': 'Clustering Loss', 'value': f"{float(clustering_loss):.4f}"}
                        ]
                        
                        # Show prototype similarity matrix summary
                        sim_matrix = proto_analysis.get('prototype_similarity_matrix', [])
                        if sim_matrix:
                            avg_off_diag = sum(sim_matrix[i][j] for i in range(len(sim_matrix)) for j in range(len(sim_matrix)) if i != j) / (len(sim_matrix) * (len(sim_matrix) - 1) + 1e-8)
                            cluster_details.append({'label': 'Avg Proto Similarity', 'value': f"{avg_off_diag:.3f}"})
                        
                        # Show cosine similarity to prototypes for recent items
                        if cos_sims is not None:
                            cos_sims_np = cos_sims[0].cpu().numpy()  # [L, K]
                            valid_cos = cos_sims_np[valid_mask] if valid_mask.sum() > 0 else cos_sims_np
                            if len(valid_cos) > 0:
                                # Average similarity to each prototype
                                proto_affinities = []
                                for k in range(K):
                                    avg_sim = valid_cos[:, k].mean()
                                    proto_affinities.append({'proto': f'P{k+1}', 'avg_sim': f"{avg_sim:.3f}"})
                                cluster_details.append({'label': 'Proto Affinities', 'value': ', '.join([f"{p['proto']}:{p['avg_sim']}" for p in proto_affinities])})
                    
                        debug_info['stages'].append({
                            'name': '5. Prototype Clustering',
                            'icon': '🔮',
                            'details': cluster_details
                        })
                        debug_info['prototype_analysis'] = proto_analysis
                    except Exception as pe:
                        debug_info['stages'].append({
                            'name': '5. Prototype Analysis',
                            'icon': '⚠️',
                            'details': [{'label': 'Error', 'value': str(pe)[:50]}]
                        })
                
        except Exception as e:
            debug_info['stages'].append({
                'name': '2. Model Error',
                'icon': '❌',
                'details': [{'label': 'Error', 'value': str(e)[:50]}]
            })
    
    # Get final recommendations
    recs = get_recommendations_for_user(history, top_k=6)
    debug_info['final_recommendations'] = [
        {
            'name': r['product'].get('Product_Name', '')[:40],
            'reason': r.get('reason', ''),
            'score': round(r.get('score', 0), 2),
            'intent': r.get('intent_label', '')
        }
        for r in recs[:6]
    ]
    
    return jsonify(debug_info)

@app.route('/api/explain',methods=['POST'])
def explain_recommendation():
    data=request.json or {}
    pname=data.get('product_name',''); pcat=data.get('product_category',''); reason=data.get('reason','')
    il=data.get('intent_label',''); intents=data.get('intents',[]); ctx=data.get('context',{})
    ri=ctx.get('recent_items',[]); ra=ctx.get('recent_actions',[]); rc=ctx.get('recent_categories',[])
    prompt=f"You are a recommendation explainer for an e-commerce store. Explain in 2-3 short sentences WHY this product was recommended. Be specific and natural.\n\nProduct: {pname}\nCategory: {pcat}\nReason: {reason}\nIntent: {il}\n"
    if intents:
        intent_str = ', '.join(str(i['label'])+' ('+str(i['pct'])+'%)' for i in intents[:3])
        prompt += f"Intent mix: {intent_str}\n"
    if ra: prompt+=f"Recent actions: {'; '.join(ra[-6:])}\n"
    elif ri: prompt+=f"Recently viewed: {', '.join(r[:50] for r in ri[-5:])}\n"
    prompt+="\nGenerate explanation (2-3 sentences max):"
    try:
        exp = call_gemini_llm(prompt, max_tokens=200, temperature=0.7)
        if exp:
            return jsonify({'explanation':exp})
        else:
            raise Exception("Empty response")
    except:
        fb=f"Recommended based on your interest in {il or 'similar items'}."
        if ri: fb+=f" Matches your recent browsing of {ri[-1][:40]}."
        return jsonify({'explanation':fb})

@app.route('/api/stats')
def get_stats():
    return jsonify({'num_products':len(PRODUCTS_DF),'num_users':len(USER_SESSIONS),'num_interactions':sum(len(v) for v in USER_SESSIONS.values()),'model_status':'Trained' if MODEL else 'Not Trained'})

@app.route('/api/train',methods=['POST'])
def train():
    global TRAINING_PROGRESS
    epochs=15 if request.json.get('advanced') else 10
    TRAINING_PROGRESS={'status':'training','epoch':0,'total_epochs':epochs,'loss_s2i':0.0,'loss_s2s':0.0,'loss_entropy':0.0,'loss_total':0.0,'metrics':{},'batch_progress':0.0,'time_elapsed':0}
    Thread(target=train_model_background,args=(epochs,),daemon=True).start()
    return jsonify({'status':'Training started'})

@app.route('/api/training_progress')
def training_progress(): return jsonify(TRAINING_PROGRESS)

@app.route('/api/reset_training',methods=['POST'])
def reset_training():
    global MODEL,TRAINING_PROGRESS,AUTO_TRAIN_LOCK,USER_SESSIONS,POPULARITY_COUNTER,FEATURED_CACHE
    if TRAINING_PROGRESS.get('status')=='training': return jsonify({'status':'error','message':'Cannot reset while training'}),400
    MODEL=None; AUTO_TRAIN_LOCK=False; USER_SESSIONS={}; POPULARITY_COUNTER=defaultdict(int)
    USER_CART.clear(); USER_WISHLIST.clear(); USER_PURCHASES.clear()
    TRAINING_PROGRESS={'status':'idle','epoch':0,'total_epochs':0,'loss_s2i':0.0,'loss_s2s':0.0,'loss_entropy':0.0,'loss_total':0.0,'metrics':{},'batch_progress':0.0,'time_elapsed':0}
    s=PRODUCTS_CACHE.copy(); random.shuffle(s); FEATURED_CACHE=s
    for f in ['best_model.pth','final_model.pth']:
        if os.path.exists(f): os.remove(f)
    return jsonify({'status':'ok'})

@app.route('/api/users')
def list_users():
    users=[]
    for uid,hist in USER_SESSIONS.items():
        acts=defaultdict(int)
        for h in hist: acts[h.get('action','view') if isinstance(h,dict) else 'view']+=1
        users.append({'user_id':uid,'interactions':len(hist),'views':acts.get('view',0),'carts':acts.get('cart',0),'wishlists':acts.get('wishlist',0),'purchases':acts.get('buy',0)})
    return jsonify(sorted(users,key=lambda u: u['interactions'],reverse=True))

@app.route('/api/switch_user',methods=['POST'])
def switch_user():
    uid=request.json.get('user_id')
    if not uid or uid not in USER_SESSIONS: return jsonify({'status':'error','message':'User not found'}),404
    session['user_id']=uid; session.modified=True
    hist=[h if isinstance(h,dict) else {'pid':h[0],'action':h[1],'ts':0} for h in USER_SESSIONS[uid]]
    return jsonify({'status':'ok','user_id':uid,'history':hist,'cart':USER_CART.get(uid,[]),'wishlist':USER_WISHLIST.get(uid,[]),'purchases':USER_PURCHASES.get(uid,[])})

@app.route('/api/current_user')
def current_user():
    uid=get_session_user_id()
    return jsonify({'user_id':uid,'interactions':len(USER_SESSIONS.get(uid,[]))})

@app.route('/api/create_user',methods=['POST'])
def create_new_user():
    uid=hashlib.sha256(f"{time.time()}-{random.random()}".encode()).hexdigest()[:16]
    USER_SESSIONS[uid]=[]; session['user_id']=uid; session.modified=True
    return jsonify({'status':'ok','user_id':uid})

@app.route('/api/categories')
def get_categories():
    cats=PRODUCTS_DF.groupby('main_category').size().reset_index(name='count')
    return jsonify([{'name':r['main_category'],'count':r['count']} for _,r in cats.iterrows()])

@app.route('/api/products/category')
def get_by_category():
    cat=request.args.get('cat',''); o=int(request.args.get('offset',0)); l=int(request.args.get('limit',24))
    filtered=PRODUCTS_DF[PRODUCTS_DF['main_category']==cat][PRODUCT_FIELDS]
    return jsonify({'items':filtered.iloc[o:o+l].to_dict('records'),'total':len(filtered)})

@app.route('/api/user/cart')
def get_user_cart():
    uid=get_session_user_id()
    return jsonify([PRODUCTS_BY_ID[p] for p in USER_CART.get(uid,[]) if p in PRODUCTS_BY_ID])

@app.route('/api/user/wishlist')
def get_user_wishlist():
    uid=get_session_user_id()
    return jsonify([PRODUCTS_BY_ID[p] for p in USER_WISHLIST.get(uid,[]) if p in PRODUCTS_BY_ID])

@app.route('/api/user/purchases')
def get_user_purchases():
    uid=get_session_user_id()
    return jsonify([PRODUCTS_BY_ID[p] for p in USER_PURCHASES.get(uid,[]) if p in PRODUCTS_BY_ID])

@app.route('/api/user/cart/remove',methods=['POST'])
def remove_from_cart():
    uid=get_session_user_id(); pid=request.json.get('product_id')
    if pid and pid in USER_CART.get(uid,[]): USER_CART[uid].remove(pid)
    return jsonify({'status':'ok'})

@app.route('/api/user/wishlist/remove',methods=['POST'])
def remove_from_wishlist():
    uid=get_session_user_id(); pid=request.json.get('product_id')
    if pid and pid in USER_WISHLIST.get(uid,[]): USER_WISHLIST[uid].remove(pid)
    return jsonify({'status':'ok'})


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    HTML TEMPLATES                               ║
# ╚══════════════════════════════════════════════════════════════════╝

MAIN_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DisReq — Smart Store</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0a12;--surface:#13131f;--surface2:#1a1a2e;
  --accent:#f5c518;--accent2:#e85d7a;--accent3:#00d4aa;
  --text:#f0eff5;--muted:#7a7a94;--border:rgba(245,197,24,0.12);
  --radius:14px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

.navbar{background:rgba(10,10,18,0.95);backdrop-filter:blur(20px);padding:14px 30px;
  display:flex;justify-content:space-between;align-items:center;
  position:sticky;top:0;z-index:90;border-bottom:1px solid var(--border)}
.brand{font-family:'Playfair Display',serif;font-size:1.5rem;color:var(--accent);letter-spacing:.5px;cursor:pointer}
.brand small{display:block;font-family:'DM Sans',sans-serif;font-size:.6rem;color:var(--muted);
  font-weight:400;letter-spacing:3px;text-transform:uppercase;margin-top:1px}
.search-bar{display:flex;gap:8px;flex:0 1 400px}
.search-bar input{flex:1;padding:10px 18px;border:1px solid var(--border);border-radius:30px;
  background:var(--surface2);color:var(--text);font-size:.88rem;outline:none;transition:.3s}
.search-bar input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(245,197,24,.08)}
.search-bar input::placeholder{color:var(--muted)}
.search-bar button{padding:10px 20px;border:none;border-radius:30px;background:var(--accent);
  color:#0a0a12;font-weight:700;cursor:pointer;font-size:.82rem;transition:.2s}
.search-bar button:hover{filter:brightness(1.1)}
.nav-links{display:flex;gap:8px;align-items:center}
.nav-links a{color:var(--muted);text-decoration:none;font-size:.8rem;transition:.2s;
  padding:6px 12px;border-radius:20px;display:flex;align-items:center;gap:4px}
.nav-links a:hover{color:var(--text);background:var(--surface2)}
.badge{background:var(--accent2);color:#fff;border-radius:10px;padding:1px 7px;font-size:.62rem;font-weight:700}

.container{max-width:1360px;margin:0 auto;padding:22px 26px}

.stats-strip{display:flex;gap:14px;margin-bottom:18px;padding:14px 20px;
  background:var(--surface);border-radius:var(--radius);border:1px solid var(--border)}
.si{text-align:center;flex:1}
.si .v{font-size:1.2rem;font-weight:700;color:var(--accent);font-family:'Playfair Display',serif}
.si .l{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-top:2px}

.cats{display:flex;gap:7px;overflow-x:auto;padding:2px 0 14px;scrollbar-width:none}
.cats::-webkit-scrollbar{display:none}
.cat-chip{padding:6px 15px;background:var(--surface2);border:1px solid var(--border);
  border-radius:20px;cursor:pointer;white-space:nowrap;font-size:.78rem;color:var(--muted);transition:.2s}
.cat-chip:hover,.cat-chip.active{background:var(--accent);color:#0a0a12;border-color:var(--accent);font-weight:600}

.sec-hdr{display:flex;justify-content:space-between;align-items:center;margin:18px 0 14px}
.sec-hdr h3{font-family:'Playfair Display',serif;font-size:1.2rem;color:var(--text)}
.sec-hdr span{font-size:.78rem;color:var(--muted)}

.product-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-bottom:20px}
.product-card{background:var(--surface);border-radius:var(--radius);overflow:hidden;
  border:1px solid transparent;transition:all .25s;cursor:pointer;position:relative}
.product-card:hover{border-color:var(--border);transform:translateY(-4px);box-shadow:0 10px 35px rgba(0,0,0,.4)}
.product-card img{width:100%;height:185px;object-fit:contain;background:var(--surface2);padding:10px;transition:.3s}
.card-body{padding:13px}
.card-name{font-size:.83rem;font-weight:500;color:var(--text);display:-webkit-box;-webkit-line-clamp:2;
  -webkit-box-orient:vertical;overflow:hidden;line-height:1.44;min-height:2.5em}
.card-cat{font-size:.68rem;color:var(--muted);margin:4px 0}
.price-row{display:flex;align-items:center;gap:7px;margin-top:5px;flex-wrap:wrap}
.card-price{font-size:.98rem;font-weight:700;color:var(--accent);font-family:'Playfair Display',serif}
.card-mrp{font-size:.72rem;color:var(--muted);text-decoration:line-through}
.card-disc{background:rgba(232,93,122,.15);color:var(--accent2);border-radius:4px;padding:1px 6px;font-size:.62rem;font-weight:700}
.card-actions{display:flex;gap:5px;margin-top:9px}
.card-actions button{flex:1;padding:7px 4px;border:none;border-radius:8px;font-size:.7rem;font-weight:600;cursor:pointer;transition:.2s}
.bw{background:rgba(232,93,122,.1);color:var(--accent2)}.bw:hover{background:rgba(232,93,122,.22)}
.bc{background:rgba(245,197,24,.1);color:var(--accent)}.bc:hover{background:rgba(245,197,24,.22)}
.bb{background:var(--accent);color:#0a0a12}.bb:hover{filter:brightness(1.1)}

.load-more{display:block;margin:6px auto 28px;padding:11px 36px;background:transparent;
  color:var(--accent);border:1.5px solid var(--accent);border-radius:30px;
  font-size:.82rem;font-weight:600;cursor:pointer;transition:.2s}
.load-more:hover{background:var(--accent);color:#0a0a12}

/* ── REC TRIGGER ── */
#rec-trigger{
  position:fixed;bottom:26px;right:26px;z-index:110;
  background:linear-gradient(135deg,var(--accent) 0%,#e8a800 100%);
  color:#0a0a12;border:none;border-radius:50px;padding:13px 20px;
  font-family:'DM Sans',sans-serif;font-weight:700;font-size:.86rem;
  cursor:pointer;display:none;box-shadow:0 6px 28px rgba(245,197,24,.4);
  align-items:center;gap:8px;transition:all .3s;
}
#rec-trigger:hover{transform:translateY(-3px);box-shadow:0 10px 36px rgba(245,197,24,.5)}
.pulse-dot{width:8px;height:8px;background:var(--accent2);border-radius:50%;animation:pdot 1.5s ease-in-out infinite}
@keyframes pdot{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.6);opacity:.5}}

/* ── REC DRAWER ── */
#rec-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:115;backdrop-filter:blur(5px)}
#rec-backdrop.show{display:block}

#rec-drawer{
  position:fixed;bottom:0;left:0;right:0;z-index:120;
  background:var(--surface);border-top:1px solid var(--border);
  border-radius:22px 22px 0 0;
  transform:translateY(100%);transition:transform .45s cubic-bezier(.16,1,.3,1);
  display:flex;flex-direction:column;box-shadow:0 -16px 60px rgba(0,0,0,.7);
  max-height:90vh;
}
#rec-drawer.open{transform:translateY(0)}

.drawer-handle{width:42px;height:4px;background:rgba(255,255,255,.12);border-radius:3px;margin:12px auto 0;cursor:pointer}

.drawer-hdr{padding:14px 26px 12px;display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--border)}
.drawer-title{font-family:'Playfair Display',serif;font-size:1.35rem;color:var(--accent)}
.drawer-sub{font-size:.75rem;color:var(--muted);margin-top:2px}
.drawer-close{background:var(--surface2);border:none;color:var(--muted);width:32px;height:32px;
  border-radius:50%;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.2s}
.drawer-close:hover{color:var(--text);background:var(--surface)}

.rec-scroll{overflow-x:auto;overflow-y:hidden;padding:18px 26px 26px;
  display:flex;gap:14px;scrollbar-width:thin;scrollbar-color:var(--border) transparent;flex:1}

/* ── REC POPUP CARDS ── */
.rpc{flex:0 0 222px;background:var(--surface2);border-radius:16px;overflow:hidden;
  border:1px solid transparent;transition:all .28s;cursor:pointer;display:flex;flex-direction:column;position:relative}
.rpc:hover{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 14px 40px rgba(0,0,0,.45);transform:translateY(-4px)}

/* Dynamic overlay positioning - NO FIXED POSITIONS */
.rpc-overlay{position:absolute;z-index:2;padding:3px 9px;border-radius:18px;font-size:.6rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase;transition:all .2s}
.rpc-badge{position:absolute;z-index:2;padding:3px 9px;border-radius:18px;font-size:.6rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
/* Dynamic positions applied via inline styles - NO DEFAULTS */
.rpc-rank{z-index:3}
.rpc-discount{z-index:4}
.rb-hot{background:var(--accent2);color:#fff}
.rb-deal{background:var(--accent3);color:#0a0a12}
.rb-trend{background:#7c5cfc;color:#fff}
.rb-rec{background:var(--accent);color:#0a0a12}

.rpc-img{width:100%;height:158px;object-fit:contain;background:#07070f;padding:10px}

.rpc-body{padding:12px 13px 14px;display:flex;flex-direction:column;gap:5px;flex:1}
.rpc-tagline{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent2)}
.rpc-name{font-size:.84rem;font-weight:500;color:var(--text);line-height:1.38;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.rpc-price-row{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.rpc-price{font-family:'Playfair Display',serif;font-size:1.05rem;color:var(--accent);font-weight:700}
.rpc-mrp{font-size:.7rem;color:var(--muted);text-decoration:line-through}
.rpc-save{font-size:.62rem;color:var(--accent3);font-weight:700;background:rgba(0,212,170,.1);padding:2px 6px;border-radius:8px}
.rpc-intent{font-size:.65rem;color:var(--muted);padding:3px 8px;background:var(--surface);border-radius:5px;border:1px solid var(--border)}
.rpc-explain{margin-top:5px;padding:7px 9px;background:var(--surface);border-radius:7px;
  border-left:3px solid var(--accent);font-size:.7rem;color:var(--muted);line-height:1.5;display:none}
.rpc-explain.show{display:block;animation:fup .22s ease}
@keyframes fup{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}

.rpc-ctas{display:flex;gap:6px;margin-top:auto;padding-top:8px}
.rpc-cta{flex:1;padding:8px 6px;border:none;border-radius:9px;font-size:.7rem;font-weight:700;cursor:pointer;transition:.2s}
.rpc-cta-p{background:var(--accent);color:#0a0a12}.rpc-cta-p:hover{filter:brightness(1.08)}
.rpc-cta-s{background:var(--surface);border:1px solid var(--border);color:var(--muted)}.rpc-cta-s:hover{border-color:var(--accent);color:var(--accent)}

/* ── PRODUCT MODAL ── */
.modal-ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;
  justify-content:center;align-items:center;backdrop-filter:blur(5px)}
.modal-ov.show{display:flex}
.modal{background:var(--surface);border-radius:20px;max-width:470px;width:90%;max-height:88vh;overflow-y:auto;
  padding:26px;position:relative;border:1px solid var(--border)}
.modal .x{position:absolute;top:12px;right:12px;background:var(--surface2);border:none;color:var(--muted);
  width:30px;height:30px;border-radius:50%;font-size:.95rem;cursor:pointer}
.modal img{width:100%;max-height:270px;object-fit:contain;background:var(--surface2);border-radius:12px;padding:14px}
.modal h2{margin:12px 0 4px;font-family:'Playfair Display',serif;font-size:1.25rem;line-height:1.3}
.modal .mcat{font-size:.78rem;color:var(--muted)}
.modal .mprow{display:flex;align-items:center;gap:10px;margin:10px 0;flex-wrap:wrap}
.modal .mpr{font-size:1.4rem;font-weight:700;color:var(--accent);font-family:'Playfair Display',serif}
.modal .mmrp{font-size:.82rem;color:var(--muted);text-decoration:line-through}
.modal .msave{background:rgba(0,212,170,.12);color:var(--accent3);font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:10px}
.modal .macts{display:flex;gap:8px;margin-top:16px}
.modal .macts button{flex:1;padding:12px;border:none;border-radius:10px;font-weight:700;font-size:.85rem;cursor:pointer;transition:.2s}
.mw{background:rgba(232,93,122,.1);color:var(--accent2);border:1px solid rgba(232,93,122,.2)}.mw:hover{background:rgba(232,93,122,.22)}
.mc{background:rgba(245,197,24,.1);color:var(--accent);border:1px solid rgba(245,197,24,.2)}.mc:hover{background:rgba(245,197,24,.2)}
.mb{background:var(--accent);color:#0a0a12}.mb:hover{filter:brightness(1.08)}
.mp{background:rgba(124,92,252,.1);color:#7c5cfc;border:1px solid rgba(124,92,252,.2)}.mp:hover{background:rgba(124,92,252,.22)}

/* ── USER PANEL ── */
.up-ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:150}
.up-ov.show{display:block}
.up{position:fixed;top:0;right:-420px;width:390px;max-width:92vw;height:100%;background:var(--surface);
  z-index:160;transition:right .35s cubic-bezier(.16,1,.3,1);border-left:1px solid var(--border);display:flex;flex-direction:column}
.up.show{right:0}
.up-hdr{padding:18px 20px;background:var(--surface2);display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border)}
.up-hdr h2{font-family:'Playfair Display',serif;color:var(--accent);font-size:1.05rem}
.up-hdr button{background:none;border:none;color:var(--muted);font-size:1.4rem;cursor:pointer}
.up-tabs{display:flex;border-bottom:1px solid var(--border)}
.up-tab{flex:1;padding:10px;text-align:center;font-size:.76rem;font-weight:600;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;transition:.2s}
.up-tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.up-body{flex:1;overflow-y:auto;padding:13px}
.up-item{display:flex;gap:10px;padding:11px;background:var(--surface2);border-radius:10px;margin-bottom:9px;align-items:center;border:1px solid transparent;cursor:pointer;transition:.2s}
.up-item:hover{border-color:var(--border)}
.up-item img{width:60px;height:60px;object-fit:contain;border-radius:7px;background:var(--surface)}
.up-nm{font-size:.8rem;font-weight:500;color:var(--text);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.up-pr{font-size:.88rem;font-weight:700;color:var(--accent);margin-top:3px}
.up-ct{font-size:.65rem;color:var(--muted);margin-top:2px}
.up-empty{text-align:center;padding:46px 18px;color:var(--muted);font-size:.88rem}
.up-sum{padding:14px 20px;border-top:1px solid var(--border);background:var(--surface2)}
.up-tot{font-size:.95rem;font-weight:700;display:flex;justify-content:space-between}
.up-tot .amt{color:var(--accent);font-family:'Playfair Display',serif;font-size:1.15rem}

/* ── USER SWITCHER ── */
.sw-modal{background:var(--surface);border-radius:20px;max-width:490px;width:91%;max-height:78vh;overflow-y:auto;padding:24px;position:relative;border:1px solid var(--border)}
.user-row{display:flex;justify-content:space-between;align-items:center;padding:11px 13px;background:var(--surface2);border-radius:9px;margin-bottom:7px;border:1px solid transparent;transition:.2s}
.user-row:hover{border-color:var(--border)}
.user-row .uid{font-size:.78rem;font-weight:600;color:var(--text);font-family:monospace}
.user-row .ui{font-size:.68rem;color:var(--muted);margin-top:2px}
.user-row button{padding:5px 13px;background:var(--accent);border:none;border-radius:7px;color:#0a0a12;font-size:.7rem;font-weight:700;cursor:pointer}

/* ── TOAST ── */
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(70px);background:var(--surface2);
  color:var(--text);padding:11px 22px;border-radius:28px;font-size:.82rem;z-index:400;
  opacity:0;transition:all .3s;border:1px solid var(--border);box-shadow:0 6px 28px rgba(0,0,0,.45);white-space:nowrap}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* ── OVERLAY DEBUG PANEL ── */
#overlay-debug{position:fixed;bottom:80px;right:16px;width:380px;max-height:400px;
  background:var(--surface);border:1px solid var(--accent);border-radius:12px;
  padding:12px;font-size:.72rem;overflow-y:auto;z-index:500;display:none;
  box-shadow:0 10px 40px rgba(0,0,0,.5);font-family:monospace;}
#overlay-debug.show{display:block}
#overlay-debug h4{margin:0 0 10px;color:var(--accent);font-size:.8rem;display:flex;justify-content:space-between;align-items:center}
#overlay-debug h4 button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:1rem}
#overlay-debug .debug-entry{padding:8px;margin-bottom:6px;background:var(--surface2);border-radius:6px;
  border-left:3px solid var(--accent3)}
#overlay-debug .debug-entry.error{border-left-color:var(--accent2)}
#overlay-debug .debug-label{color:var(--accent);font-weight:700;margin-bottom:4px}
#overlay-debug .debug-value{color:var(--text);word-break:break-all}
#overlay-debug-trigger{position:fixed;bottom:26px;right:120px;background:var(--surface2);
  border:1px solid var(--border);border-radius:50%;width:36px;height:36px;
  display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:110;
  font-size:.8rem;color:var(--muted);transition:.2s}
#overlay-debug-trigger:hover{border-color:var(--accent);color:var(--accent)}
/* Visual debug markers on cards */
.rpc.debug-mode [data-overlay]{outline:2px dashed rgba(0,212,170,.5);outline-offset:-2px}
.rpc.debug-mode [data-overlay]::after{content:attr(data-position);position:absolute;
  bottom:-14px;left:50%;transform:translateX(-50%);font-size:.55rem;color:var(--accent3);
  background:rgba(0,0,0,.8);padding:1px 4px;border-radius:3px;white-space:nowrap}

/* ── LOADING ── */
.loading{text-align:center;padding:46px;color:var(--muted)}
.spinner{width:30px;height:30px;border:3px solid var(--surface2);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── AD POSTER MODAL ── */
#adPosterModal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:300;
  justify-content:center;align-items:center;backdrop-filter:blur(8px)}
#adPosterModal.show{display:flex}
.ad-poster-wrap{position:relative;max-width:95vw;max-height:95vh;}
.ad-poster-wrap .x{position:absolute;top:-12px;right:-12px;background:var(--accent);border:none;color:#0a0a12;
  width:36px;height:36px;border-radius:50%;font-size:1.1rem;cursor:pointer;font-weight:700;z-index:301;box-shadow:0 4px 20px rgba(0,0,0,.5)}
.ad-poster-wrap img{max-width:100%;max-height:90vh;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.6)}
.ad-poster-loading{color:var(--accent);font-size:1rem;text-align:center;padding:40px}

/* ── CATEGORY DROPDOWN ── */
.cat-dropdown-wrap{display:flex;gap:10px;align-items:center;margin-bottom:14px}
.cat-dropdown{padding:10px 18px;border:1px solid var(--border);border-radius:12px;
  background:var(--surface2);color:var(--text);font-size:.85rem;outline:none;cursor:pointer;min-width:200px}
.cat-dropdown:focus{border-color:var(--accent)}
.cat-dropdown option{background:var(--surface);color:var(--text)}
.cat-label{font-size:.78rem;color:var(--muted);font-weight:600}

/* ── INSIGHTS PANEL ── */
#insights-trigger{
  position:fixed;bottom:26px;left:26px;z-index:110;
  background:linear-gradient(135deg,#9b59b6 0%,#8e44ad 100%);
  color:#fff;border:none;border-radius:50px;padding:12px 18px;
  font-family:'DM Sans',sans-serif;font-weight:600;font-size:.82rem;
  cursor:pointer;display:none;box-shadow:0 6px 28px rgba(155,89,182,.4);
  align-items:center;gap:6px;transition:all .3s;
}
#insights-trigger:hover{transform:translateY(-3px);box-shadow:0 10px 36px rgba(155,89,182,.5)}
#insights-trigger .pulse-dot{background:#00d4aa}

#insights-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:115;backdrop-filter:blur(5px)}
#insights-backdrop.show{display:block}

#insights-panel{
  position:fixed;top:0;left:0;bottom:0;width:420px;max-width:90vw;z-index:120;
  background:var(--surface);border-right:1px solid var(--border);
  transform:translateX(-100%);transition:transform .4s cubic-bezier(.16,1,.3,1);
  display:flex;flex-direction:column;box-shadow:16px 0 60px rgba(0,0,0,.7);
  overflow:hidden;
}
#insights-panel.open{transform:translateX(0)}

.insights-hdr{padding:18px 22px;display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid var(--border);background:var(--surface2)}
.insights-title{font-family:'Playfair Display',serif;font-size:1.2rem;color:#9b59b6;display:flex;align-items:center;gap:8px}
.insights-close{background:var(--surface);border:none;color:var(--muted);width:32px;height:32px;
  border-radius:50%;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center}
.insights-close:hover{color:var(--text)}

.insights-body{flex:1;overflow-y:auto;padding:16px}
.insights-loading{text-align:center;padding:40px;color:var(--muted)}

.insight-stage{background:var(--surface2);border-radius:12px;padding:14px;margin-bottom:12px;border:1px solid var(--border)}
.stage-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.stage-icon{font-size:1.4rem}
.stage-name{font-weight:600;color:var(--text);font-size:.9rem}
.stage-details{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.stage-detail{font-size:.75rem}
.stage-detail .label{color:var(--muted)}
.stage-detail .value{color:var(--text);font-weight:500}

.intent-list{display:flex;flex-direction:column;gap:8px;margin-top:10px}
.intent-item{background:var(--surface);border-radius:8px;padding:10px 12px;border:1px solid var(--border)}
.intent-header{display:flex;justify-content:space-between;align-items:center}
.intent-id{font-weight:700;color:#9b59b6;font-size:.85rem}
.intent-weight{font-size:.75rem;color:var(--accent);font-weight:600}
.intent-bar{height:4px;background:var(--border);border-radius:2px;margin:6px 0;overflow:hidden}
.intent-bar-fill{height:100%;background:linear-gradient(90deg,#9b59b6,#e85d7a);border-radius:2px;transition:width .3s}
.intent-items{font-size:.7rem;color:var(--muted);margin-top:4px}

.scoring-list{margin-top:10px}
.scoring-item{display:flex;justify-content:space-between;align-items:center;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:.78rem}
.scoring-item:last-child{border-bottom:none}
.scoring-name{color:var(--text);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-right:8px}
.scoring-score{color:var(--accent);font-weight:600}
.scoring-badge{font-size:.65rem;padding:2px 6px;border-radius:4px;margin-left:6px}
.scoring-badge.history{background:rgba(232,93,122,.2);color:#e85d7a}

.final-recs{margin-top:16px}
.final-rec{background:var(--surface2);border-radius:10px;padding:12px;margin-bottom:8px;border:1px solid var(--border)}
.final-rec-name{font-weight:600;color:var(--text);font-size:.85rem;margin-bottom:4px}
.final-rec-reason{font-size:.72rem;color:var(--muted)}
.final-rec-meta{display:flex;justify-content:space-between;margin-top:6px;font-size:.7rem}
.final-rec-score{color:var(--accent);font-weight:600}
.final-rec-intent{color:#9b59b6}

/* ── PIPELINE TRACE ── */
.pipeline-trace{background:linear-gradient(135deg,rgba(155,89,182,.08),rgba(232,93,122,.05));border-color:rgba(155,89,182,.3)}
.pipeline-trace .stage-name{color:#9b59b6}
.trace-steps{display:flex;flex-direction:column;gap:10px;max-height:400px;overflow-y:auto}
.trace-step{background:var(--surface);border-radius:8px;padding:10px 12px;border:1px solid var(--border);font-size:.72rem}
.trace-step-header{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.trace-step-num{background:#9b59b6;color:#fff;padding:2px 8px;border-radius:10px;font-weight:700;font-size:.65rem}
.trace-step-name{font-weight:600;color:var(--text);flex:1}
.trace-step-shape{background:var(--surface2);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:.65rem;color:var(--accent)}
.trace-step-body{display:flex;flex-direction:column;gap:4px}
.trace-row{display:flex;gap:6px}
.trace-label{color:var(--muted);min-width:45px;font-weight:500}
.trace-val{color:var(--text);font-family:monospace;word-break:break-word}
.trace-op{color:#00d4aa}
.trace-out{color:var(--accent)}

/* ── BANNER AD (Single Poster Card) ── */
.banner-section{margin-bottom:28px;}
.banner-title{font-family:'Playfair Display',serif;font-size:1.05rem;color:var(--accent);margin-bottom:14px;display:flex;align-items:center;gap:10px;}
.banner-title::before{content:'';width:3px;height:18px;background:linear-gradient(180deg,var(--accent),var(--accent2));border-radius:2px;flex-shrink:0;}
.banner-grid{display:flex;justify-content:center;}
.banner-poster{position:relative;border-radius:18px;overflow:hidden;
  border:2px solid transparent;cursor:pointer;transition:all .35s cubic-bezier(.16,1,.3,1);
  box-shadow:0 8px 40px rgba(0,0,0,.5);width:100%;max-width:380px;aspect-ratio:3/4;}
.banner-poster:hover{transform:translateY(-5px);border-color:var(--accent);
  box-shadow:0 16px 60px rgba(245,197,24,.25);}
.banner-poster-img{width:100%;height:100%;object-fit:cover;display:block;
  background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);}
.banner-poster-loading{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);color:var(--muted);gap:12px;z-index:1;}
.banner-poster-loading .spinner{width:28px;height:28px;}
</style>
</head>
<body>

<nav class="navbar">
  <div class="brand" onclick="resetHome()">DisReq<small>Multi-Intent Recommendations</small></div>
  <div class="search-bar">
    <input type="text" id="searchInput" placeholder="Search products…" onkeydown="if(event.key==='Enter')searchProducts()">
    <button onclick="searchProducts()">Search</button>
  </div>
  <div class="nav-links">
    <a href="#" onclick="resetHome();return false">🏠</a>
    <a href="#" onclick="showUp('wishlist');return false">♥ <span class="badge" id="wishBadge">0</span></a>
    <a href="#" onclick="showUp('cart');return false">🛒 <span class="badge" id="cartBadge">0</span></a>
    <a href="#" onclick="showUp('purchases');return false">📦 <span class="badge" id="buyBadge">0</span></a>
    <a href="#" onclick="openSwitcher();return false" style="background:rgba(245,197,24,.07);border:1px solid var(--border)">👤 <span id="curUserLbl">Me</span></a>
    <a href="/admin">⚙</a>
  </div>
</nav>

<div class="container">
  <!-- Banner Ad Section -->
  <div class="banner-section" id="bannerSection">
    <div class="banner-title">🎯 Featured — Generated Poster</div>
    <div class="banner-grid" id="bannerGrid">
      <div class="banner-poster" id="skel-0"><div class="banner-poster-loading"><div class="spinner"></div><span>Generating poster…</span></div></div>
    </div>
  </div>

  <div class="stats-strip">
    <div class="si"><div class="v" id="sP">—</div><div class="l">Products</div></div>
    <div class="si"><div class="v" id="sU">—</div><div class="l">Users</div></div>
    <div class="si"><div class="v" id="sI">—</div><div class="l">Interactions</div></div>
    <div class="si"><div class="v" id="sM">—</div><div class="l">Model</div></div>
  </div>
  <div class="cat-dropdown-wrap">
    <span class="cat-label">Category:</span>
    <select class="cat-dropdown" id="catDropdown" onchange="onCatDropdownChange()">
      <option value="">All Categories</option>
    </select>
  </div>
  <div class="cats" id="catsBar"></div>
  <div class="sec-hdr">
    <h3 id="secLabel">✦ Featured Products</h3>
    <span id="secCount"></span>
  </div>
  <div class="product-grid" id="prodGrid"></div>
  <div class="loading" id="loadInd" style="display:none"><div class="spinner"></div><p>Loading…</p></div>
  <button class="load-more" id="loadMoreBtn" onclick="loadMore()">Load More</button>
</div>

<!-- REC TRIGGER -->
<button id="rec-trigger" onclick="openDrawer()">
  <span class="pulse-dot"></span> ✦ Picks For You
  <span id="recCountBadge" style="background:rgba(0,0,0,.2);padding:2px 8px;border-radius:8px;font-size:.68rem;margin-left:4px"></span>
</button>

<!-- INSIGHTS TRIGGER -->
<button id="insights-trigger" onclick="openInsights()">
  <span class="pulse-dot"></span> 🔬 Debug View
</button>

<!-- INSIGHTS BACKDROP -->
<div id="insights-backdrop" onclick="closeInsights()"></div>

<!-- INSIGHTS PANEL -->
<div id="insights-panel">
  <div class="insights-hdr">
    <div class="insights-title">🔬 Prediction Insights</div>
    <button class="insights-close" onclick="closeInsights()">✕</button>
  </div>
  <div class="insights-body" id="insightsBody">
    <div class="insights-loading"><div class="spinner"></div><p>Loading insights...</p></div>
  </div>
</div>

<!-- BACKDROP -->
<div id="rec-backdrop" onclick="closeDrawer()"></div>

<!-- REC DRAWER -->
<div id="rec-drawer">
  <div class="drawer-handle" onclick="closeDrawer()"></div>
  <div class="drawer-hdr">
    <div>
      <div class="drawer-title">✦ Curated For You</div>
      <div class="drawer-sub" id="drawerSub">Browse products to unlock your personalised picks</div>
    </div>
    <button class="drawer-close" onclick="closeDrawer()">✕</button>
  </div>
  <div class="rec-scroll" id="recScroll">
    <div class="loading"><div class="spinner"></div><p>Finding your picks…</p></div>
  </div>
</div>

<!-- PRODUCT MODAL -->
<div class="modal-ov" id="prodModal">
  <div class="modal">
    <button class="x" onclick="closeModal()">✕</button>
    <img id="mImg" src="" alt="" onerror="this.src='https://via.placeholder.com/300x300?text=No+Image'">
    <h2 id="mName"></h2>
    <div class="mcat" id="mCat"></div>
    <div class="mprow">
      <span class="mpr" id="mPrice"></span>
      <span class="mmrp" id="mMrp"></span>
      <span class="msave" id="mSave" style="display:none"></span>
    </div>
    <div class="macts">
      <button class="mw" onclick="doAct('wishlist')">♥ Wishlist</button>
      <button class="mc" onclick="doAct('cart')">🛒 Add Cart</button>
      <button class="mb" onclick="doAct('buy')">⚡ Buy Now</button>
    </div>
    <div class="macts" style="margin-top:8px">
      <button class="mp" onclick="viewCurProdPoster()">🎨 View Poster</button>
    </div>
  </div>
</div>

<!-- USER PANEL -->
<div class="up-ov" id="upOv" onclick="closeUp()"></div>
<div class="up" id="upPanel">
  <div class="up-hdr"><h2 id="upTitle">My Cart</h2><button onclick="closeUp()">✕</button></div>
  <div class="up-tabs">
    <div class="up-tab active" onclick="switchTab('cart',this)">🛒 Cart</div>
    <div class="up-tab" onclick="switchTab('wishlist',this)">♥ Wishlist</div>
    <div class="up-tab" onclick="switchTab('purchases',this)">📦 Orders</div>
  </div>
  <div class="up-body" id="upBody"></div>
  <div class="up-sum" id="upSum" style="display:none">
    <div class="up-tot"><span>Total</span><span class="amt" id="upTotal">₹0</span></div>
  </div>
</div>

<!-- USER SWITCHER -->
<div class="modal-ov" id="swModal">
  <div class="sw-modal">
    <button class="x" onclick="closeSwitcher()">✕</button>
    <h2 style="font-family:'Playfair Display',serif;color:var(--accent);margin-bottom:8px">Switch User</h2>
    <p style="font-size:.78rem;color:var(--muted);margin-bottom:14px">See personalised picks for different shoppers.</p>
    <button onclick="createUser()" style="width:100%;padding:10px;background:var(--accent);border:none;border-radius:10px;color:#0a0a12;font-weight:700;cursor:pointer;font-size:.8rem;margin-bottom:14px">+ Create New User</button>
    <div id="usrList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<!-- AD POSTER MODAL -->
<div id="adPosterModal" onclick="if(event.target===this)closeAdPoster()">
  <div class="ad-poster-wrap">
    <button class="x" onclick="closeAdPoster()">✕</button>
    <div id="adPosterContent"><div class="ad-poster-loading"><div class="spinner"></div>Generating poster...</div></div>
  </div>
</div>

<!-- OVERLAY DEBUG PANEL -->
<button id="overlay-debug-trigger" onclick="toggleDebugPanel()" title="Debug Overlay Placement">🔍</button>
<div id="overlay-debug">
  <h4>Overlay Placement Debug <button onclick="toggleDebugPanel()">✕</button></h4>
  <div id="debug-entries">
    <div class="debug-entry">
      <div class="debug-label">System Status</div>
      <div class="debug-value">Image-dependent placement active. Each product card analyzes its image to determine optimal overlay positions.</div>
    </div>
  </div>
</div>

<script>
// ── Debug Panel State ──────────────────────────
let debugMode = false;
const debugLog = [];

function toggleDebugPanel() {
  const panel = document.getElementById('overlay-debug');
  panel.classList.toggle('show');
  debugMode = panel.classList.contains('show');
  
  // Toggle debug mode on all cards
  document.querySelectorAll('.rpc').forEach(card => {
    card.classList.toggle('debug-mode', debugMode);
  });
  
  if (debugMode) {
    renderDebugLog();
  }
}

function logPlacement(idx, imageUrl, placement) {
  debugLog.unshift({
    timestamp: new Date().toLocaleTimeString(),
    cardIdx: idx,
    imageUrl: imageUrl?.substring(0, 50) + '...',
    placement: placement
  });
  if (debugLog.length > 20) debugLog.pop();
  if (debugMode) renderDebugLog();
}

function renderDebugLog() {
  const container = document.getElementById('debug-entries');
  if (!container) return;
  
  let html = `<div class="debug-entry">
    <div class="debug-label">🎯 Placement Algorithm</div>
    <div class="debug-value">
      <strong>Steps:</strong><br>
      1. PRODUCT_REGION: Detect where product is located<br>
      2. VISUAL_DENSITY: Map busy vs calm regions<br>
      3. SAFE_ZONES: Identify non-blocking positions<br>
      4. PLACEMENT_PLAN: Assign overlays to safe zones
    </div>
  </div>`;
  
  debugLog.forEach(entry => {
    const p = entry.placement || {};
    const debug = p.debug || {};
    const prod = debug.product_region || {};
    
    html += `<div class="debug-entry">
      <div class="debug-label">Card #${entry.cardIdx + 1} @ ${entry.timestamp}</div>
      <div class="debug-value">
        <strong>Orientation:</strong> ${prod.orientation || 'unknown'}<br>
        <strong>Product Center:</strong> (${(prod.center?.[0]*100||0).toFixed(0)}%, ${(prod.center?.[1]*100||0).toFixed(0)}%)<br>
        <strong>Coverage:</strong> ${((prod.coverage||0)*100).toFixed(0)}%<br>
        <strong>Safest Zones:</strong> ${(debug.safest_zones || []).slice(0,3).join(', ')}<br>
        <strong>Rank Position:</strong> ${p.rank_badge?.position || 'default'}<br>
        <strong>Discount Position:</strong> ${p.discount_badge?.position || 'default'}
      </div>
    </div>`;
  });
  
  container.innerHTML = html;
}

// ── State ──────────────────────────────────────
let prods=[],offset=0,LIMIT=24;
let hist=[],cartC=0,wishC=0,buyC=0;
let curProd=null,curCat=null,curSrch=null,total=0; 
let curTab='cart',curUser=null,curRecs=[];

let interactCount=0; // Throttle recommendations to every 8 interactions

// ── Category catchy copy (notebook concept integrated) ──────────
const CAT_COPY={
  'Electronics':{tag:'⚡ Power up your world',ctas:['Grab the Tech','Get It Now','Upgrade']},
  'Mobile':{tag:'📱 Stay ahead of the curve',ctas:['Shop Mobiles','Explore Deals','Get Yours']},
  'Laptop':{tag:'💻 Work smarter, not harder',ctas:['Shop Laptops','Upgrade Now','Get Yours']},
  'Fashion':{tag:'✨ Style that speaks',ctas:['Wear It','Shop Style','Grab Yours']},
  'Clothing':{tag:'👗 Dress to impress',ctas:['Shop Look','Try It On','Add to Bag']},
  'Shoes':{tag:'👟 Step into greatness',ctas:['Shop Shoes','Walk In Style','Get Pair']},
  'Home':{tag:'🏡 Transform your space',ctas:['Shop Home','Elevate It','Buy Now']},
  'Furniture':{tag:'🛋 Living, reimagined',ctas:['Style It','Shop Now','Add to Room']},
  'Kitchen':{tag:'🍳 Cook with confidence',ctas:['Shop Kitchen','Cook Better','Buy Now']},
  'Books':{tag:'📚 Knowledge is power',ctas:['Start Reading','Buy Book','Add to List']},
  'Sports':{tag:'🏆 Level up your game',ctas:['Shop Sports','Get Active','Gear Up']},
  'Fitness':{tag:'💪 Crush your goals',ctas:['Get Fit','Shop Now','Grab It']},
  'Toys':{tag:'🎮 Joy delivered fast',ctas:['Shop Toys','Gift It','Order Now']},
  'Beauty':{tag:'💄 Glow up, inside out',ctas:['Shop Beauty','Try It','Get Yours']},
  'Skincare':{tag:'✨ Your skin deserves this',ctas:['Shop Skin','Try It','Get Yours']},
  'Grocery':{tag:'🛒 Fresh, fast, delivered',ctas:['Add to Cart','Shop Now','Order']},
  'default':{tag:'🔥 Hot pick for you',ctas:['Shop Now','Get It','Buy Now']}
};
function getCopy(cat){
  if(!cat)return CAT_COPY.default;
  const k=Object.keys(CAT_COPY).find(k=>cat.toLowerCase().includes(k.toLowerCase()));
  return k?CAT_COPY[k]:CAT_COPY.default;
}
function rnd(arr){return arr[Math.floor(Math.random()*arr.length)];}
function badge(disc,score){
  if(disc>=30)return{cls:'rb-hot',txt:'🔥 Hot Deal'};
  if(disc>=15)return{cls:'rb-deal',txt:'💚 Great Value'};
  if(score>80)return{cls:'rb-trend',txt:'⭐ Trending'};
  return{cls:'rb-rec',txt:'✦ For You'};
}

// ── Helpers ────────────────────────────────────
window.onload=()=>{
  loadUser();loadFeatured();loadCats();loadStats();refreshBadges();loadInitialBanners();
  setInterval(loadStats,15000);setInterval(refreshBadges,10000);
};
async function api(u,o){
  try{
    const r=await fetch(u,o);
    if(!r.ok)throw new Error('HTTP '+r.status);
    return await r.json();
  }catch(e){
    console.error('API error:',u,e);
    return null;
  }
}
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500);}
function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function fp(p){return p>0?'₹'+Number(p).toLocaleString('en-IN'):'N/A';}

// ── Data ───────────────────────────────────────
async function loadFeatured(){
  document.getElementById('loadInd').style.display='block';
  try{
    const d=await api('/api/products/featured?offset='+offset+'&limit='+LIMIT);
    document.getElementById('loadInd').style.display='none';
    if(!d||!Array.isArray(d)){
      document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted)">Unable to load products</div>';
      return;
    }
    prods=prods.concat(d);renderProds(d,true);
    document.getElementById('secLabel').textContent='✦ Featured Products';
    if(d.length<LIMIT)document.getElementById('loadMoreBtn').style.display='none';
  }catch(e){
    document.getElementById('loadInd').style.display='none';
    document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--accent2)">Failed to load products</div>';
  }
}
async function loadMore(){
  offset+=LIMIT;
  if(curCat)await loadCatPage();else if(curSrch)await loadSrchPage();else await loadFeatured();
}
async function searchProducts(){
  const q=document.getElementById('searchInput').value.trim();if(!q)return;
  curCat=null;curSrch=q;offset=0;prods=[];document.getElementById('prodGrid').innerHTML='';
  await loadSrchPage();
}
async function loadSrchPage(){
  document.getElementById('loadInd').style.display='block';
  try{
    const r=await api('/api/search?q='+encodeURIComponent(curSrch)+'&offset='+offset+'&limit='+LIMIT);
    document.getElementById('loadInd').style.display='none';
    if(!r){
      document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted)">Search failed. Please try again.</div>';
      return;
    }
    const d=r.items||r||[];total=r.total||d.length;prods=prods.concat(d);renderProds(d,true);
    document.getElementById('secLabel').textContent='🔍 "'+curSrch+'"';
    document.getElementById('secCount').textContent=total+' found';
    document.getElementById('loadMoreBtn').style.display=prods.length<total?'block':'none';
    if(d.length===0){
      document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted)">No products found for "'+esc(curSrch)+'"</div>';
    }
  }catch(e){
    document.getElementById('loadInd').style.display='none';
    document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--accent2)">Search error</div>';
  }
}
async function loadCats(){
  try{
    const cs=await api('/api/categories');
    const bar=document.getElementById('catsBar');
    const dropdown=document.getElementById('catDropdown');
    bar.innerHTML='<div class="cat-chip active" onclick="resetHome()">All</div>';
    dropdown.innerHTML='<option value="">All Categories</option>';
    if(cs&&Array.isArray(cs)){
      cs.forEach(c=>{
        if(!c.name)return;
        const ch=document.createElement('div');ch.className='cat-chip';ch.textContent=c.name+' ('+c.count+')';ch.onclick=()=>filterCat(c.name,ch);bar.appendChild(ch);
        const opt=document.createElement('option');opt.value=c.name;opt.textContent=c.name+' ('+c.count+')';dropdown.appendChild(opt);
      });
    }
  }catch(e){
    console.error('Failed to load categories:',e);
  }
}
function onCatDropdownChange(){
  const val=document.getElementById('catDropdown').value;
  if(!val){resetHome();return;}
  // Find and highlight matching chip
  document.querySelectorAll('.cat-chip').forEach(c=>{
    c.classList.remove('active');
    if(c.textContent.startsWith(val+' ') || c.textContent===val){
      c.classList.add('active');
    }
  });
  curCat=val;curSrch=null;offset=0;prods=[];
  document.getElementById('prodGrid').innerHTML='';
  loadCatPage();
}
async function filterCat(cat,el){
  curCat=cat;curSrch=null;offset=0;prods=[];
  document.querySelectorAll('.cat-chip').forEach(c=>c.classList.remove('active'));
  if(el)el.classList.add('active');
  // Sync dropdown with chip selection
  document.getElementById('catDropdown').value=cat;
  document.getElementById('prodGrid').innerHTML='';await loadCatPage();
}
async function loadCatPage(){
  document.getElementById('loadInd').style.display='block';
  try{
    const r=await api('/api/products/category?cat='+encodeURIComponent(curCat)+'&offset='+offset+'&limit='+LIMIT);
    document.getElementById('loadInd').style.display='none';
    const d=r.items||r||[];
    total=r.total||d.length;prods=prods.concat(d);renderProds(d,true);
    document.getElementById('secLabel').textContent=curCat;
    document.getElementById('secCount').textContent=total+' products';
    document.getElementById('loadMoreBtn').style.display=prods.length<total?'block':'none';
    if(d.length===0){
      document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted)">No products found in this category</div>';
    }
  }catch(e){
    document.getElementById('loadInd').style.display='none';
    document.getElementById('prodGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--accent2)">Failed to load products. Please try again.</div>';
  }
}
function resetHome(){
  curCat=null;curSrch=null;prods=[];offset=0;
  document.querySelectorAll('.cat-chip').forEach(c=>c.classList.remove('active'));
  const f=document.querySelector('.cat-chip');if(f)f.classList.add('active');
  document.getElementById('catDropdown').value='';
  document.getElementById('loadMoreBtn').style.display='block';
  document.getElementById('prodGrid').innerHTML='';document.getElementById('secCount').textContent='';
  loadFeatured();
}
async function loadStats(){
  try{const s=await api('/api/stats');
    document.getElementById('sP').textContent=s.num_products;
    document.getElementById('sU').textContent=s.num_users;
    document.getElementById('sI').textContent=s.num_interactions;
    document.getElementById('sM').textContent=s.model_status;}catch(e){}
}
async function refreshBadges(){
  try{const[c,w,b]=await Promise.all([api('/api/user/cart'),api('/api/user/wishlist'),api('/api/user/purchases')]);
    cartC=c.length;wishC=w.length;buyC=b.length;
    document.getElementById('cartBadge').textContent=cartC;
    document.getElementById('wishBadge').textContent=wishC;
    document.getElementById('buyBadge').textContent=buyC;}catch(e){}
}

// ── Product rendering ──────────────────────────
function renderProds(data,append){
  const g=document.getElementById('prodGrid');if(!append)g.innerHTML='';
  data.forEach(p=>{
    const disc=p.discount_pct||0;
    const c=document.createElement('div');c.className='product-card';
    c.innerHTML=`<img src="${esc(p.primary_image)}" alt="${esc(p.Product_Name)}" loading="lazy" onerror="this.src='https://via.placeholder.com/300x300?text=No+Image'">
<div class="card-body">
  <div class="card-name">${esc(p.Product_Name)}</div>
  <div class="card-cat">${esc(p.combined_category)}</div>
  <div class="price-row">
    <span class="card-price">${fp(p.price)}</span>
    ${p.mrp&&p.mrp>p.price?`<span class="card-mrp">${fp(p.mrp)}</span><span class="card-disc">${disc}% OFF</span>`:''}
  </div>
  <div class="card-actions">
    <button class="bw" data-pid="${p.product_id}" data-a="wishlist">♥</button>
    <button class="bc" data-pid="${p.product_id}" data-a="cart">🛒</button>
    <button class="bb" data-pid="${p.product_id}" data-a="buy">⚡</button>
  </div>
</div>`;
    c.querySelectorAll('.card-actions button').forEach(b=>{b.onclick=e=>{e.stopPropagation();quickAct(+b.dataset.pid,b.dataset.a);};});
    c.onclick=()=>openProd(p);g.appendChild(c);
  });
}

// ── Modal ──────────────────────────────────────
function openProd(p){
  curProd=p;
  document.getElementById('mImg').src=p.primary_image||'';
  document.getElementById('mName').textContent=p.Product_Name;
  document.getElementById('mCat').textContent=p.combined_category;
  document.getElementById('mPrice').textContent=fp(p.price);
  const disc=p.discount_pct||0;
  if(p.mrp&&p.mrp>p.price){document.getElementById('mMrp').textContent=fp(p.mrp);document.getElementById('mSave').textContent='Save '+disc+'%';document.getElementById('mSave').style.display='';}
  else{document.getElementById('mMrp').textContent='';document.getElementById('mSave').style.display='none';}
  document.getElementById('prodModal').classList.add('show');
  trackView(p.product_id);
}
function closeModal(){document.getElementById('prodModal').classList.remove('show');curProd=null;}
function viewCurProdPoster(){if(curProd){closeModal();showAdPoster(curProd.product_id);}}

async function trackView(pid){
  hist.push({pid,action:'view',ts:Date.now()});
  await api('/api/track_view',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product_id:pid})});
  interactCount++;
  if(interactCount%8===0)fetchRecs();
}
async function quickAct(pid,action){
  hist.push({pid,action,ts:Date.now()});
  await api('/api/track_'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product_id:pid})});
  const msgs={cart:'Added to cart ✓',wishlist:'Saved ♥',buy:'Purchase recorded ⚡'};
  if(action==='cart'){cartC++;document.getElementById('cartBadge').textContent=cartC;}
  else if(action==='wishlist'){wishC++;document.getElementById('wishBadge').textContent=wishC;}
  else if(action==='buy'){buyC++;document.getElementById('buyBadge').textContent=buyC;refreshBadges();}
  toast(msgs[action]||'Done ✓');
  interactCount++;
  if(interactCount%8===0)fetchRecs();
}
function doAct(a){if(!curProd)return;quickAct(curProd.product_id,a);if(a!=='wishlist')closeModal();}

// ── Rec Drawer ─────────────────────────────────
function openDrawer(){
  document.getElementById('rec-drawer').classList.add('open');
  document.getElementById('rec-backdrop').classList.add('show');
  document.body.style.overflow='hidden';
  if(curRecs.length>0)renderRecs(curRecs);
}
function closeDrawer(){
  document.getElementById('rec-drawer').classList.remove('open');
  document.getElementById('rec-backdrop').classList.remove('show');
  document.body.style.overflow='';
}

// ── Insights Panel ─────────────────────────────
function openInsights(){
  document.getElementById('insights-panel').classList.add('open');
  document.getElementById('insights-backdrop').classList.add('show');
  document.body.style.overflow='hidden';
  fetchInsights();
}
function closeInsights(){
  document.getElementById('insights-panel').classList.remove('open');
  document.getElementById('insights-backdrop').classList.remove('show');
  document.body.style.overflow='';
}
async function fetchInsights(){
  const body=document.getElementById('insightsBody');
  body.innerHTML='<div class="insights-loading"><div class="spinner"></div><p>Analyzing predictions...</p></div>';
  try{
    const data=await api('/api/prediction_debug',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history:hist})});
    if(!data){body.innerHTML='<div class="insights-loading">No data available</div>';return;}
    renderInsights(data);
  }catch(e){
    body.innerHTML='<div class="insights-loading">Error loading insights</div>';
  }
}
function renderInsights(data){
  const body=document.getElementById('insightsBody');
  let html='';
  
  // Stages
  if(data.stages&&data.stages.length>0){
    data.stages.forEach(stage=>{
      html+=`<div class="insight-stage">
        <div class="stage-header">
          <span class="stage-icon">${stage.icon||'📌'}</span>
          <span class="stage-name">${esc(stage.name)}</span>
        </div>
        <div class="stage-details">`;
      if(stage.details){
        stage.details.forEach(d=>{
          html+=`<div class="stage-detail"><span class="label">${esc(d.label)}:</span> <span class="value">${esc(String(d.value))}</span></div>`;
        });
      }
      html+=`</div>`;
      // Show items if present
      if(stage.items&&stage.items.length>0){
        html+=`<div style="margin-top:10px;font-size:.72rem;color:var(--muted)"><strong>Recent History:</strong></div>`;
        html+=`<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">`;
        stage.items.forEach(item=>{
          const color=item.action==='buy'?'#00d4aa':item.action==='cart'?'#e85d7a':item.action==='wishlist'?'#9b59b6':'#7a7a94';
          html+=`<span style="font-size:.65rem;padding:2px 6px;background:${color}22;color:${color};border-radius:4px">${esc(item.name.slice(0,20))}${item.weight>1?' ('+item.weight+'x)':''}</span>`;
        });
        html+=`</div>`;
      }
      // Show recency weights if present
      if(stage.weights&&stage.weights.length>0){
        html+=`<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">`;
        stage.weights.forEach(w=>{
          html+=`<div style="text-align:center"><div style="font-size:.65rem;color:var(--muted)">${w.pos}</div><div style="font-size:.75rem;font-weight:600;color:var(--accent)">${w.weight}%</div></div>`;
        });
        html+=`</div>`;
      }
      html+=`</div>`;
    });
  }
  
  // Intents
  if(data.intents&&data.intents.length>0){
    html+=`<div class="insight-stage">
      <div class="stage-header">
        <span class="stage-icon">🎯</span>
        <span class="stage-name">Intent Distribution</span>
      </div>
      <div class="intent-list">`;
    data.intents.forEach(intent=>{
      html+=`<div class="intent-item">
        <div class="intent-header">
          <span class="intent-id">Intent ${intent.intent_id}</span>
          <span class="intent-weight">${intent.avg_weight}% active</span>
        </div>
        <div class="intent-bar"><div class="intent-bar-fill" style="width:${Math.min(100,intent.avg_weight*2)}%"></div></div>
        ${intent.top_items&&intent.top_items.length>0?`<div class="intent-items">Top: ${intent.top_items.map(i=>esc(i)).join(', ')}</div>`:''}
      </div>`;
    });
    html+=`</div></div>`;
  }
  
  // Scoring breakdown
  if(data.scoring_breakdown&&data.scoring_breakdown.length>0){
    html+=`<div class="insight-stage">
      <div class="stage-header">
        <span class="stage-icon">📊</span>
        <span class="stage-name">Top Scoring Items</span>
      </div>
      <div class="scoring-list">`;
    data.scoring_breakdown.forEach(item=>{
      html+=`<div class="scoring-item">
        <span class="scoring-name">${esc(item.product)}</span>
        <span class="scoring-score">${item.score}</span>
        ${item.in_history?'<span class="scoring-badge history">Seen</span>':''}
      </div>`;
    });
    html+=`</div></div>`;
  }
  
  // Pipeline Trace (detailed step-by-step)
  if(data.pipeline_trace&&data.pipeline_trace.length>0){
    html+=`<div class="insight-stage pipeline-trace">
      <div class="stage-header">
        <span class="stage-icon">🔬</span>
        <span class="stage-name">Pipeline Trace (Step-by-Step)</span>
      </div>
      <div class="trace-steps">`;
    data.pipeline_trace.forEach(step=>{
      html+=`<div class="trace-step">
        <div class="trace-step-header">
          <span class="trace-step-num">Step ${step.step}</span>
          <span class="trace-step-name">${esc(step.name)}</span>
          <span class="trace-step-shape">${esc(step.shape)}</span>
        </div>
        <div class="trace-step-body">
          <div class="trace-row"><span class="trace-label">Input:</span> <span class="trace-val">${esc(step.input)}</span></div>
          <div class="trace-row"><span class="trace-label">Op:</span> <span class="trace-val trace-op">${esc(step.operation)}</span></div>
          <div class="trace-row"><span class="trace-label">Output:</span> <span class="trace-val trace-out">${esc(step.output)}</span></div>
        </div>
      </div>`;
    });
    html+=`</div></div>`;
  }
  
  // Final recommendations
  if(data.final_recommendations&&data.final_recommendations.length>0){
    html+=`<div class="final-recs">
      <div style="font-weight:600;margin-bottom:10px;color:var(--accent)">✦ Final Recommendations</div>`;
    data.final_recommendations.forEach(rec=>{
      html+=`<div class="final-rec">
        <div class="final-rec-name">${esc(rec.name)}</div>
        <div class="final-rec-reason">${esc(rec.reason)}</div>
        <div class="final-rec-meta">
          <span class="final-rec-score">Score: ${rec.score}</span>
          ${rec.intent?`<span class="final-rec-intent">🧠 ${esc(rec.intent)}</span>`:''}
        </div>
      </div>`;
    });
    html+=`</div>`;
  }
  
  if(!html)html='<div class="insights-loading">No prediction data available.<br>Browse some products first!</div>';
  body.innerHTML=html;
}

async function fetchRecs(){
  if(hist.length<1){
    document.getElementById('rec-trigger').style.display='none';
    document.getElementById('insights-trigger').style.display='none';
    return;
  }
  // Show insights button when history exists
  document.getElementById('insights-trigger').style.display='flex';
  try{
    const recs=await api('/api/recommendations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history:hist,llm_rerank:hist.length>=3})});
    if(recs&&Array.isArray(recs)&&recs.length>0){
      curRecs=recs;
      const btn=document.getElementById('rec-trigger');
      btn.style.display='flex';
      document.getElementById('recCountBadge').textContent=recs.length;
      document.getElementById('drawerSub').textContent=recs.length+' hand-picked items, just for you';
      if(document.getElementById('rec-drawer').classList.contains('open'))renderRecs(recs);
      // Render banner ads with top 2-3 picks
      renderBannerAds(recs);
    }else{
      // No personalized recs yet - leave initial banners as-is
      document.getElementById('rec-trigger').style.display='none';
    }
  }catch(e){
    console.error('Fetch recs error:',e);
    // Don't hide trigger on error - show cached recs if any
    if(curRecs.length>0){
      if(document.getElementById('rec-drawer').classList.contains('open'))renderRecs(curRecs);
      renderBannerAds(curRecs);
    }
  }
}

// Renders recommendation popup cards with catchy copy and DYNAMIC OVERLAY PLACEMENT
function renderRecs(recs){
  const area=document.getElementById('recScroll');area.innerHTML='';
  recs.forEach((r,idx)=>{
    const p=r.product||r;
    const disc=p.discount_pct||0;
    const copy=getCopy(p.combined_category);
    const ctaTxt=rnd(copy.ctas);
    const bdg=badge(disc,r.score||50);
    const exId='rex-'+idx;
    const cardId='rpc-'+idx;

    const card=document.createElement('div');card.className='rpc';card.id=cardId;
    
    // Create card showing generated poster as the banner ad image directly
    card.innerHTML=`
<div class="rpc-overlays" id="overlays-${idx}">
  <span class="rpc-badge rpc-rank ${bdg.cls}" data-overlay="rank" style="position:absolute;top:8px;left:8px;opacity:1;z-index:2">#${idx+1}</span>
  ${disc>0?`<span class="rpc-badge rpc-discount rb-deal" data-overlay="discount" style="position:absolute;top:8px;right:8px;opacity:1;z-index:2">${disc}% OFF</span>`:''}
</div>
<div class="rpc-poster-loading" id="rpc-load-${idx}" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--surface2);z-index:1">
  <div class="spinner" style="width:22px;height:22px"></div>
  <span style="font-size:.65rem;color:var(--muted);margin-top:6px">Generating ad…</span>
</div>
<img class="rpc-img" id="rpc-poster-${idx}" src="/api/product_ad?product_id=${p.product_id}&t=${Date.now()}" alt="${esc(p.Product_Name)}" loading="lazy" style="display:none" data-image-url="${esc(p.primary_image)}">
<div class="rpc-body">
  <div class="rpc-tagline">${esc(copy.tag)}</div>
  <div class="rpc-name">${esc(p.Product_Name)}</div>
  <div class="rpc-price-row">
    <span class="rpc-price">${fp(p.price)}</span>
    ${p.mrp&&p.mrp>p.price?`<span class="rpc-mrp">${fp(p.mrp)}</span><span class="rpc-save">Save ${disc}%</span>`:''}
  </div>
  ${r.intent_label?`<div class="rpc-intent">🧠 ${esc(r.intent_label)} match</div>`:''}
  <div class="rpc-explain" id="${exId}"></div>
  <div class="rpc-ctas">
    <button class="rpc-cta rpc-cta-p" id="rp-add-${idx}">${ctaTxt}</button>
    <button class="rpc-cta rpc-cta-s" id="rp-why-${idx}" title="Why recommended?">💡</button>
  </div>
</div>`;
    // Wire poster load/error: show generated poster as the card banner ad
    const posterImg = card.querySelector('#rpc-poster-'+idx);
    const posterLoader = card.querySelector('#rpc-load-'+idx);
    posterImg.onload = function(){ posterLoader.style.display='none'; posterImg.style.display='block'; };
    posterImg.onerror = function(){ posterLoader.style.display='none'; posterImg.src=p.primary_image||'https://via.placeholder.com/300x300?text=No+Image'; posterImg.style.display='block'; };
    card.querySelector('.rpc-cta-p').onclick=e=>{e.stopPropagation();quickAct(p.product_id,'cart');toast('Added to cart ✓');};
    card.querySelector('#rp-why-'+idx).onclick=e=>{e.stopPropagation();toggleExplain(idx,r,p);};
    card.onclick=()=>openProd(p);
    area.appendChild(card);
  });
}

// Image-dependent overlay placement system
const placementCache = new Map();

async function analyzeAndPlaceOverlays(idx, imageUrl, discountPct) {
  if (!imageUrl) {
    applyDefaultPlacement(idx, discountPct);
    logPlacement(idx, null, {debug: {image_analyzed: false, reason: 'No image URL'}});
    return;
  }
  
  // Check cache
  const cacheKey = imageUrl;
  if (placementCache.has(cacheKey)) {
    const cached = placementCache.get(cacheKey);
    applyPlacement(idx, cached, discountPct);
    logPlacement(idx, imageUrl, {...cached, debug: {...cached.debug, from_cache: true}});
    return;
  }
  
  try {
    const resp = await fetch(`/api/analyze_image_placement?image_url=${encodeURIComponent(imageUrl)}&card_width=222&card_height=158`);
    if (!resp.ok) throw new Error('API error');
    const placement = await resp.json();
    
    // Cache the result
    placementCache.set(cacheKey, placement);
    
    // Apply dynamic placement
    applyPlacement(idx, placement, discountPct);
    
    // Log to debug panel
    logPlacement(idx, imageUrl, placement);
    
    // Log debug info for verification
    if (placement.debug) {
      console.log(`[Overlay ${idx}] Image Analysis:`, {
        orientation: placement.debug.product_region?.orientation,
        safestZones: placement.debug.safest_zones,
        rankPos: placement.rank_badge?.position,
        discountPos: placement.discount_badge?.position
      });
    }
  } catch (e) {
    console.warn('Placement analysis failed, using fallback for card', idx, e);
    applyDefaultPlacement(idx, discountPct);
    logPlacement(idx, imageUrl, {debug: {image_analyzed: false, error: e.message}});
  }
}

function applyPlacement(idx, placement, discountPct) {
  const overlays = document.getElementById(`overlays-${idx}`);
  if (!overlays) return;
  
  // Apply rank badge position
  const rankBadge = overlays.querySelector('[data-overlay="rank"]');
  if (rankBadge && placement.rank_badge) {
    Object.assign(rankBadge.style, {
      top: placement.rank_badge.top,
      right: placement.rank_badge.right,
      bottom: placement.rank_badge.bottom,
      left: placement.rank_badge.left,
      transform: placement.rank_badge.transform || 'none',
      opacity: '1'
    });
    rankBadge.dataset.position = placement.rank_badge.position;
  }
  
  // Apply discount badge position
  const discBadge = overlays.querySelector('[data-overlay="discount"]');
  if (discBadge && placement.discount_badge && discountPct > 0) {
    Object.assign(discBadge.style, {
      top: placement.discount_badge.top,
      right: placement.discount_badge.right,
      bottom: placement.discount_badge.bottom,
      left: placement.discount_badge.left,
      transform: placement.discount_badge.transform || 'none',
      opacity: '1'
    });
    discBadge.dataset.position = placement.discount_badge.position;
  }
  
  // Add layout type class to card for additional styling
  const card = document.getElementById(`rpc-${idx}`);
  if (card && placement.layout_type) {
    card.classList.add(`layout-${placement.layout_type}`);
  }
}

function applyDefaultPlacement(idx, discountPct) {
  // Fallback: use simple algorithm without API
  const overlays = document.getElementById(`overlays-${idx}`);
  if (!overlays) return;
  
  // Vary placement based on card index to avoid identical layouts
  const patterns = [
    { rank: {top:'8px',left:'8px'}, discount: {top:'8px',right:'8px'} },
    { rank: {top:'8px',right:'8px'}, discount: {bottom:'8px',left:'8px'} },
    { rank: {bottom:'8px',left:'8px'}, discount: {top:'8px',right:'8px'} },
    { rank: {top:'8px',left:'8px'}, discount: {bottom:'8px',right:'8px'} },
  ];
  const pattern = patterns[idx % patterns.length];
  
  const rankBadge = overlays.querySelector('[data-overlay="rank"]');
  if (rankBadge) {
    Object.assign(rankBadge.style, {
      top: pattern.rank.top || 'auto',
      right: pattern.rank.right || 'auto',
      bottom: pattern.rank.bottom || 'auto',
      left: pattern.rank.left || 'auto',
      opacity: '1'
    });
  }
  
  const discBadge = overlays.querySelector('[data-overlay="discount"]');
  if (discBadge && discountPct > 0) {
    Object.assign(discBadge.style, {
      top: pattern.discount.top || 'auto',
      right: pattern.discount.right || 'auto',
      bottom: pattern.discount.bottom || 'auto',
      left: pattern.discount.left || 'auto',
      opacity: '1'
    });
  }
}

async function toggleExplain(idx,rec,prod){
  const box=document.getElementById('rex-'+idx);
  const btn=document.getElementById('rp-why-'+idx);
  if(box.classList.contains('show')){box.classList.remove('show');btn.textContent='💡';return;}
  btn.textContent='⏳';btn.disabled=true;
  try{
    const resp=await api('/api/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      product_name:prod.Product_Name||'',product_category:prod.combined_category||'',
      reason:rec.reason||'',intent_label:rec.intent_label||'',intents:rec.intents||[],context:rec._context||{}})});
    box.textContent=resp.explanation||'Matched to your browsing patterns.';
    box.classList.add('show');
  }catch(e){box.textContent='This product matches your recent browsing interests.';box.classList.add('show');}
  btn.textContent='💡';btn.disabled=false;
}

// ── Initial banner load on page start ──────────
async function loadInitialBanners() {
  try {
    const data = await api('/api/products/featured?offset=0&limit=24');
    if (!data || !Array.isArray(data) || data.length === 0) return;
    const candidates = data
      .filter(p => p.primary_image && p.primary_image.startsWith('http'))
      .sort((a, b) => (b.discount_pct || 0) - (a.discount_pct || 0));
    if (candidates.length > 0) renderBannerAds([{product: candidates[0]}]);
  } catch(e) {
    console.warn('Initial banner failed:', e);
  }
}

// Renders a single poster-card banner ad (all text baked into the generated image)
function renderBannerAds(recs){
  const grid = document.getElementById('bannerGrid');
  if (!recs || recs.length === 0) return;
  grid.innerHTML = '';

  const title = document.querySelector('.banner-title');
  if (title && curRecs.length > 0) title.textContent = '🎯 Recommended For You — Generated Poster';

  const r = recs[0];
  const p = r.product || r;

  const poster = document.createElement('div');
  poster.className = 'banner-poster';
  poster.id = 'banner-poster-0';
  poster.innerHTML = `
    <div class="banner-poster-loading" id="banner-load-0">
      <div class="spinner"></div>
      <span>Generating poster…</span>
    </div>
    <img class="banner-poster-img" id="banner-img-0" style="display:none" alt="${esc(p.Product_Name)}">`;
  poster.onclick = () => openProd(p);
  grid.appendChild(poster);

  window._bannerProds = [p];

  const img = document.getElementById('banner-img-0');
  const loader = document.getElementById('banner-load-0');
  img.onload = function() { loader.style.display='none'; img.style.display='block'; };
  img.onerror = function() { loader.style.display='none'; img.src=p.primary_image||'https://via.placeholder.com/480x640?text=Poster'; img.style.display='block'; };
  // Request poster-sized image (portrait 3:4 ratio)
  img.src = '/api/product_ad?product_id=' + p.product_id + '&width=480&t=' + Date.now();
}

// Analyze banner image and apply dynamic placement
async function analyzeBannerPlacement(idx, imageUrl) {
  const rankBadge = document.getElementById(`banner-rank-${idx}`);
  const ctaOverlay = document.getElementById(`banner-cta-${idx}`);
  
  if (!imageUrl || !rankBadge) {
    // Apply default + show badge
    applyDefaultBannerPlacement(idx);
    return;
  }
  
  try {
    const resp = await fetch(`/api/analyze_image_placement?image_url=${encodeURIComponent(imageUrl)}&card_width=320&card_height=400`);
    if (!resp.ok) throw new Error('Banner placement API error');
    const placement = await resp.json();
    
    // Apply dynamic placement to rank badge
    if (placement.rank_badge) {
      Object.assign(rankBadge.style, {
        top: placement.rank_badge.top,
        right: placement.rank_badge.right,
        bottom: placement.rank_badge.bottom,
        left: placement.rank_badge.left,
        transform: placement.rank_badge.transform || 'none'
      });
      rankBadge.classList.add('visible');
    }
    
    // Apply dynamic CTA position based on product orientation
    if (ctaOverlay && placement.debug?.product_region) {
      const prodCenter = placement.debug.product_region.center;
      // If product is in bottom half, move CTA to top
      if (prodCenter && prodCenter[1] > 0.6) {
        ctaOverlay.classList.remove('pos-bottom');
        ctaOverlay.classList.add('pos-top');
      }
    }
    
    console.log(`[Banner ${idx}] Dynamic placement applied:`, placement.rank_badge?.position);
    
  } catch (e) {
    console.warn('Banner placement analysis failed for', idx, e);
    applyDefaultBannerPlacement(idx);
  }
}

function applyDefaultBannerPlacement(idx) {
  const rankBadge = document.getElementById(`banner-rank-${idx}`);
  if (!rankBadge) return;
  
  // Vary default placement by index to avoid identical layouts
  const positions = [
    { top: '12px', left: '12px', right: 'auto', bottom: 'auto' },
    { top: '12px', right: '12px', left: 'auto', bottom: 'auto' },
    { bottom: '60px', left: '12px', right: 'auto', top: 'auto' },
  ];
  const pos = positions[idx % positions.length];
  
  Object.assign(rankBadge.style, pos);
  rankBadge.classList.add('visible');
}

// ── Ad Poster Modal ─────────────────────────
function showAdPoster(productId){
  const modal=document.getElementById('adPosterModal');
  const content=document.getElementById('adPosterContent');
  content.innerHTML='<div class="ad-poster-loading"><div class="spinner"></div>Generating poster...</div>';
  modal.classList.add('show');
  document.body.style.overflow='hidden';
  const img=new Image();
  img.onload=function(){content.innerHTML='';content.appendChild(img);};
  img.onerror=function(){content.innerHTML='<div class="ad-poster-loading">Failed to generate poster</div>';};
  img.src='/api/product_ad?product_id='+productId+'&t='+Date.now();
  img.alt='Product Poster';
}
function closeAdPoster(){
  document.getElementById('adPosterModal').classList.remove('show');
  document.body.style.overflow='';
}

// ── User Panel ─────────────────────────────────
function showUp(tab){
  document.getElementById('upOv').classList.add('show');
  document.getElementById('upPanel').classList.add('show');
  document.querySelectorAll('.up-tab').forEach((t,i)=>t.classList.toggle('active',['cart','wishlist','purchases'][i]===tab));
  curTab=tab;loadTabContent(tab);
}
function closeUp(){document.getElementById('upOv').classList.remove('show');document.getElementById('upPanel').classList.remove('show');}
function switchTab(tab,el){document.querySelectorAll('.up-tab').forEach(t=>t.classList.remove('active'));el.classList.add('active');curTab=tab;loadTabContent(tab);}
async function loadTabContent(tab){
  const body=document.getElementById('upBody');const sum=document.getElementById('upSum');
  body.innerHTML='<div class="loading"><div class="spinner"></div></div>';
  try{
    const m={cart:'/api/user/cart',wishlist:'/api/user/wishlist',purchases:'/api/user/purchases'};
    const items=await api(m[tab]);
    if(!items||!items.length){body.innerHTML='<div class="up-empty">Nothing here yet 👀<br><small>Start exploring!</small></div>';sum.style.display='none';return;}
    body.innerHTML='';let tot=0;
    items.forEach(p=>{
      tot+=p.price||0;const d=document.createElement('div');d.className='up-item';
      d.innerHTML=`<img src="${esc(p.primary_image)}" onerror="this.src='https://via.placeholder.com/60x60'"><div><div class="up-nm">${esc(p.Product_Name)}</div><div class="up-ct">${esc(p.combined_category)}</div><div class="up-pr">${fp(p.price)}</div></div>`;
      d.onclick=()=>openProd(p);body.appendChild(d);
    });
    if(tab==='cart'){sum.style.display='block';document.getElementById('upTotal').textContent=fp(tot);}else sum.style.display='none';
  }catch(e){body.innerHTML='<div class="up-empty">Error loading items.</div>';}
}

// ── User Switcher ──────────────────────────────
function openSwitcher(){document.getElementById('swModal').classList.add('show');loadUsers();}
function closeSwitcher(){document.getElementById('swModal').classList.remove('show');}
async function loadUser(){
  try{const u=await api('/api/current_user');curUser=u.user_id;document.getElementById('curUserLbl').textContent=u.user_id.slice(0,7)+'…';}catch(e){}
}
async function loadUsers(){
  const c=document.getElementById('usrList');c.innerHTML='<div class="loading"><div class="spinner"></div></div>';
  try{
    const us=await api('/api/users');if(!us.length){c.innerHTML='<p style="color:var(--muted);font-size:.8rem">No users yet.</p>';return;}
    c.innerHTML='';
    us.forEach(u=>{
      const row=document.createElement('div');row.className='user-row';
      const me=u.user_id===curUser;
      row.innerHTML=`<div><div class="uid">${u.user_id}${me?' (you)':''}</div><div class="ui">${u.interactions} interactions · ${u.carts} carts · ${u.purchases} purchases</div></div><button onclick="switchToUser('${u.user_id}')" ${me?'disabled':''}>Switch</button>`;
      c.appendChild(row);
    });
  }catch(e){c.innerHTML='<p style="color:var(--muted)">Error.</p>';}
}
async function switchToUser(uid){
  try{
    const r=await api('/api/switch_user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid})});
    if(r.status==='ok'){curUser=uid;hist=r.history||[];document.getElementById('curUserLbl').textContent=uid.slice(0,7)+'…';closeSwitcher();fetchRecs();refreshBadges();toast('Switched to user ✓');}
  }catch(e){}
}
async function createUser(){
  const r=await api('/api/create_user',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  if(r.status==='ok'){curUser=r.user_id;hist=[];closeSwitcher();toast('New user created ✓');}
}
</script>
</body>
</html>'''


ADMIN_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DisReq Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070f;--surface:#0f0f1a;--surface2:#16162a;--accent:#f5c518;--accent2:#e85d7a;--text:#f0eff5;--muted:#7a7a94;--border:rgba(245,197,24,.1);--radius:14px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.nav{background:var(--surface);padding:15px 30px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border)}
.nav h1{font-family:'Playfair Display',serif;color:var(--accent);font-size:1.25rem}
.nav h1 small{display:block;font-family:'DM Sans',sans-serif;font-size:.58rem;color:var(--muted);font-weight:400;letter-spacing:2.5px;text-transform:uppercase}
.nav a{color:var(--accent2);text-decoration:none;font-size:.82rem;font-weight:600;padding:6px 15px;border:1px solid rgba(232,93,122,.25);border-radius:18px;transition:.2s}
.nav a:hover{background:var(--accent2);color:#fff}
.wrap{max-width:1100px;margin:0 auto;padding:26px}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:22px}
.sc{background:var(--surface);border-radius:var(--radius);padding:18px;text-align:center;border:1px solid var(--border);transition:.2s}
.sc:hover{border-color:var(--accent)}.sc .v{font-size:1.9rem;font-weight:700;color:var(--accent);font-family:'Playfair Display',serif}
.sc .l{font-size:.62rem;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:1.5px}
.panel{background:var(--surface);border-radius:var(--radius);padding:20px;margin-bottom:16px;border:1px solid var(--border)}
.panel h3{color:var(--accent);margin-bottom:14px;font-size:.88rem;text-transform:uppercase;letter-spacing:1.2px;font-weight:600}
.ctrls{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.btn{padding:9px 20px;border:none;border-radius:9px;font-weight:700;cursor:pointer;font-size:.8rem;transition:.2s}
.btn-p{background:var(--accent);color:#07070f}.btn-p:hover{filter:brightness(1.08)}
.btn-s{background:var(--surface2);color:var(--accent2);border:1px solid rgba(232,93,122,.25)}.btn-s:hover{background:var(--accent2);color:#fff}
.btn:disabled{opacity:.4;cursor:not-allowed}
.ctrl-lbl{font-size:.75rem;color:var(--muted);display:flex;align-items:center;gap:5px}
.ctrl-lbl input[type=checkbox]{accent-color:var(--accent)}
.schip{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:18px;font-size:.72rem;background:var(--surface2);border:1px solid var(--border)}
.dot{width:7px;height:7px;border-radius:50%}
.dot-i{background:var(--muted)}.dot-t{background:#f39c12;animation:pd .9s infinite}.dot-c{background:#00d4aa}.dot-e{background:var(--accent2)}
@keyframes pd{0%,100%{opacity:1}50%{opacity:.2}}
.prog-wrap{margin-top:16px}
.prog-bar{background:var(--surface2);border-radius:7px;height:20px;overflow:hidden;margin:7px 0;border:1px solid var(--border)}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .5s;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;color:#07070f}
.loss-row{display:flex;gap:14px;font-size:.76rem;color:var(--muted);margin-top:5px}
.loss-row strong{color:var(--text)}
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:9px;margin-top:12px}
.met{background:var(--surface2);padding:13px;border-radius:9px;text-align:center;border:1px solid var(--border)}
.met .v{font-size:1.2rem;font-weight:700;color:var(--accent);font-family:'Playfair Display',serif}
.met .l{font-size:.63rem;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:1px}
.cc{display:inline-block;padding:4px 13px;background:var(--surface2);border:1px solid var(--border);border-radius:14px;font-size:.73rem;color:var(--muted);margin:3px}
.cc strong{color:var(--accent2)}
</style>
</head>
<body>
<nav class="nav">
  <h1>⚙ Admin Dashboard<small>DisReq Recommendation System</small></h1>
  <a href="/">← Back to Store</a>
</nav>
<div class="wrap">
  <div class="sg">
    <div class="sc"><div class="v" id="sP">—</div><div class="l">Products</div></div>
    <div class="sc"><div class="v" id="sU">—</div><div class="l">Users</div></div>
    <div class="sc"><div class="v" id="sI">—</div><div class="l">Interactions</div></div>
    <div class="sc"><div class="v" id="sM">—</div><div class="l">Model</div></div>
  </div>
  <div class="panel">
    <h3>🧠 Model Training</h3>
    <div class="ctrls">
      <button class="btn btn-p" id="tBtn" onclick="startTrain()">▶ Start Training</button>
      <button class="btn btn-s" id="rBtn" onclick="resetAll()">↺ Reset All</button>
      <label class="ctrl-lbl"><input type="checkbox" id="adv"> Advanced (15 epochs)</label>
      <span class="schip" id="tStatus"><span class="dot dot-i"></span> Idle</span>
    </div>
    <div class="prog-wrap" id="progWrap" style="display:none">
      <div style="font-size:.77rem;color:var(--muted)">Epoch <strong id="eInfo" style="color:var(--text)">0/0</strong> · <strong id="tInfo" style="color:var(--text)">0s</strong></div>
      <div class="prog-bar"><div class="prog-fill" id="pFill" style="width:0%">0%</div></div>
      <div class="loss-row">
        <span>S2I: <strong id="lS2I">—</strong></span>
        <span>S2S: <strong id="lS2S">—</strong></span>
        <span>Total: <strong id="lTot">—</strong></span>
      </div>
    </div>
    <div class="mg" id="mgrid" style="display:none">
      <div class="met"><div class="v" id="mRec">—</div><div class="l">Recall@10</div></div>
      <div class="met"><div class="v" id="mNDCG">—</div><div class="l">NDCG@10</div></div>
      <div class="met"><div class="v" id="mMRR">—</div><div class="l">MRR</div></div>
      <div class="met"><div class="v" id="mEnt">—</div><div class="l">Intent Entropy</div></div>
    </div>
  </div>
  <div class="panel"><h3>📂 Product Categories</h3><div id="catList" style="margin-top:5px"></div></div>
</div>
<script>
let poll=null;
window.onload=()=>{stats();cats();setInterval(stats,10000);checkTrain();};
async function api(u,o){const r=await fetch(u,o);return r.json();}
async function stats(){try{const s=await api('/api/stats');document.getElementById('sP').textContent=s.num_products;document.getElementById('sU').textContent=s.num_users;document.getElementById('sI').textContent=s.num_interactions;document.getElementById('sM').textContent=s.model_status;}catch(e){}}
async function cats(){const cs=await api('/api/categories');document.getElementById('catList').innerHTML=cs.map(c=>`<span class="cc">${c.name} <strong>(${c.count})</strong></span>`).join('');}
async function startTrain(){document.getElementById('tBtn').disabled=true;await api('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({advanced:document.getElementById('adv').checked})});document.getElementById('progWrap').style.display='block';startPoll();}
function startPoll(){if(poll)clearInterval(poll);poll=setInterval(doPoll,1500);}
async function checkTrain(){try{const p=await api('/api/training_progress');if(p.status==='training'){document.getElementById('progWrap').style.display='block';document.getElementById('tBtn').disabled=true;startPoll();}updProg(p);}catch(e){}}
async function doPoll(){try{const p=await api('/api/training_progress');updProg(p);if(p.status!=='training'){clearInterval(poll);poll=null;document.getElementById('tBtn').disabled=false;}}catch(e){}}
async function resetAll(){
  if(!confirm('Reset all training, model, and user data? Cannot be undone.'))return;
  document.getElementById('rBtn').disabled=true;
  try{const r=await api('/api/reset_training',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(r.status==='ok'){document.getElementById('progWrap').style.display='none';document.getElementById('mgrid').style.display='none';document.getElementById('tStatus').innerHTML='<span class="dot dot-i"></span> Idle';stats();alert('Reset complete.');}else alert(r.message||'Failed.');}catch(e){alert('Error: '+e);}
  document.getElementById('rBtn').disabled=false;
}
function updProg(p){
  const cls={training:'t',completed:'c',idle:'i'}[p.status]||(p.status&&p.status.startsWith('error')?'e':'i');
  document.getElementById('tStatus').innerHTML=`<span class="dot dot-${cls}"></span> ${p.status}`;
  const pct=Math.round(p.batch_progress||0);
  document.getElementById('pFill').style.width=pct+'%';document.getElementById('pFill').textContent=pct+'%';
  document.getElementById('eInfo').textContent=p.epoch+'/'+p.total_epochs;
  document.getElementById('tInfo').textContent=Math.round(p.time_elapsed||0)+'s';
  document.getElementById('lS2I').textContent=(p.loss_s2i||0).toFixed(4);
  document.getElementById('lS2S').textContent=(p.loss_s2s||0).toFixed(4);
  document.getElementById('lTot').textContent=(p.loss_total||0).toFixed(4);
  if(p.metrics&&Object.keys(p.metrics).length>0){
    document.getElementById('mgrid').style.display='grid';
    document.getElementById('mRec').textContent=(p.metrics['recall@10']||0).toFixed(4);
    document.getElementById('mNDCG').textContent=(p.metrics['ndcg@10']||0).toFixed(4);
    document.getElementById('mMRR').textContent=(p.metrics['mrr']||0).toFixed(4);
    document.getElementById('mEnt').textContent=(p.metrics['intent_entropy']||0).toFixed(4);
  }
}
</script>
</body>
</html>'''


# ── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  DisReq - E-Commerce Recommendation System")
    print("  Single-File Edition (Enhanced SASRec with Intent Gating)")
    print("="*60)

    load_product_data()
    
    # Initialize CATALOG and SESSION_GENERATOR after loading data
    if CATALOG is not None and SESSION_GENERATOR is not None:
        print("  [OK] Catalog and session generator initialized")
    else:
        print("  [X] WARNING: CATALOG or SESSION_GENERATOR not initialized")

    # Try to load saved model weights using new architecture
    for mf in ['final_model.pth', 'best_model.pth']:
        mf_path = os.path.join(os.path.dirname(__file__), mf)
        if os.path.exists(mf_path) and CATALOG is not None:
            try:
                loaded = EnhancedSASRec(
                    catalog=CATALOG,
                    hidden_dim=128,
                    num_blocks=2,
                    num_intents=4
                ).to(device)
                loaded.load_state_dict(torch.load(mf_path, map_location=device, weights_only=False))
                MODEL = loaded
                print(f"  [OK] Loaded model: {mf}")
                break
            except Exception as e:
                print(f"  [X] Could not load {mf}: {e}")
                # Create fresh model if can't load weights
                MODEL = EnhancedSASRec(
                    catalog=CATALOG,
                    hidden_dim=128,
                    num_blocks=2,
                    num_intents=4
                ).to(device)
                print("  [OK] Created fresh model (weights incompatible)")
                break
    else:
        # No saved model found, create fresh one
        if CATALOG is not None:
            MODEL = EnhancedSASRec(
                catalog=CATALOG,
                hidden_dim=128,
                num_blocks=2,
                num_intents=4
            ).to(device)
            print("  [OK] Created fresh model (no saved weights)")

    print("\n  [Home] Store: http://localhost:5002")
    print("  [Admin] Admin: http://localhost:5002/admin")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5002, host='0.0.0.0', use_reloader=False)