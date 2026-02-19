"""
Shared URL feature extraction module.
Used by both train_model_advanced.py and app_streamlit.py
to guarantee identical features at training and inference time.
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse


def extract_features(url):
    """
    Extract numerical features from a URL without any network calls.
    Returns a dict of feature_name -> float value.
    """
    features = {}
    try:
        url = str(url).strip()
        url_for_parse = url if url.startswith('http') else 'http://' + url
        parsed = urlparse(url_for_parse)
        domain = parsed.netloc.lower()
        path   = parsed.path
        query  = parsed.query

        # ── Length features ──────────────────────────────────────────────
        features['url_length']    = len(url)
        features['domain_length'] = len(domain)
        features['path_length']   = len(path)
        features['query_length']  = len(query)

        # ── Character counts ─────────────────────────────────────────────
        features['num_dots']           = url.count('.')
        features['num_hyphens']        = url.count('-')
        features['num_underscores']    = url.count('_')
        features['num_slashes']        = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equal']          = url.count('=')
        features['num_at']             = url.count('@')
        features['num_ampersand']      = url.count('&')
        features['num_percent']        = url.count('%')
        features['num_hash']           = url.count('#')
        features['num_digits']         = sum(c.isdigit() for c in url)
        features['num_uppercase']      = sum(c.isupper() for c in url)

        special = ['!', '#', '$', '^', '*', '+', '[', ']', '{', '}',
                   '|', '\\', ':', ';', '"', "'", '<', '>', ',', '~']
        features['num_special_chars'] = sum(url.count(c) for c in special)

        # ── Ratios ───────────────────────────────────────────────────────
        url_len = max(len(url), 1)
        features['digit_ratio']       = features['num_digits'] / url_len
        features['hyphen_ratio']      = features['num_hyphens'] / url_len
        features['special_char_ratio']= features['num_special_chars'] / url_len
        features['domain_url_ratio']  = len(domain) / url_len

        # ── Protocol ─────────────────────────────────────────────────────
        features['is_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_port'] = 1 if parsed.port else 0

        # ── Domain analysis ──────────────────────────────────────────────
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        features['has_ip'] = 1 if ip_pattern.match(domain) else 0

        domain_parts = domain.split('.')
        features['num_subdomains']     = max(0, len(domain_parts) - 2)
        features['domain_has_digits']  = 1 if any(c.isdigit() for c in domain) else 0
        features['domain_has_hyphen']  = 1 if '-' in domain else 0
        features['prefix_suffix_dash'] = 1 if '-' in domain else 0

        base_domain = domain_parts[-2] if len(domain_parts) >= 2 else domain
        features['base_domain_length'] = len(base_domain)

        digits_in_domain  = sum(c.isdigit() for c in domain)
        letters_in_domain = sum(c.isalpha() for c in domain)
        features['digit_letter_ratio'] = digits_in_domain / max(letters_in_domain, 1)

        # ── TLD suspiciousness ───────────────────────────────────────────
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz',
                           '.top', '.work', '.click', '.link', '.info',
                           '.pw', '.cc', '.su', '.ws', '.biz', '.online',
                           '.site', '.icu', '.cyou', '.monster']
        url_lower = url.lower()
        tld_hit = 1 if any(url_lower.endswith(t) or
                           ('.' + url_lower.split('.')[-1]) == t
                           for t in suspicious_tlds) else 0
        features['has_suspicious_tld']    = tld_hit
        features['suspicious_tld_weight'] = tld_hit * 3   # amplified signal

        # ── Suspicious keywords ──────────────────────────────────────────
        susp_words = ['login', 'signin', 'bank', 'account', 'update', 'secure',
                      'verify', 'confirm', 'password', 'paypal', 'suspended',
                      'locked', 'urgent', 'alert', 'warning', 'billing', 'free',
                      'winner', 'prize', 'click', 'limited', 'offer', 'deal']
        features['has_suspicious_words'] = 1 if any(w in url_lower for w in susp_words) else 0
        features['num_suspicious_words'] = sum(1 for w in susp_words if w in url_lower)

        # ── Brand impersonation ──────────────────────────────────────────
        brands = ['google', 'facebook', 'amazon', 'microsoft', 'apple',
                  'netflix', 'paypal', 'ebay', 'instagram', 'twitter',
                  'whatsapp', 'linkedin', 'yahoo', 'outlook', 'dropbox']
        features['has_brand_name'] = 1 if any(b in url_lower for b in brands) else 0
        # Brand in domain but domain isn't the real brand = strong phishing signal
        brand_in_domain     = any(b in domain for b in brands)
        is_real_brand_domain= any(domain.endswith(b + '.com') or
                                  domain.endswith(b + '.org') or
                                  domain == b + '.com'
                                  for b in brands)
        features['brand_in_domain_not_real'] = 1 if (brand_in_domain and not is_real_brand_domain) else 0

        # ── Path features ────────────────────────────────────────────────
        features['path_depth']   = len([p for p in path.split('/') if p])
        features['num_params']   = len(query.split('&')) if query else 0
        features['double_slash'] = 1 if '//' in path else 0

        # ── URL shorteners ───────────────────────────────────────────────
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd']
        features['is_shortened'] = 1 if any(s in domain for s in shorteners) else 0

        # ── Entropy (randomness) ─────────────────────────────────────────
        c = Counter(url)
        n = len(url)
        features['url_entropy'] = -sum((v/n) * math.log2(v/n) for v in c.values()) if n else 0

        if domain:
            cd = Counter(domain)
            nd = len(domain)
            features['domain_entropy'] = -sum((v/nd) * math.log2(v/nd)
                                               for v in cd.values()) if nd else 0
        else:
            features['domain_entropy'] = 0

        # ── Misc ─────────────────────────────────────────────────────────
        features['consecutive_dots'] = 1 if '..' in url else 0
        features['has_www']          = 1 if 'www.' in domain else 0

    except Exception:
        # Return all zeros on any parse failure
        for k in list(features.keys()):
            features[k] = 0

    return features


# Canonical ordered list — MUST stay in sync with training
FEATURE_NAMES = list(extract_features('http://example.com').keys())
