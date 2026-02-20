import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from urllib.parse import urlparse
import time
from datetime import datetime
import os
import re

# ── Shared feature extractor (identical to training) ─────────────────────────
from feature_extractor import extract_features, FEATURE_NAMES

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.8rem; font-weight: bold; text-align: center;
    background: linear-gradient(120deg, #2563eb, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.phishing-alert {
    background-color: #fee2e2; border-left: 4px solid #dc2626;
    padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;
}
.safe-alert {
    background-color: #d1fae5; border-left: 4px solid #10b981;
    padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        with open('phishing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        feature_importance = None
        if os.path.exists('feature_importance.csv'):
            feature_importance = pd.read_csv('feature_importance.csv')
        return model, scaler, metrics, feature_importance
    except FileNotFoundError:
        st.error("❌ Model files not found! Run: `python train_model_advanced.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


# ── Rule-based overrides for signals the ML model may underweight ─────────────
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
                   '.work', '.click', '.pw', '.su', '.icu', '.cyou', '.monster']

def rule_based_check(url):
    url_lower = url.lower()
    # Suspicious TLD
    for tld in SUSPICIOUS_TLDS:
        if (url_lower.endswith(tld) or f"{tld}/" in url_lower
                or f"{tld}?" in url_lower or f"{tld}#" in url_lower):
            return True, f"Suspicious free TLD detected ({tld})"
    # @ in URL
    if '@' in url:
        return True, "@ symbol in URL — can mask the real destination"
    # Multiple protocols
    if url.count('//') > 1:
        return True, "Multiple '//' found in URL"
    # Bare IP address
    if re.search(r'https?://(\d{1,3}\.){3}\d{1,3}', url):
        return True, "IP address used instead of a domain name"
    return False, None


def gauge_chart(probability):
    color = "green" if probability < 0.3 else ("orange" if probability < 0.7 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Phishing Probability (%)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': color},
            'steps': [
                {'range': [0,  30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100],'color': '#fee2e2'},
            ],
            'threshold': {'line': {'color': 'red', 'width': 4},
                          'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def pie_chart(probabilities):
    fig = go.Figure(data=[go.Pie(
        labels=['Legitimate', 'Phishing'],
        values=[probabilities[0]*100, probabilities[1]*100],
        hole=0.4,
        marker=dict(colors=['#10b981', '#dc2626']),
        textinfo='label+percent', textfont_size=15
    )])
    fig.update_layout(title="Classification Confidence", height=280,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.markdown('<h1 class="main-header">🔒 Advanced Phishing URL Detector</h1>',
                unsafe_allow_html=True)
    st.markdown("#### Powered by Machine Learning Ensemble Models ")

    model, scaler, metrics, feature_importance = load_models()

    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("Recall",    f"{metrics['recall']*100:.2f}%")
        st.metric("F1-Score",  f"{metrics['f1_score']*100:.2f}%")
        st.metric("ROC-AUC",   f"{metrics['roc_auc']*100:.2f}%")
        st.markdown("---")
        st.info(f"**{len(FEATURE_NAMES)} features** extracted from URL structure.\n\n"
                "No network calls — instant analysis.")

    st.markdown("---")

    col1, col2 = st.columns([4, 1])
    with col1:
        url_input = st.text_input("🔗 Enter URL to analyze:",
                                  placeholder="https://example.com")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with st.expander("💡 Example URLs"):
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Legitimate:**")
            st.code("https://www.google.com")
            st.code("https://www.github.com")
        with cb:
            st.markdown("**Suspicious:**")
            st.code("http://suspicious-site.tk")
            st.code("http://paypal-verify.ml")
            st.code("http://amazon-login-secure.xyz/update")

    if analyze_btn and url_input:
        with st.spinner("Analyzing..."):
            time.sleep(0.2)
            try:
                # 1. Rule-based check
                is_phishing_rule, rule_reason = rule_based_check(url_input)

                # 2. ML prediction — uses EXACT same features as training
                feat_dict     = extract_features(url_input)
                feature_array = np.array(
                    [feat_dict.get(fn, 0) for fn in FEATURE_NAMES]
                ).reshape(1, -1)
                scaled        = scaler.transform(feature_array)
                ml_pred       = model.predict(scaled)[0]
                ml_proba      = model.predict_proba(scaled)[0]
                ml_phish_prob = float(ml_proba[1])

                # 3. Combine
                if is_phishing_rule:
                    final_pred  = 1
                    phish_prob  = max(ml_phish_prob, 0.85)
                    note        = f"⚠️ Rule override: {rule_reason}"
                else:
                    final_pred  = ml_pred
                    phish_prob  = ml_phish_prob
                    note        = None

                legit_prob    = 1.0 - phish_prob

                # Display
                st.markdown("---")
                st.markdown("## 📊 Results")

                if note:
                    st.warning(note)

                if final_pred == 1:
                    st.markdown("""
                    <div class="phishing-alert">
                        <h2 style="color:#dc2626;margin:0">⚠️ PHISHING DETECTED</h2>
                        <p style="margin:.4rem 0 0 0">Strong phishing signals found. Do not visit this URL!</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe-alert">
                        <h2 style="color:#10b981;margin:0">✅ Appears Legitimate</h2>
                        <p style="margin:.4rem 0 0 0">No phishing signals detected. Always stay cautious.</p>
                    </div>""", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction",   "🚨 PHISHING" if final_pred else "✅ SAFE")
                c2.metric("Phishing Prob", f"{phish_prob*100:.1f}%")
                c3.metric("Confidence",    f"{max(phish_prob, legit_prob)*100:.1f}%")
                risk = ("🔴 HIGH" if phish_prob > 0.7
                        else "🟡 MEDIUM" if phish_prob > 0.3 else "🟢 LOW")
                c4.metric("Risk Level", risk)

                st.markdown("---")
                g_col, p_col = st.columns(2)
                with g_col:
                    st.plotly_chart(gauge_chart(phish_prob), use_container_width=True)
                with p_col:
                    st.plotly_chart(pie_chart([legit_prob, phish_prob]),
                                    use_container_width=True)

                if feature_importance is not None:
                    st.markdown("### 🔎 Top 15 Feature Importances")
                    st.dataframe(feature_importance.head(15),
                                 use_container_width=True, hide_index=True)

                with st.expander("🔬 All Extracted Features for This URL"):
                    st.dataframe(
                        pd.DataFrame(list(feat_dict.items()),
                                     columns=['Feature', 'Value']),
                        use_container_width=True, hide_index=True
                    )

                with st.expander("🔗 URL Structure Breakdown"):
                    parsed = urlparse(url_input if url_input.startswith('http')
                                      else 'http://' + url_input)
                    x, y = st.columns(2)
                    with x:
                        st.code(f"Scheme : {parsed.scheme or 'N/A'}")
                        st.code(f"Domain : {parsed.netloc or 'N/A'}")
                        st.code(f"Path   : {parsed.path or '/'}")
                    with y:
                        st.code(f"Length : {len(url_input)}")
                        st.code(f"HTTPS  : {'Yes' if parsed.scheme == 'https' else 'No'}")
                        st.code(f"Query  : {parsed.query or 'None'}")

                st.caption(f"Analyzed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

    elif analyze_btn:
        st.warning("Please enter a URL.")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6b7280;padding:1rem 0">
        <strong>Phishing URL Detection System</strong> · Stay safe online 🔒
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
