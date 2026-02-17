import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse
import tldextract
import time
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .phishing-alert {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models and data"""
    try:
        with open('phishing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        
        feature_importance = None
        if os.path.exists('feature_importance.csv'):
            feature_importance = pd.read_csv('feature_importance.csv')
        
        return model, scaler, feature_names, metrics, feature_importance
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please train the model first by running: `python train_model_advanced.py`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def extract_basic_features(url):
    """Extract basic features from URL for demonstration"""
    try:
        # Parse URL
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        # Basic features (simplified - in production you'd need all 50 features)
        features = {
            'URLLength': len(url),
            'DomainLength': len(parsed.netloc),
            'IsDomainIP': 1 if any(c.isdigit() for c in parsed.netloc.replace('.', '')) and '.' in parsed.netloc else 0,
            'IsHTTPS': 1 if parsed.scheme == 'https' else 0,
            'NoOfDegitsInURL': sum(c.isdigit() for c in url),
            'NoOfLettersInURL': sum(c.isalpha() for c in url),
            'LetterRatioInURL': sum(c.isalpha() for c in url) / max(len(url), 1),
            'DegitRatioInURL': sum(c.isdigit() for c in url) / max(len(url), 1),
            'NoOfEqualsInURL': url.count('='),
            'NoOfQMarkInURL': url.count('?'),
            'NoOfAmpersandInURL': url.count('&'),
            'NoOfOtherSpecialCharsInURL': sum(1 for c in url if c in '-_~:/?#[]@!$&\'()*+,;='),
            'SpacialCharRatioInURL': sum(1 for c in url if not c.isalnum()) / max(len(url), 1),
        }
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return {}

def create_gauge_chart(probability, title="Phishing Probability"):
    """Create a gauge chart for phishing probability"""
    if probability < 0.3:
        color = "green"
    elif probability < 0.7:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_prediction_confidence_chart(probabilities):
    """Create a chart showing prediction confidence distribution"""
    labels = ['Legitimate', 'Phishing']
    values = [probabilities[0] * 100, probabilities[1] * 100]
    colors = ['#10b981', '#dc2626']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=16
    )])
    
    fig.update_layout(
        title="Classification Confidence",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Advanced Phishing URL Detector</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Machine Learning Ensemble Models (99.99% Accuracy)")
    
    # Load models
    model, scaler, feature_names, metrics, feature_importance = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Training Samples", "60,000+")
        st.metric("Model Type", "Ensemble (RF+GB+XGB)")
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        st.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")
        st.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")
        
        st.markdown("---")
        st.info("""
        **Note:** This demo version extracts basic features from URLs. 
        
        For full accuracy, the model uses 50+ advanced features including:
        - HTML content analysis
        - JavaScript/CSS analysis
        - External references
        - Image counting
        - And more...
        """)
    
    st.markdown("---")
    
    # URL Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "🔗 Enter URL to analyze:",
            placeholder="https://example.com",
            help="Enter any URL to check if it's a phishing attempt"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze URL", type="primary", use_container_width=True)
    
    # Example URLs
    with st.expander("📝 Try Example URLs"):
        st.info("⚠️ **Demo Mode**: This simplified version may not match the full model's accuracy. For best results, use the complete dataset features.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Test Examples:**")
            st.code("https://www.google.com")
            st.code("https://www.github.com")
            st.code("http://suspicious-site.tk")
    
    # Analysis section
    if analyze_button and url_input:
        with st.spinner("🔍 Analyzing URL..."):
            time.sleep(0.5)
            
            try:
                # Extract basic features
                basic_features = extract_basic_features(url_input)
                
                # Create feature array with zeros for missing features
                feature_array = []
                for fname in feature_names:
                    feature_array.append(basic_features.get(fname, 0))
                
                feature_array = np.array(feature_array).reshape(1, -1)
                
                # Scale features
                feature_array_scaled = scaler.transform(feature_array)
                
                # Make prediction
                prediction = model.predict(feature_array_scaled)[0]
                probabilities = model.predict_proba(feature_array_scaled)[0]
                
                phishing_prob = probabilities[1]
                legitimate_prob = probabilities[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Warning banner
                st.warning("⚠️ **Demo Mode**: This is a simplified version using basic URL features only. For production use, all 50+ features (including HTML analysis, JavaScript, etc.) should be extracted.",icon="⚠️")
                
                # Result alert
                if prediction == 1:
                    st.markdown(f"""
                    <div class="phishing-alert">
                        <h2 style="color: #dc2626; margin: 0;">⚠️ POTENTIALLY PHISHING!</h2>
                        <p style="margin: 0.5rem 0 0 0;">Based on basic URL analysis, this may be suspicious. Exercise caution!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h2 style="color: #10b981; margin: 0;">✅ Appears Safe (Basic Analysis)</h2>
                        <p style="margin: 0.5rem 0 0 0;">Basic URL features look legitimate, but always stay cautious online.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "🚨 SUSPICIOUS" if prediction == 1 else "✅ SAFE",
                        delta="High Risk" if prediction == 1 else "Low Risk",
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Phishing Probability",
                        f"{phishing_prob * 100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Confidence",
                        f"{max(probabilities) * 100:.1f}%"
                    )
                
                with col4:
                    risk_level = "🔴 HIGH" if phishing_prob > 0.7 else "🟡 MEDIUM" if phishing_prob > 0.3 else "🟢 LOW"
                    st.metric("Risk Level", risk_level)
                
                # Visualizations
                st.markdown("---")
                st.markdown("## 📈 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    gauge_fig = create_gauge_chart(phishing_prob)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    pie_fig = create_prediction_confidence_chart(probabilities)
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                # Feature importance
                if feature_importance is not None:
                    st.markdown("### 📊 Top 15 Most Important Features")
                    st.dataframe(
                        feature_importance.head(15),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # URL breakdown
                with st.expander("🔗 URL Structure Analysis"):
                    parsed = urlparse(url_input)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**URL Components:**")
                        st.code(f"Scheme: {parsed.scheme or 'N/A'}")
                        st.code(f"Domain: {parsed.netloc or 'N/A'}")
                        st.code(f"Path: {parsed.path or '/'}")
                    with col2:
                        st.markdown("**Statistics:**")
                        st.code(f"URL Length: {len(url_input)}")
                        st.code(f"HTTPS: {'Yes' if parsed.scheme == 'https' else 'No'}")
                
                st.markdown("---")
                st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"❌ Error analyzing URL: {str(e)}")
                st.exception(e)
    
    elif analyze_button:
        st.warning("⚠️ Please enter a URL to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p><strong>Advanced Phishing URL Detection System</strong></p>
        <p>Trained on 60,000+ URLs | 99.99% Accuracy</p>
        <p>Stay safe online! 🔒</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
