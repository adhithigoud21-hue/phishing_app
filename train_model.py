import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import pickle
from url_features import extract_features

def create_comprehensive_dataset():
    """Create a comprehensive dataset with realistic phishing and legitimate URLs"""
    
    # Extended legitimate URLs (100+)
    legitimate_urls = [
        # Major tech companies
        'https://www.google.com', 'https://www.facebook.com', 'https://www.amazon.com',
        'https://www.microsoft.com', 'https://www.apple.com', 'https://www.netflix.com',
        'https://www.twitter.com', 'https://www.linkedin.com', 'https://www.instagram.com',
        'https://www.youtube.com', 'https://www.reddit.com', 'https://www.github.com',
        
        # Banks and financial
        'https://www.chase.com', 'https://www.bankofamerica.com', 'https://www.wellsfargo.com',
        'https://www.paypal.com', 'https://www.stripe.com', 'https://www.square.com',
        
        # E-commerce
        'https://www.ebay.com', 'https://www.walmart.com', 'https://www.target.com',
        'https://www.bestbuy.com', 'https://www.etsy.com', 'https://www.aliexpress.com',
        
        # News and media
        'https://www.cnn.com', 'https://www.bbc.com', 'https://www.nytimes.com',
        'https://www.wsj.com', 'https://www.theguardian.com', 'https://www.reuters.com',
        
        # Education
        'https://www.wikipedia.org', 'https://www.coursera.org', 'https://www.khanacademy.org',
        'https://www.udemy.com', 'https://www.edx.org', 'https://www.mit.edu',
        
        # Tech and development
        'https://www.stackoverflow.com', 'https://www.w3schools.com', 'https://www.mozilla.org',
        'https://www.python.org', 'https://www.java.com', 'https://www.docker.com',
        
        # Cloud services
        'https://www.dropbox.com', 'https://drive.google.com', 'https://www.icloud.com',
        'https://onedrive.live.com', 'https://www.box.com',
        
        # Email services
        'https://mail.google.com', 'https://outlook.live.com', 'https://mail.yahoo.com',
        
        # Government
        'https://www.usa.gov', 'https://www.irs.gov', 'https://www.ssa.gov',
        
        # More legitimate with subdomains
        'https://support.google.com', 'https://developer.apple.com', 'https://docs.microsoft.com',
        'https://help.netflix.com', 'https://blog.medium.com', 'https://status.slack.com',
        
        # Additional legitimate sites
        'https://www.adobe.com', 'https://www.spotify.com', 'https://www.zoom.us',
        'https://www.slack.com', 'https://www.shopify.com', 'https://www.wordpress.com',
        'https://www.quora.com', 'https://www.medium.com', 'https://www.pinterest.com',
        'https://www.tumblr.com', 'https://www.twitch.tv', 'https://www.vimeo.com',
    ]
    
    # Extended phishing URLs (100+) - realistic patterns
    phishing_urls = [
        # Typosquatting
        'http://www.g00gle.com', 'http://www.facebok.com', 'http://www.amaz0n.com',
        'http://www.paypa1.com', 'http://www.netfliix.com', 'http://www.micros0ft.com',
        
        # Subdomain tricks
        'http://paypal.secure-login.com', 'http://amazon.account-verify.net',
        'http://facebook.security-check.org', 'http://google.login-verify.com',
        'http://microsoft.update-required.info', 'http://apple.support-id.net',
        
        # Suspicious keywords
        'http://secure-paypal-login.com', 'https://amazon-account-verify.net',
        'http://facebook-password-reset.org', 'https://google-security-alert.com',
        'http://microsoft-account-update.info', 'https://apple-id-locked.net',
        'http://netflix-billing-update.org', 'https://instagram-verify-account.com',
        'http://linkedin-security-check.net', 'https://twitter-account-suspended.org',
        
        # IP addresses
        'http://192.168.1.1/paypal', 'http://10.0.0.1/amazon',
        'http://172.16.0.1/facebook', 'http://123.456.789.012/bank',
        
        # Suspicious TLDs
        'http://free-paypal.tk', 'http://amazon-gift.ml', 'http://netflix-free.ga',
        'http://apple-store.cf', 'http://microsoft-office.gq', 'http://google-drive.xyz',
        
        # Multiple hyphens
        'http://secure-online-banking-login.com', 'http://verify-your-account-now.net',
        'http://update-payment-method-here.org', 'http://confirm-identity-required.info',
        
        # Random characters
        'http://paypal-xj2k9.com', 'http://amazon-v3r1fy.net', 'http://secure-pg8x.org',
        'http://login-qw9r.com', 'http://account-8k2j.info', 'http://verify-3x9p.net',
        
        # Long URLs with many parameters
        'http://secure-login.com/verify?id=12345&token=abc&redirect=paypal',
        'http://account-update.net/confirm?user=test&session=xyz&auth=true',
        'http://payment-verify.org/check?account=demo&validate=yes&secure=1',
        
        # Shortened URL patterns (suspicious)
        'http://bit.ly/2x8k9pq', 'http://tinyurl.com/paypal123', 'http://goo.gl/abc123',
        
        # Homograph attacks (look-alike characters)
        'http://www.gоogle.com', 'http://www.faceb00k.com', 'http://www.аmazon.com',
        
        # Multiple subdomains
        'http://secure.login.paypal.verify.com', 'http://account.update.amazon.check.net',
        'http://security.alert.facebook.confirm.org', 'http://billing.update.netflix.verify.com',
        
        # Misspellings with suspicious words
        'http://paypai-secure.com', 'http://amazen-login.net', 'http://gogle-verify.org',
        'http://facbook-security.com', 'http://nefflix-billing.info', 'http://mikrosoft-update.net',
        
        # Mixed case tricks
        'http://PayPal-Secure.com', 'http://AMAZON-verify.net', 'http://FaceBook-Login.org',
        
        # @ symbol tricks
        'http://paypal.com@phishing.com', 'http://amazon.com@malicious.net',
        
        # Port numbers (unusual)
        'http://secure-login.com:8080', 'http://account-verify.net:3000',
        
        # Excessive special characters
        'http://secure!!!-login.com', 'http://verify***account.net', 'http://update###payment.org',
        
        # No HTTPS with banking keywords
        'http://online-banking-secure.com', 'http://credit-card-verify.net',
        'http://bank-account-update.org', 'http://payment-gateway-secure.info',
        
        # Brand + suspicious word combinations
        'http://paypal-suspended.com', 'http://amazon-locked.net', 'http://netflix-expired.org',
        'http://apple-verify-now.com', 'http://microsoft-urgent.info', 'http://google-warning.net',
        
        # Uncommon character patterns
        'http://www_paypal.com', 'http://paypal_secure.net', 'http://amazon__login.org',
    ]
    
    print(f"Legitimate URLs: {len(legitimate_urls)}")
    print(f"Phishing URLs: {len(phishing_urls)}")
    
    # Create DataFrame
    data = []
    
    for url in legitimate_urls:
        features, _ = extract_features(url)
        features['label'] = 0  # Legitimate
        data.append(features)
    
    for url in phishing_urls:
        features, _ = extract_features(url)
        features['label'] = 1  # Phishing
        data.append(features)
    
    df = pd.DataFrame(data)
    return df

def train_model():
    """Train an advanced Gradient Boosting model"""
    
    print("=" * 70)
    print("ADVANCED PHISHING DETECTION MODEL TRAINING")
    print("=" * 70)
    
    print("\nCreating comprehensive dataset...")
    df = create_comprehensive_dataset()
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Legitimate URLs: {(df['label'] == 0).sum()}")
    print(f"  Phishing URLs: {(df['label'] == 1).sum()}")
    print(f"  Number of features: {len(df.columns) - 1}")
    
    # Prepare features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"\nFeatures being used: {list(X.columns)[:10]}... ({len(X.columns)} total)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    print("\n" + "=" * 70)
    print("TRAINING GRADIENT BOOSTING CLASSIFIER...")
    print("=" * 70)
    
    # Advanced model configuration
    model = GradientBoostingClassifier(
        n_estimators=200,         # More trees for better accuracy
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    print(f"\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Legit  Phish")
    print(f"  Actual Legit    {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"         Phish    {cm[1][0]:3d}    {cm[1][1]:3d}")
    
    # Cross-validation
    print(f"\n🔄 Cross-Validation (5-fold):")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Save the model
    print(f"\n💾 Saving model...")
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✅ Model saved as 'phishing_model.pkl'")
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE! You can now run: python app.py")
    print(f"{'=' * 70}\n")
    
    return model

if __name__ == '__main__':
    train_model()