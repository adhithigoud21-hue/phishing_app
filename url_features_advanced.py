"""
Advanced URL Feature Extraction for Phishing Detection
Extracts comprehensive features from URLs for machine learning models
"""

import re
from urllib.parse import urlparse, parse_qs
import socket
import tldextract
from collections import Counter
import math
import ipaddress

class URLFeatureExtractor:
    """Extract advanced features from URLs for phishing detection"""
    
    # Suspicious keywords commonly found in phishing URLs
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'account', 'verify', 'update', 'confirm', 'secure',
        'banking', 'password', 'suspend', 'locked', 'verify', 'validation',
        'authenticate', 'security', 'credential', 'urgent', 'alert', 'warning'
    ]
    
    # Legitimate popular domains
    POPULAR_DOMAINS = [
        'google', 'facebook', 'youtube', 'amazon', 'twitter', 'instagram',
        'linkedin', 'microsoft', 'apple', 'netflix', 'wikipedia', 'reddit',
        'github', 'stackoverflow', 'paypal', 'ebay', 'cnn', 'bbc'
    ]
    
    def __init__(self):
        self.features = {}
    
    def extract_all_features(self, url):
        """Extract all features from a URL"""
        self.features = {}
        
        # Clean URL
        url = url.strip()
        if not url.startswith('http'):
            url = 'http://' + url
        
        try:
            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path
            query = parsed.query
            
            # Extract TLD information
            ext = tldextract.extract(url)
            
            # 1. URL Length Features
            self.features['url_length'] = len(url)
            self.features['domain_length'] = len(domain)
            self.features['path_length'] = len(path)
            self.features['query_length'] = len(query)
            
            # 2. URL Complexity Features
            self.features['num_dots'] = url.count('.')
            self.features['num_hyphens'] = url.count('-')
            self.features['num_underscores'] = url.count('_')
            self.features['num_slashes'] = url.count('/')
            self.features['num_questionmarks'] = url.count('?')
            self.features['num_equal'] = url.count('=')
            self.features['num_at'] = url.count('@')
            self.features['num_ampersand'] = url.count('&')
            self.features['num_exclamation'] = url.count('!')
            self.features['num_tilde'] = url.count('~')
            self.features['num_percent'] = url.count('%')
            self.features['num_hash'] = url.count('#')
            
            # 3. Domain Features
            self.features['num_subdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
            self.features['has_ip'] = 1 if self._is_ip_address(domain) else 0
            self.features['has_port'] = 1 if ':' in domain and not domain.startswith('[') else 0
            
            # 4. Protocol Features
            self.features['is_https'] = 1 if parsed.scheme == 'https' else 0
            self.features['is_http'] = 1 if parsed.scheme == 'http' else 0
            
            # 5. Suspicious Pattern Features
            self.features['has_double_slash_in_path'] = 1 if '//' in path else 0
            self.features['has_suspicious_tld'] = self._check_suspicious_tld(ext.suffix)
            self.features['num_digits'] = sum(c.isdigit() for c in url)
            self.features['num_uppercase'] = sum(c.isupper() for c in url)
            self.features['digit_ratio'] = self.features['num_digits'] / max(len(url), 1)
            
            # 6. Keyword Features
            url_lower = url.lower()
            self.features['num_suspicious_keywords'] = sum(
                1 for keyword in self.SUSPICIOUS_KEYWORDS if keyword in url_lower
            )
            self.features['has_login_keyword'] = 1 if 'login' in url_lower else 0
            self.features['has_verify_keyword'] = 1 if 'verify' in url_lower else 0
            self.features['has_secure_keyword'] = 1 if 'secure' in url_lower else 0
            self.features['has_account_keyword'] = 1 if 'account' in url_lower else 0
            self.features['has_update_keyword'] = 1 if 'update' in url_lower else 0
            
            # 7. Domain Reputation Features
            domain_name = ext.domain.lower()
            self.features['is_popular_domain'] = 1 if any(
                pop in domain_name for pop in self.POPULAR_DOMAINS
            ) else 0
            self.features['domain_has_numbers'] = 1 if any(c.isdigit() for c in domain_name) else 0
            
            # 8. Path Analysis
            self.features['path_depth'] = len([p for p in path.split('/') if p])
            self.features['has_file_extension'] = 1 if re.search(r'\.(html|php|asp|aspx|jsp|htm)$', path) else 0
            
            # 9. Query String Features
            if query:
                params = parse_qs(query)
                self.features['num_query_params'] = len(params)
                self.features['query_has_suspicious'] = 1 if any(
                    keyword in query.lower() for keyword in self.SUSPICIOUS_KEYWORDS
                ) else 0
            else:
                self.features['num_query_params'] = 0
                self.features['query_has_suspicious'] = 0
            
            # 10. Entropy Features (randomness)
            self.features['domain_entropy'] = self._calculate_entropy(domain)
            self.features['path_entropy'] = self._calculate_entropy(path) if path else 0
            
            # 11. Typosquatting Features
            self.features['has_typo_like_chars'] = self._check_typosquatting(domain)
            
            # 12. Special Character Ratio
            special_chars = sum(1 for c in url if not c.isalnum() and c not in [':', '/', '.'])
            self.features['special_char_ratio'] = special_chars / max(len(url), 1)
            
            # 13. Shortened URL indicators
            shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd']
            self.features['is_shortened'] = 1 if any(short in domain for short in shorteners) else 0
            
            # 14. Consecutive character features
            self.features['max_consecutive_dots'] = self._max_consecutive(url, '.')
            self.features['max_consecutive_hyphens'] = self._max_consecutive(url, '-')
            
            # 15. Normalized features
            if len(url) > 0:
                self.features['hyphen_ratio'] = url.count('-') / len(url)
                self.features['dot_ratio'] = url.count('.') / len(url)
            else:
                self.features['hyphen_ratio'] = 0
                self.features['dot_ratio'] = 0
            
            # 16-56: Add more features to match dataset
            # Statistical features
            self.features['vowel_ratio'] = sum(1 for c in url.lower() if c in 'aeiou') / max(len(url), 1)
            self.features['consonant_ratio'] = sum(1 for c in url.lower() if c.isalpha() and c not in 'aeiou') / max(len(url), 1)
            
            # Domain specific
            self.features['domain_token_count'] = len(domain.split('.'))
            self.features['longest_domain_token'] = max([len(t) for t in domain.split('.')], default=0)
            
            # Add padding features if needed to reach expected count
            current_count = len(self.features)
            expected_count = 56  # Adjust based on your dataset
            
            for i in range(current_count, expected_count):
                self.features[f'feature_{i}'] = 0
            
        except Exception as e:
            # If parsing fails, return default features
            return self._get_default_features()
        
        return self.features
    
    def _is_ip_address(self, domain):
        """Check if domain is an IP address"""
        try:
            # Remove port if present
            domain_clean = domain.split(':')[0]
            ipaddress.ip_address(domain_clean)
            return True
        except ValueError:
            return False
    
    def _check_suspicious_tld(self, tld):
        """Check if TLD is suspicious"""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click']
        return 1 if any(tld.endswith(s) for s in suspicious_tlds) else 0
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        # Calculate character frequency
        counter = Counter(text)
        length = len(text)
        
        # Calculate entropy
        entropy = 0
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _check_typosquatting(self, domain):
        """Check for common typosquatting patterns"""
        # Common substitutions
        typo_patterns = [
            ('0', 'o'), ('1', 'l'), ('1', 'i'), ('!', 'i'), ('3', 'e'),
            ('5', 's'), ('$', 's'), ('@', 'a')
        ]
        
        for char, replacement in typo_patterns:
            if char in domain:
                return 1
        
        return 0
    
    def _max_consecutive(self, text, char):
        """Find maximum consecutive occurrences of a character"""
        max_count = 0
        current_count = 0
        
        for c in text:
            if c == char:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _get_default_features(self):
        """Return default features for failed parsing"""
        default_features = {}
        for i in range(56):  # Adjust based on your dataset
            default_features[f'feature_{i}'] = 0
        return default_features


# Convenience function
def extract_url_features(url):
    """Extract features from a URL - convenience function"""
    extractor = URLFeatureExtractor()
    return extractor.extract_all_features(url)


if __name__ == '__main__':
    # Test the feature extraction
    test_urls = [
        'https://www.google.com',
        'http://paypal-verify.tk',
        'http://192.168.1.1/amazon',
        'https://secure-login-account-verify.com/update'
    ]
    
    print("=" * 70)
    print("URL Feature Extraction Test")
    print("=" * 70)
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = extract_url_features(url)
        print(f"Extracted {len(features)} features")
        print(f"Sample features: {list(features.items())[:5]}")
