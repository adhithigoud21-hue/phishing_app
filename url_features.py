import re
from urllib.parse import urlparse
import socket
import dns.resolver
from collections import Counter
import math

def is_valid_domain(domain):
    """Check if domain has valid format"""
    if not domain:
        return False
    
    # Check for valid domain pattern
    domain_pattern = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    )
    return bool(domain_pattern.match(domain))

def check_dns_exists(domain):
    """Check if domain has DNS records (actually exists)"""
    try:
        # Try to resolve domain
        dns.resolver.resolve(domain, 'A')
        return True
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers, dns.exception.Timeout):
        return False
    except Exception:
        return False

def is_reachable(domain):
    """Check if domain is reachable"""
    try:
        socket.setdefaulttimeout(2)
        socket.gethostbyname(domain)
        return True
    except (socket.gaierror, socket.timeout):
        return False
    except Exception:
        return False

def extract_features(url):
    """Extract comprehensive features from URL for phishing detection"""
    
    features = {}
    validation_errors = []
    
    try:
        # Basic validation
        if not url or len(url) < 10:
            validation_errors.append("URL too short - not a valid URL")
            # Return high-risk features for invalid URLs
            return create_high_risk_features(), validation_errors
        
        # Check for malformed URL patterns
        if url.count('://') > 1:
            validation_errors.append("Multiple protocol definitions detected")
        
        if url.count('http') > 1:
            validation_errors.append("Multiple HTTP protocols in URL")
        
        # Check for common typos
        if 'wwww.' in url.lower():
            validation_errors.append("Invalid 'wwww' detected - typo in URL")
        
        if '.htp' in url.lower() or 'htp://' in url.lower():
            validation_errors.append("Malformed HTTP protocol detected")
        
        # Fix URL for parsing
        url_cleaned = url.strip()
        if not url_cleaned.startswith('http'):
            url_cleaned = 'http://' + url_cleaned
        
        # Parse URL
        parsed = urlparse(url_cleaned)
        domain = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query
        
        # CRITICAL: Check if domain exists
        if domain:
            # Check domain format
            if not is_valid_domain(domain):
                validation_errors.append(f"Invalid domain format: '{domain}'")
            
            # Check if it's just a name (no TLD)
            if '.' not in domain:
                validation_errors.append(f"Not a valid domain - missing TLD (e.g., .com, .org): '{domain}'")
            
            # Check DNS existence
            dns_exists = check_dns_exists(domain)
            if not dns_exists:
                validation_errors.append(f"Domain does not exist or is not reachable: '{domain}'")
            
            # Check if domain is just random text
            tld_pattern = re.compile(r'\.(com|org|net|edu|gov|mil|int|co|uk|us|ca|de|fr|jp|cn|in|au|br|ru)$', re.IGNORECASE)
            if not tld_pattern.search(domain):
                # Check for any valid TLD
                if not re.search(r'\.[a-z]{2,}$', domain, re.IGNORECASE):
                    validation_errors.append(f"No valid top-level domain found: '{domain}'")
        else:
            validation_errors.append("No domain found in URL")
        
        # If critical errors, return high-risk features immediately
        if validation_errors:
            return create_high_risk_features(), validation_errors
        
        # === BASIC FEATURES ===
        
        # 1. URL Length
        features['url_length'] = len(url)
        
        # 2. Domain Length
        features['domain_length'] = len(domain) if domain else 0
        
        # 3. Has HTTPS
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        
        # 4. Has IP Address
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        features['has_ip'] = 1 if ip_pattern.match(domain) else 0
        if features['has_ip']:
            validation_errors.append("IP address used instead of domain name")
        
        # === CHARACTER COUNT FEATURES ===
        
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equal'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_percent'] = url.count('%')
        
        # Special characters count
        special_chars = ['!', '#', '$', '%', '^', '*', '+', '=', '[', ']', 
                        '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '~']
        features['num_special_chars'] = sum([url.count(char) for char in special_chars])
        
        # === SUSPICIOUS PATTERNS ===
        
        # Suspicious keywords
        suspicious_words = ['login', 'signin', 'bank', 'account', 'update', 'secure', 
                           'verify', 'confirm', 'password', 'paypal', 'ebay', 'amazon',
                           'microsoft', 'apple', 'google', 'netflix', 'facebook',
                           'suspended', 'locked', 'urgent', 'alert', 'warning']
        features['has_suspicious_words'] = 1 if any(word in url.lower() for word in suspicious_words) else 0
        
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_params'] = len(query.split('&')) if query else 0
        features['path_length'] = len(path)
        features['num_subdirectories'] = path.count('/') if path else 0
        features['has_port'] = 1 if parsed.port else 0
        features['domain_has_numbers'] = 1 if any(c.isdigit() for c in domain) else 0
        
        # === ADVANCED FEATURES ===
        
        features['domain_has_hyphens'] = 1 if '-' in domain else 0
        
        subdomain_count = len(domain.split('.')) - 2 if domain else 0
        features['num_subdomains'] = max(0, subdomain_count)
        
        # Suspicious TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click', '.link']
        features['has_suspicious_tld'] = 1 if any(url.lower().endswith(tld) for tld in suspicious_tlds) else 0
        
        if features['num_at'] > 0:
            validation_errors.append("@ symbol detected - can hide real domain")
        
        features['double_slash_path'] = 1 if '//' in path else 0
        features['prefix_suffix'] = 1 if '-' in domain else 0
        
        # Shortening service
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']
        features['is_shortened'] = 1 if any(short in domain.lower() for short in shorteners) else 0
        
        features['has_www'] = 1 if 'www.' in domain.lower() else 0
        features['domain_url_ratio'] = len(domain) / len(url) if len(url) > 0 else 0
        
        # Entropy
        if url:
            counter = Counter(url)
            length = len(url)
            entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
            features['entropy'] = round(entropy, 4)
        else:
            features['entropy'] = 0
        
        features['consecutive_dots'] = 1 if '..' in url else 0
        if features['consecutive_dots']:
            validation_errors.append("Consecutive dots detected in URL")
        
        # Digit to letter ratio
        if domain:
            digits = sum(c.isdigit() for c in domain)
            letters = sum(c.isalpha() for c in domain)
            features['digit_letter_ratio'] = digits / letters if letters > 0 else digits
        else:
            features['digit_letter_ratio'] = 0
        
        # Check for known legitimate domains
        known_legitimate = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix', 
                           'youtube', 'twitter', 'instagram', 'linkedin', 'github', 'reddit',
                           'wikipedia', 'wordpress', 'adobe', 'stackoverflow', 'paypal']
        
        # Extract base domain
        domain_parts = domain.split('.')
        base_domain = domain_parts[-2] if len(domain_parts) >= 2 else domain
        
        features['is_known_legitimate'] = 1 if base_domain in known_legitimate else 0
        
        # Domain age indicator (length of domain name - shorter often more legitimate)
        features['domain_name_length'] = len(base_domain)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        validation_errors.append(f"Error parsing URL: {str(e)}")
        return create_high_risk_features(), validation_errors
    
    return features, validation_errors

def create_high_risk_features():
    """Return features indicating high phishing risk for invalid URLs"""
    return {
        'url_length': 100,
        'domain_length': 50,
        'has_https': 0,
        'has_ip': 1,
        'num_dots': 10,
        'num_hyphens': 5,
        'num_underscores': 3,
        'num_slashes': 5,
        'num_question_marks': 2,
        'num_equal': 2,
        'num_at': 1,
        'num_ampersand': 2,
        'num_percent': 2,
        'num_special_chars': 15,
        'has_suspicious_words': 1,
        'num_digits': 10,
        'num_params': 5,
        'path_length': 30,
        'num_subdirectories': 3,
        'has_port': 1,
        'domain_has_numbers': 1,
        'domain_has_hyphens': 1,
        'num_subdomains': 5,
        'has_suspicious_tld': 1,
        'double_slash_path': 1,
        'prefix_suffix': 1,
        'is_shortened': 0,
        'has_www': 0,
        'domain_url_ratio': 0.2,
        'entropy': 4.5,
        'consecutive_dots': 1,
        'digit_letter_ratio': 0.5,
        'is_known_legitimate': 0,
        'domain_name_length': 20
    }