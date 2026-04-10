"""
Download real ICD-9-CM code descriptions from the CMS (Centers for Medicare & Medicaid Services)
official repository and build a local lookup dictionary.

Source: CMS ICD-9-CM Diagnosis Codes (Version 32, final version)
"""

import os
import json
import urllib.request
import zipfile
import io
import re


def download_and_parse_icd9_descriptions():
    """
    Download CMS ICD-9-CM short descriptions and return a code->description dict.
    Uses the official CMS data from their archived ICD-9-CM page.
    """
    
    output_path = os.path.join(os.path.dirname(__file__), 'icd9_descriptions.json')
    
    # If we already have the file, just load it
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        print(f"Loaded {len(descriptions)} ICD-9 descriptions from cache.")
        return descriptions
    
    print("Downloading CMS ICD-9-CM diagnosis code descriptions...")
    
    # CMS official ICD-9-CM V32 (final version) short description file
    url = "https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        zip_data = response.read()
    except Exception as e:
        print(f"Primary CMS URL failed: {e}")
        # Fallback: try an alternate well-known source
        url2 = "https://data.nber.org/icd9/icd9_short_dx.csv"
        print(f"Trying NBER fallback: {url2}")
        try:
            req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
            response2 = urllib.request.urlopen(req2, timeout=30)
            csv_text = response2.read().decode('latin-1')
            descriptions = parse_nber_csv(csv_text)
            save_descriptions(descriptions, output_path)
            return descriptions
        except Exception as e2:
            print(f"NBER fallback also failed: {e2}")
            print("Trying manual flat-file approach...")
            return download_flat_file(output_path)
    
    # Parse the ZIP file
    descriptions = {}
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # List files in the zip
        print(f"  ZIP contents: {zf.namelist()}")
        for name in zf.namelist():
            if 'diag' in name.lower() and ('short' in name.lower() or 'desc' in name.lower()):
                print(f"  Parsing: {name}")
                with zf.open(name) as f:
                    content = f.read().decode('latin-1')
                    descriptions.update(parse_cms_file(content))
                break
        
        # If we couldn't find a specific file, try all text files
        if not descriptions:
            for name in zf.namelist():
                if name.lower().endswith('.txt') and 'diag' in name.lower():
                    print(f"  Parsing fallback: {name}")
                    with zf.open(name) as f:
                        content = f.read().decode('latin-1')
                        descriptions.update(parse_cms_file(content))
                    if descriptions:
                        break
    
    if not descriptions:
        # Try parsing ALL files in the zip
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for name in zf.namelist():
                if name.lower().endswith('.txt'):
                    print(f"  Attempting to parse: {name}")
                    with zf.open(name) as f:
                        content = f.read().decode('latin-1')
                        parsed = parse_cms_file(content)
                        if len(parsed) > 100:
                            descriptions.update(parsed)
                            print(f"    -> Found {len(parsed)} codes")
                            break
    
    save_descriptions(descriptions, output_path)
    return descriptions


def parse_cms_file(content):
    """Parse CMS fixed-width or tab-separated diagnosis code file."""
    descriptions = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try tab-separated first
        if '\t' in line:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                code, desc = parts[0].strip(), parts[1].strip()
                if code and desc:
                    # Format code with dot (e.g., "4019" -> "401.9")
                    formatted = format_icd9_code(code)
                    descriptions[formatted] = desc
        else:
            # Fixed-width: first ~6 chars are code, rest is description
            # CMS format: code is typically 5-7 chars padded with spaces
            match = re.match(r'^(\S+)\s+(.+)$', line)
            if match:
                code = match.group(1).strip()
                desc = match.group(2).strip()
                if code and desc and len(code) <= 7:
                    formatted = format_icd9_code(code)
                    descriptions[formatted] = desc
    
    return descriptions


def parse_nber_csv(content):
    """Parse NBER CSV format."""
    descriptions = {}
    for line in content.strip().split('\n')[1:]:  # Skip header
        parts = line.split(',', 1)
        if len(parts) == 2:
            code = parts[0].strip().strip('"')
            desc = parts[1].strip().strip('"')
            formatted = format_icd9_code(code)
            descriptions[formatted] = desc
    return descriptions


def format_icd9_code(raw_code):
    """
    Convert raw ICD-9 code to dotted format.
    E.g., '4019' -> '401.9', '25000' -> '250.00', 'V5861' -> 'V58.61', 'E8788' -> 'E878.8'
    """
    code = raw_code.strip().upper()
    
    # Already has a dot
    if '.' in code:
        return code
    
    # E-codes (external cause)
    if code.startswith('E') and len(code) > 4:
        return code[:4] + '.' + code[4:]
    
    # V-codes (supplementary)
    if code.startswith('V') and len(code) > 3:
        return code[:3] + '.' + code[3:]
    
    # Regular codes: dot after 3rd digit
    if len(code) > 3:
        return code[:3] + '.' + code[3:]
    
    return code


def download_flat_file(output_path):
    """
    Last resort: download from a known working flat-file URL.
    """
    urls = [
        "https://raw.githubusercontent.com/jackwasey/icd/refs/heads/main/data-raw/icd9cm2014.txt",
    ]
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            content = response.read().decode('latin-1')
            descriptions = parse_cms_file(content)
            if descriptions:
                save_descriptions(descriptions, output_path)
                return descriptions
        except Exception as e:
            print(f"  Failed {url}: {e}")
    
    print("All download sources failed. Creating empty description map.")
    save_descriptions({}, output_path)
    return {}


def save_descriptions(descriptions, output_path):
    """Save the descriptions dict to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(descriptions)} ICD-9 descriptions to {os.path.basename(output_path)}")


if __name__ == '__main__':
    descriptions = download_and_parse_icd9_descriptions()
    
    # Test some codes from our model output
    test_codes = ['427.31', '414.01', '250.00', '401.9', '584.9', '410.11']
    print("\nSample lookups:")
    for code in test_codes:
        desc = descriptions.get(code, 'NOT FOUND')
        print(f"  {code}: {desc}")
    
    print(f"\nTotal descriptions available: {len(descriptions)}")
