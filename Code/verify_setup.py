#!/usr/bin/env python3
"""
Complete Verification & Setup Script
Checks all project files and dependencies
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
from datetime import datetime

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'
BOLD = '\033[1m'


def check_file(filepath: str, description: str = "") -> bool:
    """Check if file exists"""
    path = Path(filepath)
    status = "✓" if path.exists() else "✗"
    color = GREEN if path.exists() else RED
    
    size = f" ({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
    print(f"  {color}{status}{END} {description:40s} {filepath}{size}")
    return path.exists()


def check_python_package(package_name: str, import_name: str = None) -> bool:
    """Check if Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        version = importlib.import_module(import_name).__version__ if hasattr(importlib.import_module(import_name), '__version__') else "unknown"
        print(f"  {GREEN}✓{END} {package_name:20s} ({version})")
        return True
    except ImportError:
        print(f"  {RED}✗{END} {package_name:20s} NOT INSTALLED")
        return False


def main():
    """Run complete verification"""
    
    print(f"\n{BOLD}{BLUE}{'='*70}{END}")
    print(f"{BOLD}{BLUE}EPITOPE ESCAPE MUTATION PREDICTION - SETUP VERIFICATION{END}")
    print(f"{BOLD}{BLUE}{'='*70}{END}\n")
    
    print(f"{BOLD}Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{END}\n")
    
    # ============= CHECK PROJECT FILES =============
    print(f"{BOLD}1. Project Files{END}")
    print(f"{'-'*70}")
    
    required_files = [
        ('train.py', 'Main training script'),
        ('data_loader.py', 'Data loading & graph construction'),
        ('model.py', 'EGNN model & trainer'),
        ('inference.py', 'Inference & evaluation metrics'),
        ('esm_utils.py', 'ESM-C API integration'),
        ('requirements.txt', 'Python dependencies'),
        ('README.md', 'Full documentation'),
        ('QUICKSTART.md', 'Quick start guide'),
    ]
    
    files_ok = sum([check_file(f, desc) for f, desc in required_files])
    print(f"\n{GREEN}✓{END} {files_ok}/{len(required_files)} project files present\n")
    
    # ============= CHECK DATA FILES =============
    print(f"{BOLD}2. Input Data Files{END}")
    print(f"{'-'*70}")
    
    data_files = [
        ('sphere_radius_mapped_dataset.csv', 'Sphere-strain mapping data'),
        ('comprehensive_pdb_parsed_metrics.csv', 'PDB structural metrics'),
    ]
    
    data_ok = 0
    for f, desc in data_files:
        exists = check_file(f, desc)
        data_ok += exists
    
    if data_ok == 0:
        print(f"\n{YELLOW}⚠ WARNING: Data files not found in current directory{END}")
        print(f"   Copy CSV files here or provide path when running train.py\n")
    else:
        print(f"\n{GREEN}✓{END} {data_ok}/{len(data_files)} data files present\n")
    
    # ============= CHECK PYTHON PACKAGES =============
    print(f"{BOLD}3. Python Dependencies{END}")
    print(f"{'-'*70}")
    
    required_packages = [
        ('torch', 'torch'),
        ('torch-geometric', 'torch_geometric'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('requests', 'requests'),
        ('tqdm', 'tqdm'),
    ]
    
    packages_ok = sum([check_python_package(pkg, imp) for pkg, imp in required_packages])
    print(f"\n{GREEN}✓{END} {packages_ok}/{len(required_packages)} core packages installed")
    
    if packages_ok < len(required_packages):
        print(f"\n{YELLOW}⚠ Some packages missing. Install with:{END}")
        print(f"  pip install -r requirements.txt\n")
    else:
        print()
    
    # ============= CHECK CUDA =============
    print(f"{BOLD}4. GPU / CUDA Support{END}")
    print(f"{'-'*70}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {GREEN}✓{END} CUDA Available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"    Version: {torch.version.cuda}")
        else:
            print(f"  {YELLOW}⚠{END} CUDA Not Available (will use CPU - slower)")
    except Exception as e:
        print(f"  {YELLOW}⚠{END} Could not check CUDA: {e}")
    print()
    
    # ============= CHECK FILE CONTENTS =============
    print(f"{BOLD}5. File Structure Validation{END}")
    print(f"{'-'*70}")
    
    # Check train.py structure
    try:
        with open('train.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('ESMCClient import', 'from esm_utils import' in content),
            ('ESMEmbeddingManager class', 'class ESMEmbeddingManager' in content),
            ('TrainingPipeline class', 'class TrainingPipeline' in content),
            ('ArgumentParser setup', 'argparse.ArgumentParser' in content),
            ('ESM API integration', 'esm_api_key' in content or 'esm_token' in content),
        ]
        
        for desc, passed in checks:
            status = "✓" if passed else "✗"
            color = GREEN if passed else RED
            print(f"  {color}{status}{END} {desc}")
    except:
        print(f"  {RED}✗{END} Could not validate train.py")
    
    print()
    
    # Check model.py structure
    try:
        with open('model.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('StructuralMessagePassing class', 'class StructuralMessagePassing' in content),
            ('RegionScorer model', 'class RegionScorer' in content),
            ('WeightedBCELoss function', 'class WeightedBCELoss' in content),
            ('EpitopeTrainer class', 'class EpitopeTrainer' in content),
            ('Learnable threshold', 'self.threshold = nn.Parameter' in content),
        ]
        
        for desc, passed in checks:
            status = "✓" if passed else "✗"
            color = GREEN if passed else RED
            print(f"  {color}{status}{END} {desc}")
    except:
        print(f"  {RED}✗{END} Could not validate model.py")
    
    print()
    
    # Check data_loader.py structure
    try:
        with open('data_loader.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('EpitopeGraphDataset class', 'class EpitopeGraphDataset' in content),
            ('Train/val/test splitting', 'similarity_tier' in content),
            ('Graph construction', 'create_edges' in content),
            ('dMaSIF features', 'dMaSIF' in content or 'surface_normals' in content),
        ]
        
        for desc, passed in checks:
            status = "✓" if passed else "✗"
            color = GREEN if passed else RED
            print(f"  {color}{status}{END} {desc}")
    except:
        print(f"  {RED}✗{END} Could not validate data_loader.py")
    
    print()
    
    # Check inference.py structure
    try:
        with open('inference.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('EpitopeEvaluator class', 'class EpitopeEvaluator' in content),
            ('Comprehensive metrics', 'matthews_corrcoef' in content and 'roc_auc_score' in content),
            ('Per-PDB analysis', '_compute_per_pdb_metrics' in content),
            ('Report generation', 'generate_report' in content),
        ]
        
        for desc, passed in checks:
            status = "✓" if passed else "✗"
            color = GREEN if passed else RED
            print(f"  {color}{status}{END} {desc}")
    except:
        print(f"  {RED}✗{END} Could not validate inference.py")
    
    print()
    
    # ============= API KEY STATUS =============
    print(f"{BOLD}6. ESM-C API Configuration{END}")
    print(f"{'-'*70}")
    
    api_key_provided = os.environ.get('ESM_API_KEY') or os.environ.get('ESMC_API_KEY')
    
    if api_key_provided:
        masked_key = api_key_provided[:10] + '...' + api_key_provided[-5:]
        print(f"  {GREEN}✓{END} API Key found in environment: {masked_key}")
        print(f"\n  Usage: python train.py ... --esm_token $ESM_API_KEY\n")
    else:
        print(f"  {YELLOW}⚠{END} API Key not set in environment")
        print(f"\n  Set with: export ESM_API_KEY='1j2QjkEE9UGqCaBtJ0QIOJ'\n")
        print(f"  Then use: python train.py ... --esm_token $ESM_API_KEY\n")
    
    # ============= SUMMARY =============
    print(f"{BOLD}{BLUE}{'='*70}{END}")
    print(f"{BOLD}SUMMARY{END}")
    print(f"{BOLD}{BLUE}{'='*70}{END}\n")
    
    all_ok = files_ok == len(required_files) and packages_ok == len(required_packages)
    
    if all_ok:
        print(f"{GREEN}{BOLD}✓ READY TO TRAIN{END}\n")
        print(f"Next steps:")
        print(f"  1. Set API key: export ESM_API_KEY='1j2QjkEE9UGqCaBtJ0QIOJ'")
        print(f"  2. Run training:")
        print(f"     python train.py \\")
        print(f"       --sphere_data sphere_radius_mapped_dataset.csv \\")
        print(f"       --metrics_data comprehensive_pdb_parsed_metrics.csv \\")
        print(f"       --esm_token $ESM_API_KEY\n")
    else:
        print(f"{YELLOW}{BOLD}⚠ SETUP INCOMPLETE{END}\n")
        
        if files_ok < len(required_files):
            print(f"  Missing project files - ensure all .py files are in current directory")
        
        if packages_ok < len(required_packages):
            print(f"  Install missing packages: pip install -r requirements.txt")
        
        if data_ok == 0:
            print(f"  Add CSV data files to current directory or provide paths to train.py")
        
        print()
    
    print(f"{BOLD}{BLUE}{'='*70}{END}\n")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
