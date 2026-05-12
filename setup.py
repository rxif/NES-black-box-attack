"""
setup.py
========
Downloads all datasets and pre-trained models required for NES attack.

Usage:
    python setup.py
"""

import os
import urllib.request
import tarfile
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def progress(b, bs, total):
    mb = b * bs / 1e6
    total_mb = total / 1e6
    pct = min(mb / total_mb * 100, 100) if total > 0 else 0
    print('\r  %.1f / %.1f MB (%.0f%%)' % (mb, total_mb, pct),
          end='', flush=True)

def section(title):
    print('\n' + '='*50)
    print('  ' + title)
    print('='*50)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MNIST
# ─────────────────────────────────────────────────────────────────────────────

section('1/4  MNIST (~11 MB)')
datasets.MNIST(root='./data', train=True,  download=True,
               transform=transforms.ToTensor())
datasets.MNIST(root='./data', train=False, download=True,
               transform=transforms.ToTensor())
print('  MNIST OK')

# ─────────────────────────────────────────────────────────────────────────────
# 2. CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

section('2/4  CIFAR-10 (~170 MB)')
datasets.CIFAR10(root='./data', train=True,  download=True,
                 transform=transforms.ToTensor())
datasets.CIFAR10(root='./data', train=False, download=True,
                 transform=transforms.ToTensor())
# Remove archive after extraction to save disk space
for f in ['cifar-10-python.tar.gz', 'cifar-10-batches-py.tar.gz']:
    path = os.path.join('data', f)
    if os.path.exists(path):
        os.remove(path)
print('  CIFAR-10 OK')

# ─────────────────────────────────────────────────────────────────────────────
# 3. ImageNette
# ─────────────────────────────────────────────────────────────────────────────

section('3/4  ImageNette (~100 MB)')
ARCHIVE = os.path.join('data', 'imagenette2-320.tgz')
OUT_DIR = os.path.join('data', 'imagenette2-320')
os.makedirs('data', exist_ok=True)

if not os.path.exists(OUT_DIR):
    if not os.path.exists(ARCHIVE):
        print('  Downloading...')
        urllib.request.urlretrieve(
            'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
            ARCHIVE, reporthook=progress)
        print()
    print('  Extracting...')
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        tar.extractall('data')
    # Remove archive after extraction to save disk space
    if os.path.exists(ARCHIVE):
        os.remove(ARCHIVE)
else:
    print('  Already extracted, skipping.')
print('  ImageNette OK')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-trained models
# ─────────────────────────────────────────────────────────────────────────────

section('4/4  Pre-trained models')

if os.path.exists(os.path.join('models', 'mnist_model.pt')):
    print('  mnist_model.pt    OK')
else:
    print('  mnist_model.pt    MISSING — should be committed in models/')

if os.path.exists(os.path.join('models', 'cifar10_model.pt')):
    print('  cifar10_model.pt  OK')
else:
    print('  cifar10_model.pt  MISSING — should be committed in models/')

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*50)
print('  Setup complete!')
print('  You can now run:')
print()
print('  python nes_attack.py --dataset mnist    --solver lion --samples 10')
print('  python nes_attack.py --dataset cifar10  --solver lion --samples 10')
print('  python nes_attack.py --dataset imagenet --solver lion --samples 10')
print('='*50 + '\n')
