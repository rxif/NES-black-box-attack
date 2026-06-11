"""
nes_attack.py
=============
NES (Natural Evolution Strategies) Black-Box Adversarial Attack
for MNIST, CIFAR-10, and ImageNet.

Attack type : L-inf bounded, NES gradient estimation + sign update
Reference   : Ilyas et al., 2018 — arXiv:1804.08598

Usage
-----
  python nes_attack.py --dataset mnist   --solver lion   --targeted
  python nes_attack.py --dataset cifar10 --solver adam   --samples 10
  python nes_attack.py --dataset imagenet --solver momentum --samples 10
"""

import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim

# ── Import model definitions ──────────────────────────────────────────────────
from models.setup_cifar10_model import CIFAR10
from models.setup_mnist_model   import MNIST

# ── Import NES optimizers ─────────────────────────────────────────────────────
from optimizers import make_optimizer


# ImageNette synset folders mapped to true ImageNet class IDs
IMAGENETTE_TO_IMAGENET = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}
IMAGENETTE_LABEL_IDS = list(IMAGENETTE_TO_IMAGENET.values())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / model factory
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_and_model(dataset_name, device, imagenet_dir=None):
    """Return (test_loader, model, num_labels, image_size)."""

    transform_gray = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),   # → [-0.5, 0.5]
    ])
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),   # → [-0.5, 0.5]
    ])

    if dataset_name == 'mnist':
        test_set = datasets.MNIST(root='./data', train=False,
                                  transform=transform_gray, download=True)
        model = MNIST().to(device)
        model.load_state_dict(torch.load(
            os.path.join('models', 'mnist_model.pt'),
            map_location=device, weights_only=False))
        num_labels = 10
        label_names = [str(i) for i in range(10)]

    elif dataset_name == 'cifar10':
        test_set = datasets.CIFAR10(root='./data', train=False,
                                    transform=transform_rgb, download=True)
        model = CIFAR10().to(device)
        model.load_state_dict(torch.load(
            os.path.join('models', 'cifar10_model.pt'),
            map_location=device, weights_only=False))
        num_labels = 10
        label_names = ['plane','car','bird','cat','deer',
                       'dog','frog','horse','ship','truck']

    elif dataset_name == 'imagenet':
        from torchvision import models
        from torchvision.datasets import ImageFolder

        val_dir = os.path.join('data', 'imagenette2-320', 'val')
        if not os.path.isdir(val_dir):
            import urllib.request, tarfile
            os.makedirs('data', exist_ok=True)
            archive = os.path.join('data', 'imagenette2-320.tgz')
            if not os.path.exists(archive):
                print('ImageNette not found, downloading (~100 MB)...')
                urllib.request.urlretrieve(
                    'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
                    archive,
                    reporthook=lambda b, bs, t: print(
                        '\r  %.1f/%.1f MB' % (b*bs/1e6, t/1e6), end='', flush=True))
                print()
            print('Extracting...')
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall('data')
            os.remove(archive)
            print('ImageNette ready.')

        img_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])

        class _ImageNetteDataset(ImageFolder):
            """ImageFolder with true ImageNet class IDs from synset folder names."""
            def __getitem__(self, idx):
                img, folder_idx = super().__getitem__(idx)
                synset = self.classes[folder_idx]
                label  = IMAGENETTE_TO_IMAGENET.get(synset, 0)
                return img, label

        test_set = _ImageNetteDataset(val_dir, transform=img_transform)

        # Use torchvision InceptionV3 (same as attacks.py conceptually)
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.aux_logits = False

        class _InceptionWrapper(torch.nn.Module):
            """Accepts [-0.5,0.5] input, applies ImageNet normalisation inside."""
            _MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            _STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            def __init__(self, base): super().__init__(); self.base = base
            def forward(self, x):
                x = x + 0.5                                  # → [0, 1]
                mean = self._MEAN.to(x.device, x.dtype)
                std  = self._STD.to(x.device, x.dtype)
                x = (x - mean) / std
                return self.base(x)

        model = _InceptionWrapper(inception).to(device)
        num_labels = 1000
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        label_names = weights.meta['categories']

    else:
        raise ValueError('Unknown dataset: %s' % dataset_name)

    model.eval()
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    return loader, model, num_labels, label_names


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection  (mirrors generate_data in zoo_l2_attack_black.py)
# ─────────────────────────────────────────────────────────────────────────────

def select_one_per_label(data_arr, label_arr, required_labels, start=0):
    """
    Select one sample per required label, preserving `required_labels` order.
    Returns (selected_data, selected_labels, missing_labels).
    """
    first_idx = {}
    req = set(required_labels)
    start_idx = max(int(start), 0)

    for i in range(start_idx, len(label_arr)):
        lbl = int(label_arr[i])
        if lbl in req and lbl not in first_idx:
            first_idx[lbl] = i
            if len(first_idx) == len(required_labels):
                break

    chosen_labels = [lbl for lbl in required_labels if lbl in first_idx]
    missing = [lbl for lbl in required_labels if lbl not in first_idx]

    if not chosen_labels:
        return np.empty((0, *data_arr.shape[1:]), dtype=data_arr.dtype), np.empty((0,), dtype=label_arr.dtype), missing

    idxs = [first_idx[lbl] for lbl in chosen_labels]
    return data_arr[idxs], label_arr[idxs], missing

def generate_data(loader, targeted, samples, start, num_labels, targeted_k=None,
                  target_label_pool=None):
    """
    Collect `samples` correctly-classified images starting after index `start`.
    Targeted  → returns one (image, one-hot target) per selected non-true class.
                If targeted_k is None, uses all non-true classes (original behavior).
    Untargeted → returns (image, one-hot true-class).
    """
    inputs, targets = [], []
    cnt = 0
    for i, (data, label) in enumerate(loader):
        if cnt >= samples:
            break
        if i <= start:
            continue
        x   = data[0].numpy()          # (C, H, W)
        lbl = int(label.item())
        if targeted:
            if target_label_pool is None:
                candidate_targets = [j for j in range(num_labels) if j != lbl]
            else:
                candidate_targets = [j for j in target_label_pool if j != lbl]

            if not candidate_targets:
                cnt += 1
                continue

            if targeted_k is None:
                selected_targets = candidate_targets
            else:
                k = max(1, min(int(targeted_k), len(candidate_targets)))
                selected_targets = np.random.choice(candidate_targets, size=k, replace=False).tolist()

            for j in selected_targets:
                inputs.append(x)
                targets.append(np.eye(num_labels)[j])
        else:
            inputs.append(x)
            targets.append(np.eye(num_labels)[lbl])
        cnt += 1

    return np.array(inputs), np.array(targets)


# ─────────────────────────────────────────────────────────────────────────────
# NES gradient estimation
# ─────────────────────────────────────────────────────────────────────────────

def nes_grad_estimate(model, x, label_idx, sigma, n_samples, targeted, device):
    """
    Estimate the gradient of the loss w.r.t. x using antithetic NES sampling.

    x          : (1, C, H, W) torch.Tensor on `device`
    label_idx  : int  — true class (untargeted) or target class (targeted)
    sigma      : float — noise standard deviation
    n_samples  : int  — total queries (must be even; uses n_samples/2 pairs)
    targeted   : bool
    Returns    : (grad tensor same shape as x, mean scalar loss)
    """
    n_half = n_samples // 2
    shape  = x.shape                        # (1, C, H, W)
    d      = x[0].numel()                  # C*H*W

    # Sample n_half noise vectors
    noise = torch.randn(n_half, d, device=device)   # (n_half, d)
    noise_4d = noise.view(n_half, *shape[1:])        # (n_half, C, H, W)

    x_pos = x + sigma * noise_4d           # (n_half, C, H, W)
    x_neg = x - sigma * noise_4d

    label_t = torch.full((n_half,), label_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_pos = model(x_pos)           # (n_half, num_classes)
        logits_neg = model(x_neg)

    def margin_loss(logits):
        """
        C&W margin loss — always non-zero while the attack hasn't succeeded.
        Untargeted : logit[true]  - max_{j≠true}  logit[j]  (minimise → flip)
        Targeted   : logit[other] - logit[target]             (minimise → reach target)
        """
        n = logits.shape[0]
        if targeted:
            # want logit[target] to be the max → minimise (max_other - logit[target])
            target_logit = logits[:, label_idx]                        # (n,)
            mask = torch.ones(n, logits.shape[1], dtype=torch.bool, device=device)
            mask[:, label_idx] = False
            other_max = logits[mask].view(n, -1).max(dim=1)[0]        # (n,)
            return other_max - target_logit
        else:
            # want logit[true] to stop being the max → minimise (logit[true] - max_other)
            true_logit = logits[:, label_idx]                          # (n,)
            mask = torch.ones(n, logits.shape[1], dtype=torch.bool, device=device)
            mask[:, label_idx] = False
            other_max = logits[mask].view(n, -1).max(dim=1)[0]        # (n,)
            return true_logit - other_max

    loss_pos = margin_loss(logits_pos)   # (n_half,)
    loss_neg = margin_loss(logits_neg)

    diff = (loss_pos - loss_neg).view(n_half, 1, 1, 1)
    grad = (diff * noise_4d).mean(dim=0, keepdim=True) / sigma   # (1,C,H,W)

    mean_loss = ((loss_pos + loss_neg) / 2).mean().item()
    return grad, mean_loss


# ─────────────────────────────────────────────────────────────────────────────
# Single-image NES attack
# ─────────────────────────────────────────────────────────────────────────────

def nes_attack_one(x_np, target_np, model, targeted, solver,
                   device, epsilon, sigma, n_samples,
                   max_iter, max_lr, min_lr, plateau_length, plateau_drop):
    """
    Run NES attack on a single image. Stops as soon as the attack succeeds.

    x_np      : (C, H, W) numpy array in normalised space [-0.5, 0.5]
    target_np : (num_classes,) one-hot numpy array
    Returns   : (adv_np, success bool, distortion float, loss history list,
                 queries int)
    """
    x0    = torch.from_numpy(x_np).unsqueeze(0).to(device)   # (1,C,H,W)
    adv   = x0.clone()
    lower = (x0 - epsilon).clamp(-0.5, 0.5)
    upper = (x0 + epsilon).clamp(-0.5, 0.5)

    label_idx = int(np.argmax(target_np))
    orig_idx  = int(model(x0).argmax(dim=1).item())

    opt = make_optimizer(solver, momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8)

    max_lr_cur = max_lr
    last_ls    = []
    loss_hist  = []
    queries    = 0

    for step in range(max_iter):
        raw_g, loss = nes_grad_estimate(
            model, adv, label_idx, sigma, n_samples, targeted, device)
        queries += n_samples

        g = torch.from_numpy(opt.update(raw_g.cpu().numpy())).to(device)

        loss_hist.append(loss)
        last_ls.append(loss)
        last_ls = last_ls[-plateau_length:]
        if len(last_ls) == plateau_length and last_ls[-1] > last_ls[0]:
            if max_lr_cur > min_lr:
                max_lr_cur = max(max_lr_cur / plateau_drop, min_lr)
                last_ls = []

        proposed = adv - max_lr_cur * torch.sign(g)
        proposed = proposed.clamp(lower, upper)
        adv      = proposed

        if step % 20 == 0:
            print('  step %04d | loss %.4f | lr %.2e' % (step, loss, max_lr_cur))

        with torch.no_grad():
            pred = int(model(adv).argmax(dim=1).item())
        success = (pred == label_idx) if targeted else (pred != orig_idx)
        if success:
            print('  Early stop at step %d (%d queries)' % (step + 1, queries))
            adv_np = adv.squeeze(0).cpu().numpy()
            dist   = float(np.max(np.abs(adv_np - x_np)))
            return adv_np, True, dist, loss_hist, queries

    # Final prediction
    with torch.no_grad():
        logits = model(adv)
    pred = int(logits.argmax(dim=1).item())

    if targeted:
        success = (pred == label_idx)
    else:
        success = (pred != orig_idx)

    adv_np = adv.squeeze(0).cpu().numpy()
    dist   = float(np.max(np.abs(adv_np - x_np)))

    return adv_np, success, dist, loss_hist, queries


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_uint8(x_np):
    """(C,H,W) in [-0.5,0.5] → (H,W,C) uint8 in [0,255]."""
    x = np.clip(x_np.transpose(1, 2, 0) + 0.5, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

def save_image(arr_uint8, path):
    if arr_uint8.shape[2] == 1:
        Image.fromarray(arr_uint8[:, :, 0], mode='L').save(path)
    else:
        Image.fromarray(arr_uint8, mode='RGB').save(path)

def compute_metrics(orig_np, adv_np):
    """orig/adv are (C,H,W) in [-0.5,0.5]. Returns dict of MSE,MAE,PSNR,SSIM."""
    o = np.clip(orig_np.transpose(1,2,0)+0.5, 0, 1)
    a = np.clip(adv_np.transpose(1,2,0) +0.5, 0, 1)
    mse  = float(np.sum((o-a)**2))
    mae  = float(np.sum(np.abs(o-a)))
    psnr = float(calc_psnr(o, a, data_range=1.0))
    if o.shape[2] == 1:
        ssim = float(calc_ssim(o[:,:,0], a[:,:,0], data_range=1.0))
    else:
        ssim = float(calc_ssim(o, a, data_range=1.0, channel_axis=2))
    return {'mse': mse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}


def choose_device_with_cuda_probe():
    """Pick CUDA only if a minimal GPU forward pass succeeds."""
    if not torch.cuda.is_available():
        return torch.device('cpu'), 'CUDA not available, using CPU'

    probe = (
        "import torch\n"
        "x=torch.randn(1,1,28,28,device='cuda')\n"
        "m=torch.nn.Conv2d(1,8,3).cuda()\n"
        "with torch.no_grad():\n"
        "    _=m(x)\n"
        "torch.cuda.synchronize()\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device('cuda'), 'Using GPU: %s' % gpu_name

    return torch.device('cpu'), (
        'CUDA detected but self-test failed (likely driver/WSL/CUDA runtime mismatch). '
        'Falling back to CPU to avoid Bus error.'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NES Black-Box Adversarial Attack')
    parser.add_argument('--dataset',  choices=['mnist','cifar10','imagenet'],
                        default='cifar10')
    parser.add_argument('--solver',   choices=['momentum','nesterov','adagrad',
                                               'adam','sgd','sgdsign','signum',
                                               'lion','newton','adahessian'],
                        default='momentum')
    parser.add_argument('--targeted', action='store_true',
                        help='Targeted attack (default: untargeted)')
    parser.add_argument('--targeted-k', type=int, default=None,
                        help='Number of non-true target classes per source image in targeted mode '
                             '(default: all non-true classes)')
    parser.add_argument('--imagenette-one-per-class', action='store_true',
                        help='For ImageNet: use exactly one correctly-classified sample from each '
                            'ImageNette class (10 total sources).')
    parser.add_argument('--target-label-set', choices=['all', 'imagenette10'], default='all',
                        help='Target class pool for targeted attacks (default: all classes).')
    parser.add_argument('--samples',  type=int, default=10)
    parser.add_argument('--start',    type=int, default=6,
                        help='Offset into test set (same as mate)')
    parser.add_argument('--imagenet_dir', default='./mini_imagenet')
    # Attack hyperparams (None = auto per dataset)
    parser.add_argument('--epsilon',       type=float, default=None)
    parser.add_argument('--sigma',         type=float, default=None)
    parser.add_argument('--n_samples',     type=int,   default=None)
    parser.add_argument('--max_iter',      type=int,   default=None)
    parser.add_argument('--max_lr',        type=float, default=None)
    parser.add_argument('--min_lr',        type=float, default=None)
    parser.add_argument('--plateau_length',type=int,   default=5)
    parser.add_argument('--plateau_drop',  type=float, default=2.0)
    args = parser.parse_args()

    # Dataset-specific defaults
    DEFAULTS = {
        'mnist':    dict(epsilon=0.3,  sigma=0.05, n_samples=100, max_iter=500, max_lr=0.05,  min_lr=0.001),
        'cifar10':  dict(epsilon=0.05, sigma=0.05, n_samples=100, max_iter=500, max_lr=0.01,  min_lr=0.0005),
        'imagenet': dict(epsilon=0.05, sigma=0.001,n_samples=100,  max_iter=500, max_lr=0.01,  min_lr=0.0005),
    }
    d = DEFAULTS[args.dataset]
    if args.epsilon  is None: args.epsilon  = d['epsilon']
    if args.sigma    is None: args.sigma    = d['sigma']
    if args.n_samples is None:args.n_samples= d['n_samples']
    if args.max_iter is None: args.max_iter = d['max_iter']
    if args.max_lr   is None: args.max_lr   = d['max_lr']
    if args.min_lr   is None: args.min_lr   = d['min_lr']

    np.random.seed(42)
    torch.manual_seed(42)

    device, device_msg = choose_device_with_cuda_probe()
    print(device_msg)

    print('Dataset: %s | Solver: %s | Targeted: %s | Device: %s' % (
        args.dataset, args.solver, args.targeted, device))

    # ── Load dataset + model ─────────────────────────────────────────────────
    loader, model, num_labels, label_names = load_dataset_and_model(
        args.dataset, device, args.imagenet_dir)

    # ── Filter to correctly-classified samples ───────────────────────────────
    print('Checking model accuracy on test set...')
    data_correct, label_correct = [], []
    total = 0
    num_correct = 0
    with torch.no_grad():
        for img, lbl in loader:
            img_dev = img.to(device)
            lbl_int = int(lbl.item())
            pred = int(model(img_dev).argmax(dim=1).item())

            total += 1
            if pred == lbl_int:
                num_correct += 1
                data_correct.append(img[0].numpy())
                label_correct.append(lbl_int)

    acc = float(num_correct) / max(total, 1)
    print('Model accuracy: %.2f%%' % (acc * 100))
    print('Correctly classified: %d / %d' % (num_correct, total))

    data_correct = np.array(data_correct, dtype=np.float32)
    label_correct = np.array(label_correct, dtype=np.int64)

    correct_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data_correct),
            torch.from_numpy(label_correct)),
        batch_size=1, shuffle=False)

    target_label_pool = None
    if args.targeted and args.target_label_set == 'imagenette10':
        target_label_pool = IMAGENETTE_LABEL_IDS

    use_imagenette_one_per_class = args.imagenette_one_per_class or (
        args.dataset == 'imagenet' and not args.targeted
    )

    if use_imagenette_one_per_class:
        selected_data, selected_labels, missing = select_one_per_label(
            data_correct, label_correct, IMAGENETTE_LABEL_IDS, start=args.start)
        if missing:
            raise RuntimeError(
                'Could not find correctly-classified samples for labels: %s' % missing)

        if args.dataset == 'imagenet' and not args.targeted and not args.imagenette_one_per_class:
            print('Auto-enabling class-balanced ImageNette sources for untargeted ImageNet.')
        print('Using class-balanced ImageNette sources (10 classes, 1 sample each).')
        print('Selected labels: %s' % selected_labels.tolist())

        correct_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(selected_data),
                torch.from_numpy(selected_labels)),
            batch_size=1, shuffle=False)

        # We already curated exactly 10 sources, so consume all from the start.
        args.samples = len(selected_data)
        args.start = -1

    # ── Select attack samples ────────────────────────────────────────────────
    inputs, targets = generate_data(correct_loader, args.targeted,
                                    args.samples, args.start, num_labels,
                                    targeted_k=args.targeted_k,
                                    target_label_pool=target_label_pool)
    print('Attack samples selected: %d' % len(inputs))

    # ── Output directory ─────────────────────────────────────────────────────
    targeted_str = 'targeted' if args.targeted else 'untargeted'
    out_dir = os.path.join('nes_results', args.dataset, targeted_str, args.solver)
    os.makedirs(out_dir, exist_ok=True)

    # ── Run attacks ──────────────────────────────────────────────────────────
    adv_list     = []
    success_list = []
    dist_list    = []
    queries_list = []
    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    orig_classes, adv_classes = [], []
    t0 = time.time()
    for i in range(len(inputs)):
        print('\n=== Sample %d / %d ===' % (i+1, len(inputs)))
        adv_np, success, dist, _, queries = nes_attack_one(
            inputs[i], targets[i], model, args.targeted, args.solver, device,
            epsilon        = args.epsilon,
            sigma          = args.sigma,
            n_samples      = args.n_samples,
            max_iter       = args.max_iter,
            max_lr         = args.max_lr,
            min_lr         = args.min_lr,
            plateau_length = args.plateau_length,
            plateau_drop   = args.plateau_drop,
        )
        queries_list.append(queries)

        # Record predictions
        with torch.no_grad():
            oc = int(model(torch.from_numpy(inputs[i]).unsqueeze(0).to(device)
                           ).argmax(dim=1).item())
            ac = int(model(torch.from_numpy(adv_np).unsqueeze(0).to(device)
                           ).argmax(dim=1).item())
        orig_classes.append(oc)
        adv_classes.append(ac)

        # Save images
        orig_uint8 = to_uint8(inputs[i])
        adv_uint8  = to_uint8(adv_np)
        save_image(orig_uint8, os.path.join(out_dir, 'original_%d.png'    % i))
        save_image(adv_uint8,  os.path.join(out_dir, 'adversarial_%d.png' % i))

        # Metrics
        m = compute_metrics(inputs[i], adv_np)
        mse_list.append(m['mse']);  mae_list.append(m['mae'])
        psnr_list.append(m['psnr']); ssim_list.append(m['ssim'])

        adv_list.append(adv_np)
        success_list.append(bool(success))
        dist_list.append(dist)

        status = 'SUCCESS' if success else 'FAIL'
        print('[%s] orig:%s → adv:%s | L-inf dist: %.4f' % (
            status,
            label_names[oc][:12] if isinstance(label_names[oc], str) else str(oc),
            label_names[ac][:12] if isinstance(label_names[ac], str) else str(ac),
            dist))

    elapsed = (time.time() - t0) / 60.0
    success_rate    = 100.0 * sum(success_list) / max(len(success_list), 1)
    total_distortion = float(np.sum(
        [(np.sum((adv_list[i] - inputs[i])**2)**0.5) for i in range(len(inputs))]))
    mean_l2_distortion = total_distortion / max(len(inputs), 1)

    successful_queries = [queries_list[i] for i in range(len(success_list)) if success_list[i]]
    mean_queries = float(np.mean(successful_queries)) if successful_queries else float('nan')

    print('\n' + '='*60)
    print('Solver       : %s' % args.solver)
    print('Mode         : early stop')
    print('Success Rate : %.1f %%' % success_rate)
    print('Queries (avg on success) : %.1f / %d' % (mean_queries, args.max_iter * args.n_samples))
    print('Distortion   : %.4f' % total_distortion)
    print('Time         : %.2f mins for %d samples' % (elapsed, len(inputs)))
    print('='*60)

    # ── Save results.json ────────────────────────────────────────────────────
    results = {
        'dataset':                    args.dataset,
        'targeted':                   args.targeted,
        'solver':                     args.solver,
        'num_samples':                len(inputs),
        'success_rate_pct':           success_rate,
        'total_distortion':           total_distortion,
        'mean_l2_distortion':          mean_l2_distortion,
        'per_sample_distortion_linf': dist_list,
        'distortion_metrics': {
            'total_distortion': 'sum over samples of L2 norm ||adv - orig||_2 in normalized space [-0.5, 0.5]',
            'per_sample_distortion_linf': 'per-sample L-inf norm ||adv - orig||_inf in normalized space [-0.5, 0.5]'
        },
        'time_mins':                  elapsed,
        'early_stop':                 True,
        'queries': {
            'per_sample':             queries_list,
            'mean_on_success':        mean_queries,
            'budget':                 args.max_iter * args.n_samples,
        },
        'valid_classification':       orig_classes,
        'adversarial_classification': adv_classes,
        'mse':  {'mean': float(np.mean(mse_list)),  'per_sample': mse_list},
        'mae':  {'mean': float(np.mean(mae_list)),  'per_sample': mae_list},
        'psnr': {'mean': float(np.mean(psnr_list)), 'per_sample': psnr_list},
        'ssim': {'mean': float(np.mean(ssim_list)), 'per_sample': ssim_list},
        'attack_params': {
            'epsilon':        args.epsilon,
            'sigma':          args.sigma,
            'n_samples':      args.n_samples,
            'max_iter':       args.max_iter,
            'max_lr':         args.max_lr,
            'min_lr':         args.min_lr,
        }
    }
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to %s' % results_path)

    # ── Grid visualisation ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(adv_list)
    fig, axes = plt.subplots(2, n, figsize=(n*2, 5))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        orig_disp = np.clip(inputs[i].transpose(1,2,0)+0.5, 0, 1)
        adv_disp  = np.clip(adv_list[i].transpose(1,2,0)+0.5, 0, 1)
        cmap = 'gray' if orig_disp.shape[2] == 1 else None
        orig_name = label_names[orig_classes[i]]
        adv_name  = label_names[adv_classes[i]]
        if not isinstance(orig_name, str):
            orig_name = str(orig_name)
        if not isinstance(adv_name, str):
            adv_name = str(adv_name)
        axes[0,i].imshow(orig_disp.squeeze(), cmap=cmap)
        axes[0,i].set_xlabel(orig_name, fontsize=10, labelpad=4)
        axes[0,i].set_xticks([])
        axes[0,i].set_yticks([])
        axes[1,i].imshow(adv_disp.squeeze(),  cmap=cmap)
        axes[1,i].set_xlabel(adv_name, fontsize=10, labelpad=4)
        axes[1,i].set_xticks([])
        axes[1,i].set_yticks([])
    axes[0,0].set_ylabel('Original', fontsize=8)
    axes[1,0].set_ylabel('Adversarial', fontsize=8)
    plt.suptitle('NES %s — %s — %s' % (args.dataset, targeted_str, args.solver),
                 fontsize=9)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'grid_%s_%s_%s.png' % (
        args.dataset, targeted_str, args.solver))
    plt.savefig(grid_path, dpi=120)
    print('Grid saved to %s' % grid_path)
