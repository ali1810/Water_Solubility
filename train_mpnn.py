"""
train_mpnn.py
=============
Quick-start training script.
Copy this + mpnn_model.py into your Water_Solubility folder and run:

  python train_mpnn.py

Or on Google Colab:
  !python train_mpnn.py --epochs 150 --device cuda

Author: Dr. Mushtaq Ali · KIT
"""
import os, sys, json, time
import numpy as np
import torch

# Add parent dir if needed
sys.path.insert(0, os.path.dirname(__file__))
from mpnn_model import train_mpnn, evaluate_mpnn, MPNNSolubility

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_csv',  default='final_unique_train.csv')
parser.add_argument('--test_csv',   default='final_unique_test.csv')
parser.add_argument('--output_dir', default='models')
parser.add_argument('--epochs',     type=int,   default=100)
parser.add_argument('--batch_size', type=int,   default=32)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--node_dim',   type=int,   default=128)
parser.add_argument('--n_layers',   type=int,   default=4)
parser.add_argument('--hidden_dim', type=int,   default=256)
parser.add_argument('--dropout',    type=float, default=0.1)
parser.add_argument('--patience',   type=int,   default=15)
parser.add_argument('--device',     default=None,
    help='cuda or cpu. Auto-detected if not specified.')
args = parser.parse_args()

print("=" * 60)
print("MPNN Solubility Model — Training")
print("Dr. Mushtaq Ali · KIT")
print("=" * 60)
print(f"Train CSV: {args.train_csv}")
print(f"Test CSV:  {args.test_csv}")
print(f"Output:    {args.output_dir}")
print()

t0 = time.time()

history = train_mpnn(
    train_csv   = args.train_csv,
    test_csv    = args.test_csv,
    output_dir  = args.output_dir,
    node_dim    = args.node_dim,
    n_layers    = args.n_layers,
    hidden_dim  = args.hidden_dim,
    dropout     = args.dropout,
    lr          = args.lr,
    batch_size  = args.batch_size,
    epochs      = args.epochs,
    patience    = args.patience,
    device      = args.device,
)

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed/60:.1f} minutes")

# Final evaluation
best_model = os.path.join(args.output_dir, 'mpnn_solubility_best.pt')
if os.path.exists(best_model) and os.path.exists(args.test_csv):
    print("\nFinal evaluation on test set:")
    metrics = evaluate_mpnn(best_model, args.test_csv, args.device or 'cpu')

    print("\nQuick prediction test:")
    model = MPNNSolubility.load(best_model)
    test_smiles = [
        ('CCO',                    'Ethanol',   'Expected: ~0.9'),
        ('c1ccccc1',               'Benzene',   'Expected: ~-1.6'),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin',   'Expected: ~-2.1'),
        ('Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'Caffeine', 'Expected: ~-1.2'),
    ]
    for smi, name, expected in test_smiles:
        pred = model.predict_smiles(smi)
        print(f"  {name:<12} logS={pred:>6.3f}  {expected}")
