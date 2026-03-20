"""
Main training script. Trains SNN + ANN autoencoders, evaluates, saves results.
Usage: python train.py --dataset thyroid --epochs 50 --device cuda
"""
import os, sys, argparse, time, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from src.models.snn_autoencoder import SNNAutoencoder
from src.models.ann_autoencoder import ANNAutoencoder
from src.utils.data_loader import load_odds_dataset, make_train_test_split, make_dataloaders
from src.evaluation.energy_analysis import full_comparison, plot_comparison

def train_model(model, loader, epochs, lr, device, name):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*10,
          steps_per_epoch=len(loader), epochs=epochs, pct_start=0.3)
    crit = nn.MSELoss()
    losses = []
    print(f"\nTraining {name} for {epochs} epochs...")
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); total = 0
        for (batch,) in loader:
            batch = batch.to(device); opt.zero_grad()
            loss = crit(model(batch), batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); total += loss.item()*batch.size(0)
        ep_loss = total/len(loader.dataset); losses.append(ep_loss)
        if ep%10==0 or ep==1:
            print(f"  Ep {ep:3d}/{epochs} | Loss: {ep_loss:.6f} | {time.time()-t0:.1f}s")
    return losses

def run_experiment(ds_name, epochs=50, lr=1e-3, batch_size=64, T=8, device='cpu'):
    print(f"\n{'='*55}\nEXPERIMENT: {ds_name.upper()}\n{'='*55}")
    os.makedirs(f"results/{ds_name}", exist_ok=True)
    X, y = load_odds_dataset(ds_name)
    X_train, X_test, y_test = make_train_test_split(X, y)
    train_loader, test_loader = make_dataloaders(X_train, X_test, batch_size)
    input_dim = X.shape[1]

    snn = SNNAutoencoder(input_dim, hidden_dim=64, latent_dim=16, T=T).to(device)
    ann = ANNAutoencoder(input_dim, hidden_dim=64, latent_dim=16).to(device)
    print(f"SNN params: {sum(p.numel() for p in snn.parameters()):,} | ANN params: {sum(p.numel() for p in ann.parameters()):,}")

    snn_losses = train_model(snn, train_loader, epochs, lr, device, "SNN")
    ann_losses = train_model(ann, train_loader, epochs, lr, device, "ANN")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(snn_losses, label='SNN', color='steelblue')
    ax.plot(ann_losses, label='ANN', color='darkorange')
    ax.set(xlabel='Epoch', ylabel='MSE Loss', title=f'Training Loss — {ds_name.capitalize()}')
    ax.legend(); ax.set_yscale('log'); plt.tight_layout()
    plt.savefig(f"results/{ds_name}/training_loss.png", dpi=120); plt.close()

    torch.save(snn.state_dict(), f"results/{ds_name}/snn_checkpoint.pt")
    torch.save(ann.state_dict(), f"results/{ds_name}/ann_checkpoint.pt")

    return full_comparison(snn, ann, train_loader, test_loader, y_test, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='thyroid', choices=['thyroid','arrhythmia','cardio','all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    elif torch.cuda.is_available():
        args.device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    datasets = ['thyroid','arrhythmia','cardio'] if args.dataset=='all' else [args.dataset]
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_experiment(ds, args.epochs, args.lr, args.batch_size, args.T, args.device)

    if len(all_results) > 1:
        os.makedirs("results", exist_ok=True)
        plot_comparison(all_results, "results/comparison_figure.png")
        print("\n── SUMMARY ──")
        print(f"{'Dataset':<14} {'SNN F1':>8} {'ANN F1':>8} {'SNN AUC':>8} {'ANN AUC':>8} {'Energy↓':>10}")
        for ds, r in all_results.items():
            print(f"{ds:<14} {r['snn_f1']:>8.4f} {r['ann_f1']:>8.4f} {r['snn_auc']:>8.4f} {r['ann_auc']:>8.4f} {r.get('energy_reduction_ratio',0):>9.1f}x")
