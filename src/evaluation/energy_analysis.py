"""
Computes reconstruction errors, anomaly scores, F1, AUC-ROC,
latency, SynOps vs FLOPs, and energy estimates (nJ/sample).
"""
import time, torch, numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt

def compute_reconstruction_errors(model, loader, device='cpu'):
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=1)
            errors.extend(mse.cpu().numpy())
    return np.array(errors)

def find_optimal_threshold(train_errors, percentile=95.0):
    return float(np.percentile(train_errors, percentile))

def evaluate_detection(errors, y_true, threshold):
    y_pred = (errors > threshold).astype(int)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, errors)
    tp = ((y_pred==1)&(y_true==1)).sum(); fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    precision = tp/(tp+fp+1e-8); recall = tp/(tp+fn+1e-8)
    return {'f1': float(f1), 'auc_roc': float(auc),
            'precision': float(precision), 'recall': float(recall)}

def measure_latency(model, sample, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(sample)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter(); model(sample); times.append((time.perf_counter()-t0)*1000)
    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times))}

def full_comparison(snn, ann, train_loader, test_loader, y_test, device='cpu'):
    print("\n" + "="*55 + "\nENERGY-ACCURACY ANALYSIS\n" + "="*55)
    results = {}
    sample = next(iter(test_loader))[0][:32].to(device)

    for name, model in [('SNN', snn), ('ANN', ann)]:
        print(f"\n── {name} ──")
        model.to(device)
        train_errors = compute_reconstruction_errors(model, train_loader, device)
        threshold    = find_optimal_threshold(train_errors)
        test_errors  = compute_reconstruction_errors(model, test_loader, device)
        det = evaluate_detection(test_errors, y_test, threshold)
        lat = measure_latency(model, sample)
        print(f"  F1: {det['f1']:.4f} | AUC: {det['auc_roc']:.4f}")
        print(f"  Precision: {det['precision']:.4f} | Recall: {det['recall']:.4f}")
        print(f"  Latency: {lat['mean_ms']:.2f}ms")

        if name == 'SNN':
            ops = model.count_synaptic_operations(sample)
            print(f"  SynOps/sample: {ops['total_synops_per_sample']:.1f} | Energy: {ops['estimated_energy_nJ']:.5f} nJ")
            results.update({'snn_energy_nJ': ops['estimated_energy_nJ'], 'snn_synops': ops['total_synops_per_sample']})
        else:
            ops = model.count_multiply_accumulate(sample)
            print(f"  FLOPs/sample: {ops['total_flops_per_sample']:.1f} | Energy: {ops['gpu_energy_nJ']:.5f} nJ")
            results.update({'ann_energy_nJ': ops['gpu_energy_nJ'], 'ann_flops': ops['total_flops_per_sample']})

        results.update({f'{name.lower()}_f1': det['f1'], f'{name.lower()}_auc': det['auc_roc'],
                        f'{name.lower()}_test_errors': test_errors})

    ratio = results.get('ann_energy_nJ', 0) / (results.get('snn_energy_nJ', 1e-10))
    print(f"\n Energy reduction SNN vs ANN: {ratio:.1f}x")
    results['energy_reduction_ratio'] = ratio
    return results

def plot_comparison(results_by_dataset, save_path=None):
    datasets = list(results_by_dataset.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(datasets)); w = 0.35

    for ax, key, title, ylabel in zip(axes,
        [('snn_f1','ann_f1'), ('snn_auc','ann_auc'), ('snn_energy_nJ','ann_energy_nJ')],
        ['F1-Score', 'AUC-ROC', 'Energy (nJ/sample) — log scale'],
        ['F1', 'AUC', 'nJ']):
        snn_vals = [results_by_dataset[d].get(key[0], 0) for d in datasets]
        ann_vals = [results_by_dataset[d].get(key[1], 0) for d in datasets]
        ax.bar(x-w/2, snn_vals, w, label='SNN', color='steelblue', alpha=0.85)
        ax.bar(x+w/2, ann_vals, w, label='ANN', color='darkorange', alpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.set_ylabel(ylabel); ax.legend(fontsize=9)
        if 'nJ' in title: ax.set_yscale('log')

    plt.suptitle('SNN vs ANN: Neuromorphic Energy-Accuracy Tradeoff', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
