"""
neuromorphic/lif_neuron.py
LIF Neuron: Leaky Integrate-and-Fire — the core neuromorphic building block.

V_mem[t] = V_mem[t-1] * (1 - dt/tau) + I[t]
if V_mem[t] >= V_threshold: fire spike=1, reset V_mem=0
"""
import numpy as np
import matplotlib.pyplot as plt

class ManualLIFNeuron:
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, dt=1.0):
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt = dt
        self.v_mem = 0.0
        self.v_history = []
        self.spike_history = []

    def step(self, input_current):
        leak_factor = 1.0 - (self.dt / self.tau)
        self.v_mem = self.v_mem * leak_factor + input_current
        if self.v_mem >= self.v_threshold:
            spike = 1
            self.v_mem = self.v_reset
        else:
            spike = 0
        self.v_history.append(self.v_mem)
        self.spike_history.append(spike)
        return spike

    def simulate(self, input_currents):
        return [self.step(c) for c in input_currents]

    def plot_dynamics(self, title="LIF Neuron Dynamics"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        T = len(self.v_history)
        ax1.plot(range(T), self.v_history, color='steelblue', linewidth=1.5)
        ax1.axhline(y=self.v_threshold, color='red', linestyle='--', label=f'Threshold={self.v_threshold}')
        ax1.set_ylabel('Membrane Potential'); ax1.legend(); ax1.set_title(title)
        spike_times = [t for t, s in enumerate(self.spike_history) if s == 1]
        ax2.vlines(spike_times, 0, 1, color='darkorange', linewidth=1.5)
        ax2.set_ylabel('Spike'); ax2.set_xlabel('Time Step'); ax2.set_ylim(-0.1, 1.3)
        plt.tight_layout()
        return fig

def demo_lif():
    os.makedirs("results", exist_ok=True)
    neuron = ManualLIFNeuron(tau=5.0, v_threshold=1.0)
    T = 100
    np.random.seed(42)
    currents = np.zeros(T)
    currents[10:20] = 0.4; currents[35:50] = 0.25; currents[70:85] = 0.5
    currents += np.random.normal(0, 0.05, T)
    spikes = neuron.simulate(currents.tolist())
    print(f"Total spikes: {sum(spikes)} / {T} steps")
    fig = neuron.plot_dynamics()
    fig.savefig("results/lif_demo.png", dpi=120, bbox_inches='tight')
    print("Saved results/lif_demo.png")

if __name__ == "__main__":
    import os
    demo_lif()
