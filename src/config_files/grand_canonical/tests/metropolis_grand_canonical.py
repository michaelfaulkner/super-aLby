"""Independent grand-canonical Metropolis reference for disks on a ring (shares no code with super-aLby).

    python metropolis_grand_canonical.py <soft|hard> <ring length> <log_activity> <temperature> <sweeps> <seed>
        <initial_number> <output.json>

Textbook grand-canonical Monte Carlo of labelled particles on a ring of length L (Frenkel & Smit, ch. 5.6): each trial
move is a displacement (probability 1/2), an insertion (1/4) or a deletion (1/4).
    displacement: a random particle moves by a uniform amount in [-delta, delta], accepted with min(1, exp(-beta dU));
    insertion:    a particle at a uniform position, accepted with min(1, z L / (N + 1) exp(-beta dU));
    deletion:     a random particle, accepted with min(1, N / (z L) exp(+beta dU)).
Soft disks: u = (1 / r)^12; hard disks: diameter 1.  One sweep is 20 trial moves.  N and U are recorded once per sweep
after a burn-in of 10% of the sweeps.  The output has the mean and variance of N, the mean of U and the probabilities
P(N).
"""
import json
import sys
import numpy as np


def main():
    kind, length, log_activity, temperature, sweeps, seed, initial_number, output = sys.argv[1:9]
    length = float(length)
    beta, activity_length = 1.0 / float(temperature), np.exp(float(log_activity)) * length
    sweeps, rng = int(sweeps), np.random.default_rng(int(seed))

    def pair_energy(distance):
        if kind == "soft":
            return distance ** -12.0
        return np.where(distance < 1.0, np.inf, 0.0)

    def energy_of(position, others):
        if len(others) == 0:
            return 0.0
        separation = others - position
        separation -= length * np.round(separation / length)
        return float(np.sum(pair_energy(np.abs(separation))))

    """Initial configuration: initial_number evenly spaced sites."""
    x = (np.arange(int(initial_number)) + 0.5) * length / int(initial_number)
    total_energy = sum(energy_of(x[i], x[i + 1:]) for i in range(len(x)))
    delta = 0.3 if kind == "soft" else 0.2
    numbers, energies = [], []
    for sweep in range(sweeps):
        for _ in range(20):
            choice = rng.uniform()
            n = len(x)
            if choice < 0.5:
                if n == 0:
                    continue
                i = rng.integers(n)
                others = np.delete(x, i)
                trial = np.mod(x[i] + rng.uniform(-delta, delta), length)
                difference = energy_of(trial, others) - energy_of(x[i], others)
                if difference <= 0.0 or rng.uniform() < np.exp(- beta * difference):
                    x[i] = trial
                    total_energy += difference
            elif choice < 0.75:
                trial = rng.uniform(0.0, length)
                difference = energy_of(trial, x)
                if np.isfinite(difference) and rng.uniform() < activity_length / (n + 1) * np.exp(- beta * difference):
                    x = np.append(x, trial)
                    total_energy += difference
            else:
                if n == 0:
                    continue
                i = rng.integers(n)
                others = np.delete(x, i)
                difference = - energy_of(x[i], others)
                if rng.uniform() < n / activity_length * np.exp(- beta * difference):
                    x = others
                    total_energy += difference
        if sweep >= sweeps // 10:
            numbers.append(len(x))
            energies.append(total_energy)
    numbers, energies = np.array(numbers), np.array(energies)
    with open(output, "w") as f:
        json.dump({"kind": kind, "length": length, "log_activity": float(log_activity),
                   "temperature": float(temperature), "sweeps": sweeps, "mean_number": float(np.mean(numbers)),
                   "variance_number": float(np.var(numbers)), "mean_energy": float(np.mean(energies)),
                   "distribution": (np.bincount(numbers) / len(numbers)).tolist()}, f)


if __name__ == "__main__":
    main()
