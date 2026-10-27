import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import markov_chain_diagnostics

this_directory = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(this_directory, "..", "..", "output", "optimal_sampling_strategies_figs",
                                        "fig8"))
N_values = sorted([3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
variants = ['b0', 'b_opt']
num_jobs = 12

results = {
    'b0': [],
    'b_opt': []
}

for variant in variants:
    for N in N_values:
        mixing_times_events = []
        for i in range(num_jobs):
            job_dir = os.path.join(base_dir, f"{variant}_temp_1_N_{N}", "temperature_00", f"job_{i:02d}")
            path_variance = os.path.join(job_dir, "checkpoint_00_sample_of_half_system_distance_variance.npy")
            path_json = os.path.join(job_dir, "sim_params.json")
            
            if os.path.exists(path_variance) and os.path.exists(path_json):
                data = np.load(path_variance).flatten()
                mt, eq_val = markov_chain_diagnostics.get_mixing_time(data)
                
                # Physical MT
                physical_mt = mt * 0.001 * N
                
                # Read event rate
                with open(path_json, 'r') as f:
                    sim_params = json.load(f)
                mean_event_rate = sim_params.get("mean_event_rate", 0.0)
                
                events_mt = physical_mt * mean_event_rate
                mixing_times_events.append(events_mt)
            else:
                pass
                
        if mixing_times_events:
            mean_mt_events = np.mean(mixing_times_events)
            results[variant].append(mean_mt_events)
        else:
            results[variant].append(np.nan)

N_arr = np.array(N_values)
b0_mt = np.array(results['b0'])
b_opt_mt = np.array(results['b_opt'])

output_npy = os.path.join(base_dir, "mixing_time_events_results.npy")
np.save(output_npy, np.vstack((N_arr, b0_mt, b_opt_mt)))
print(f"Saved results to {output_npy}")

plt.figure(figsize=(10, 6))
plt.loglog(N_arr, b0_mt, 'o-', color='blue', label='b0 (Events)')
plt.loglog(N_arr, b_opt_mt, 's-', color='orange', label='b_opt (Events)')

plt.xlabel('N')
plt.ylabel('Mean Events to Reach Equilibrium')
plt.title('Mixing Time (in Events) vs N')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

output_png = os.path.join(base_dir, "mixing_time_events_plot.png")
plt.savefig(output_png, dpi=300)
print(f"Saved plot to {output_png}")
