"""Unit tests of the grand-canonical ECMC components (no full simulations).  Run from src:

    python config_files/grand_canonical/tests/unit_tests.py config_files/grand_canonical/tests/unit_settings.ini

model_settings reads the ring length from the config at import time.  Prints PASS/FAIL per check; exit code 1 on any
failure.
"""
import json
import os
import sys
import tempfile
import numpy as np

CONFIG = sys.argv[1]
sys.argv = ["run.py", CONFIG]
sys.path.insert(0, os.getcwd())

from base.exceptions import ConfigurationError  # noqa: E402
from model_settings import size_of_particle_space  # noqa: E402
from potential.grand_canonical_hard_sphere_potential import GrandCanonicalHardSpherePotential  # noqa: E402
from potential.grand_canonical_soft_sphere_potential import GrandCanonicalSoftSpherePotential  # noqa: E402
from ghost_coupling_bias.tabulated_ghost_coupling_bias import TabulatedGhostCouplingBias  # noqa: E402
from grand_canonical_kernel.tempered_grand_canonical_kernel import TemperedGrandCanonicalKernel  # noqa: E402
from grand_canonical_kernel.direct_grand_canonical_kernel import DirectGrandCanonicalKernel  # noqa: E402
from grand_canonical_kernel.fixed_ghost_coupling_kernel import FixedGhostCouplingKernel  # noqa: E402
from mediator.event_chain_mediator import EventChainMediator  # noqa: E402

LENGTH = float(np.atleast_1d(size_of_particle_space)[0])
RNG = np.random.default_rng(12345)
np.random.seed(54321)
FAILURES = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)
    if not condition:
        FAILURES.append(name)


def minimum_image(separation):
    return separation - LENGTH * np.round(separation / LENGTH)


def random_configuration(number, minimum_distance, ghost_clearance=None):
    """Random sequential addition of number real particles with pairwise distance >= minimum_distance, plus (if
    ghost_clearance is not None) a ghost row at distance >= ghost_clearance from every real particle.  Returns an
    array of shape (rows, 1)."""
    positions = []
    while len(positions) < number:
        candidate = RNG.uniform(0.0, LENGTH)
        if all(abs(minimum_image(candidate - p)) >= minimum_distance for p in positions):
            positions.append(candidate)
    if ghost_clearance is not None:
        while True:
            candidate = RNG.uniform(0.0, LENGTH)
            if all(abs(minimum_image(candidate - p)) >= ghost_clearance for p in positions):
                positions.append(candidate)
                break
    return np.array(positions)[:, None]


def contact_distances(positions, active, coupling, diameter):
    """Contact distance of every pair (active, j): diameter for real-real pairs, coupling * diameter for ghost pairs
    (coupling > 0), 0 (no interaction) otherwise."""
    rows = len(positions)
    contacts = np.full(rows, diameter)
    if coupling is not None:
        ghost = rows - 1
        ghost_contact = max(coupling, 0.0) * diameter
        if active == ghost:
            contacts[:] = ghost_contact
        else:
            contacts[ghost] = ghost_contact
    contacts[active] = 0.0
    return contacts


def test_hard_sphere_events():
    """Event distance against a brute-force march: no overlap before the event, contact at the event, and overlap with
    the vetoing particle just after it."""
    potential = GrandCanonicalHardSpherePotential(diameter=1.0)
    failures = 0
    for trial in range(150):
        coupling = [None, -0.3, 0.0, 0.4, 0.9][trial % 5]
        potential.ghost_coupling = coupling
        positions = random_configuration(6, 1.0, None if coupling is None else max(coupling, 0.0))
        active = int(RNG.integers(len(positions)))
        distance, veto, _ = potential.get_next_event(positions, active, 1.0, 1)
        contacts = contact_distances(positions, active, coupling, 1.0)

        def gaps(shift):
            return np.abs(minimum_image(positions[:, 0] - positions[active, 0] - shift)) - contacts

        steps = np.linspace(0.0, distance, 4001)[:-1]
        interacting = contacts > 0.0
        before_ok = all(np.all(gaps(s)[interacting] >= -1e-9) for s in steps)
        if veto != active:
            at_contact = abs(gaps(distance)[veto]) < 1e-8
            overlap_after = gaps(distance + 1e-6)[veto] < 0.0 or contacts[veto] == 0.0
        else:
            at_contact = overlap_after = not np.any(interacting)
        if not (before_ok and at_contact and overlap_after):
            failures += 1
    check("hard-disk event distances vs brute-force march (150 random configurations, ghost couplings "
          "None/-0.3/0/0.4/0.9)", failures == 0, f"{failures} failures")


def test_soft_sphere_event_distribution():
    """Empirical survival of the event distance against exp(- beta * sum_j int [d u_j / ds]^+ ds) computed on a fine
    grid along one ring length (minimum image)."""
    temperature = 0.7
    potential = GrandCanonicalSoftSpherePotential(epsilon=1.0, sigma=1.0, power=12.0)
    for coupling in [None, 0.6]:
        potential.ghost_coupling = coupling
        positions = random_configuration(6, 0.85, None if coupling is None else 0.5)
        for active in ([0] if coupling is None else [0, len(positions) - 1]):
            samples = np.array([potential.get_next_event(positions, active, temperature, 1)[0]
                                for _ in range(20000)])
            scales = np.ones(len(positions))
            if coupling is not None:
                ghost = len(positions) - 1
                if active == ghost:
                    scales[:] = coupling
                else:
                    scales[ghost] = coupling
            scales[active] = 0.0
            grid = np.linspace(0.0, LENGTH, 200001)
            rate_integral = np.zeros(len(grid))
            for j in range(len(positions)):
                if scales[j] == 0.0:
                    continue
                path = np.abs(minimum_image(positions[j, 0] - positions[active, 0] - grid))
                energy = (scales[j] / path) ** 12
                rate_integral[1:] += np.cumsum(np.maximum(np.diff(energy), 0.0))
            theory = np.exp(- rate_integral / temperature)
            test_points = np.linspace(0.0, 0.98 * LENGTH, 60)
            empirical = np.array([np.mean(samples > s) for s in test_points])
            predicted = np.interp(test_points, grid, theory)
            deviation = float(np.max(np.abs(empirical - predicted)))
            label = "real" if coupling is None or active != len(positions) - 1 else "ghost"
            check(f"soft-disk event-distance distribution (active {label}, ghost coupling {coupling})",
                  deviation < 0.02, f"max |S_empirical - S_exact| = {deviation:.4f}")


def test_ghost_functions():
    """Ghost energy, force and coupling-event inversion against direct pair sums and finite differences."""
    soft = GrandCanonicalSoftSpherePotential(epsilon=1.3, sigma=0.9, power=12.0)
    hard = GrandCanonicalHardSpherePotential(diameter=1.0)
    worst_energy, worst_force, worst_event, no_shrinking_events = 0.0, 0.0, 0.0, True
    for trial in range(50):
        positions = random_configuration(6, 0.9, 0.3)
        soft.ghost_coupling = 0.5
        profile = soft.get_ghost_profile(positions)
        for coupling in [0.2, 0.5, 0.8, 1.0]:
            soft.ghost_coupling = coupling
            direct = soft.get_single_particle_energy(positions, len(positions) - 1)
            worst_energy = max(worst_energy, abs(direct - float(soft.get_ghost_energy(coupling, profile))) /
                               max(abs(direct), 1e-300))
            step = 1e-6
            finite_difference = (float(soft.get_ghost_energy(coupling + step, profile)) -
                                 float(soft.get_ghost_energy(coupling - step, profile))) / (2 * step)
            worst_force = max(worst_force, abs(finite_difference - soft.get_ghost_force(coupling)) /
                              max(abs(finite_difference), 1e-12))
            """Same seed twice: the exponential drawn inside get_next_coupling_event is reproduced as sample."""
            temperature, seed = 1.0 / 1.7, int(RNG.integers(2 ** 31))
            np.random.seed(seed)
            event = coupling + soft.get_next_coupling_event(coupling, temperature, 1)
            no_shrinking_events &= bool(np.isinf(soft.get_next_coupling_event(coupling, temperature, -1)))
            np.random.seed(seed)
            sample = -np.log(np.random.uniform(0.0, 1.0))
            worst_event = max(worst_event, abs((float(soft.get_ghost_energy(event, profile)) -
                                                float(soft.get_ghost_energy(coupling, profile))) / temperature - sample))
        hard.ghost_coupling = 0.2
        hard.get_ghost_profile(positions)
        clearance = hard.get_ghost_wall()
        brute = np.min(np.abs(minimum_image(positions[:-1, 0] - positions[-1, 0])))
        check_hard = abs(clearance - brute) < 1e-12 and float(hard.get_ghost_energy(clearance * 0.999, clearance)) == 0.0 \
            and np.isinf(float(hard.get_ghost_energy(clearance * 1.001, clearance)))
        if not check_hard:
            check("hard ghost wall = clearance", False)
            return
    check("soft ghost energy lambda^n A = direct pair sum", worst_energy < 1e-10, f"max relative error {worst_energy:.1e}")
    check("soft ghost force = finite difference", worst_force < 1e-5, f"max relative error {worst_force:.1e}")
    check("soft ghost coupling-event inversion beta (U(l*) - U(l)) = E", worst_event < 1e-9,
          f"max error {worst_event:.1e}")
    check("soft ghost (repulsive) has no coupling events while shrinking", no_shrinking_events)
    check("hard ghost wall = clearance, energy 0 below and infinite above", True)


def test_bias():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"nodes": [0.0, 0.3, 0.5, 1.0], "slopes": [1.0, -2.0, 4.0]}, f)
        name = f.name
    bias = TabulatedGhostCouplingBias(file_name=name)
    os.remove(name)
    expected = {0.0: 0.0, 0.3: 0.3, 0.5: -0.1, 1.0: 1.9, -0.5: 0.0, 0.4: 0.1}
    values_ok = all(abs(bias.get_value(l) - v) < 1e-12 for l, v in expected.items())
    segments_ok = (bias.get_segment(0.0, 1) == (0.3, 1.0) and bias.get_segment(0.3, 1) == (0.5, -2.0) and
                   bias.get_segment(0.31, 1) == (0.5, -2.0) and bias.get_segment(0.5, 1) == (1.0, 4.0) and
                   bias.get_segment(0.3, -1) == (0.0, 1.0) and bias.get_segment(0.5, -1) == (0.3, -2.0) and
                   bias.get_segment(1.0, -1) == (0.5, 4.0) and bias.get_segment(0.45, -1) == (0.3, -2.0))
    check("bias values eta(lambda) (incl. eta = 0 on the parked interval)", values_ok)
    check("bias intervals at and between nodes, both directions", segments_ok)
    check("bias total = eta(1)", abs(bias.total - 1.9) < 1e-12)


def test_jump_bookkeeping():
    """Deletions and insertions keep the identity (position) of the event-chain label, and the ghost is the last row."""
    potential = GrandCanonicalHardSpherePotential(diameter=1.0)
    ok_delete, ok_insert, ok_direct = True, True, True
    for trial in range(300):
        kernel = TemperedGrandCanonicalKernel(log_activity=-30.0)
        kernel.initialize(potential, 1.0)
        positions = random_configuration(6, 1.0, 0.0)
        potential.ghost_coupling = -1.0
        active = int(RNG.integers(len(positions)))
        if active == len(positions) - 1:
            new_positions, new_active = kernel._attempt_deletion(positions.copy(), active)
            ok_delete &= len(new_positions) == len(positions) and new_active == active
            continue
        old_active_position = positions[active].copy()
        new_positions, new_active = kernel._attempt_deletion(positions.copy(), active)
        ok_delete &= len(new_positions) == len(positions) - 1
        ok_delete &= bool(np.allclose(new_positions[new_active], old_active_position))
        """The new ghost (last row) is one of the old real particles, and the multiset of positions is preserved."""
        old_set = sorted(map(tuple, np.round(positions[:-1], 12)))
        new_set = sorted(map(tuple, np.round(new_positions, 12)))
        ok_delete &= old_set == new_set and potential.ghost_coupling == 1.0
        """Insertion (forced by a huge activity): the ghost becomes real, a new ghost is appended, label kept."""
        kernel = TemperedGrandCanonicalKernel(log_activity=30.0)
        kernel.initialize(potential, 1.0)
        potential.ghost_coupling = 1.0
        inserted = kernel._attempt_insertion(positions.copy())
        ok_insert &= (len(inserted) == len(positions) + 1 and np.allclose(inserted[:-1], positions) and
                      potential.ghost_coupling == - 1.0)
        """Direct deletion (forced): the active particle keeps its position."""
        direct = DirectGrandCanonicalKernel(log_activity=-30.0)
        direct.initialize(potential, 1.0)
        reals = positions[:-1].copy()
        direct_active = int(RNG.integers(len(reals)))
        np.random.seed(trial)
        for _ in range(20):
            result = direct.handle_event(reals.copy(), direct_active, 1)
            if len(result[0]) == len(reals) - 1:
                ok_direct &= bool(np.allclose(result[0][result[1]], reals[direct_active]))
                break
    check("tempered deletion keeps the label's particle, ghost = an old real particle, N - 1", ok_delete)
    check("tempered insertion: ghost becomes real, new ghost appended, label kept", ok_insert)
    check("direct deletion keeps the label's particle", ok_direct)


def test_configuration_errors():
    raised = []
    for build in [lambda: TemperedGrandCanonicalKernel().initialize(object(), 1.0),
                  lambda: TemperedGrandCanonicalKernel(parked_length=-1.0),
                  lambda: TemperedGrandCanonicalKernel(normalised_distance_between_coupling_moves=0.0),
                  lambda: DirectGrandCanonicalKernel(normalised_distance_between_direct_moves=-1.0),
                  lambda: FixedGhostCouplingKernel(ghost_coupling=1.5)]:
        try:
            build()
            raised.append(False)
        except ConfigurationError:
            raised.append(True)
    check("invalid kernels / potentials raise ConfigurationError", all(raised), str(raised))
    mediator = EventChainMediator(GrandCanonicalSoftSpherePotential(), [], grand_canonical_kernel=
                                  TemperedGrandCanonicalKernel(), output_directory=tempfile.mkdtemp())
    try:
        mediator._reload_configuration_from_file_and_reset()
        check("restart with a grand-canonical kernel raises", False)
    except ConfigurationError:
        check("restart with a grand-canonical kernel raises", True)


if __name__ == "__main__":
    test_bias()
    test_ghost_functions()
    test_jump_bookkeeping()
    test_configuration_errors()
    test_hard_sphere_events()
    test_soft_sphere_event_distribution()
    print('ALL UNIT TESTS PASSED' if not FAILURES else 'UNIT TEST FAILURES: ' + ', '.join(FAILURES))
    sys.exit(1 if FAILURES else 0)
