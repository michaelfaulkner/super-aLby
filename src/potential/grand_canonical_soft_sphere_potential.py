"""Module for the GrandCanonicalSoftSpherePotential class."""
from base.exceptions import ConfigurationError
from .grand_canonical_potential import GrandCanonicalPotential
from model_settings import range_of_initial_particle_positions, size_of_particle_space
import numpy as np


class GrandCanonicalSoftSpherePotential(GrandCanonicalPotential):
    r"""
    Class for soft disks on a ring, U(r) = epsilon * (sigma / r) ** power, for use with EventChainMediator or
    MetropolisMediator and a grand-canonical kernel.

    A ghost-real pair at coupling lambda > 0 has sigma -> lambda * sigma, i.e. U_g(r, lambda) = lambda ** power * U(r).
    Hence, at fixed positions, the ghost energy is U_g(lambda) = lambda ** power * A with A = sum_j u(r_gj) (the energy
    of a full particle at the ghost position), and dU_g / d lambda = power * lambda ** (power - 1) * A.
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, power: float = 12.0,
                 compression_contact: float = 0.5, compression_pair_range: int = 3):
        """
        The constructor of the GrandCanonicalSoftSpherePotential class.

        Parameters
        ----------
        epsilon : float, optional
            The energy scale of the potential.
        sigma : float, optional
            The length scale of the potential.
        power : float, optional
            The steepness of the potential.
        compression_contact : float, optional
            Reference contact distance c of the compression flow of TemperedGrandCanonicalKernel (excess gaps are the
            spacings minus c, or minus lambda c next to the ghost); it must stay below the typical spacing.
        compression_pair_range : int, optional
            Pairs up to this many ring neighbours apart are evaluated exactly along the compression flow; all others
            form a tail factor that is sampled by thinning with one random pair per proposal.

        Raises
        ------
        base.exceptions.ConfigurationError
            If epsilon < 0, sigma <= 0 or power <= 1.
        """
        super().__init__(prefactor=1.0)
        if epsilon < 0.0 or sigma <= 0.0 or power <= 1.0:
            raise ConfigurationError(f"Give epsilon >= 0, sigma > 0 and power > 1 in {self.__class__.__name__}.")
        if compression_contact <= 0.0 or compression_pair_range < 1:
            raise ConfigurationError(f"Give compression_contact > 0 and compression_pair_range >= 1 in "
                                     f"{self.__class__.__name__}.")
        self._epsilon = epsilon
        self._sigma = sigma
        self._power = power
        self._compression_contact = compression_contact
        self._compression_pair_range = compression_pair_range
        self._flow = None

    def get_initial_positions(self):
        """
        Returns the initial positions: an evenly spaced ring of the real particles and the ghost (no overlaps, so the
        compression flow does not start from overlapping pairs), or the file range_of_initial_particle_positions.

        Returns
        -------
        numpy.ndarray
            Array of shape (number_of_rows, 1).
        """
        if isinstance(range_of_initial_particle_positions, str):
            return super().get_initial_positions()
        number_of_rows = self._initial_number_of_real_particles + (1 if self.has_ghost() else 0)
        positions = ((np.arange(number_of_rows) + 0.5) * size_of_particle_space[0] / number_of_rows)[:, None]
        self._check_initial_positions(positions)
        return positions

    def _get_pair_energies(self, distances, scales):
        return self._epsilon * (scales * self._sigma / distances) ** self._power

    def _get_next_event_from_separations(self, separations, scales, temperature):
        """
        Returns the distance to the next factor event and the vetoing particle.
        """
        if self._epsilon == 0.0:  # Ideal gas reference
            return size_of_particle_space[0], None
        half_length = 0.5 * size_of_particle_space[0]
        ahead = separations > 0.0
        abs_separations = np.asarray(np.where(ahead, separations, half_length))
        free_distances = np.asarray(np.where(ahead, 0.0, half_length + separations))
        alpha = (-np.log(np.random.uniform(0.0, 1.0, len(separations))) * temperature /
                 (self._epsilon * (scales * self._sigma) ** self._power) + 1 / abs_separations ** self._power)
        distances = free_distances + abs_separations - alpha ** (-1 / self._power)
        vetoing_index = int(np.argmin(distances))
        return max(float(distances[vetoing_index]), 0.0), vetoing_index

    def get_ghost_profile(self, positions):
        """Returns sum over real particles the energy the ghost would have at lambda=1."""
        distances = self._get_ghost_distances(positions)
        # Set all scales to 1 including ghost particle as lambda=1.
        self._ghost_profile = float(np.sum(self._get_pair_energies(distances, np.ones(len(distances)))))
        return self._ghost_profile

    def get_ghost_energy(self, coupling, profile):
        coupling = np.asarray(coupling, dtype=float)
        return np.where(coupling > 0.0, coupling ** self._power * profile, 0.0)

    def get_ghost_force(self, coupling):
        return self._power * coupling ** (self._power - 1.0) * self._ghost_profile

    def get_next_coupling_event(self, coupling, temperature, direction):
        if direction < 0 or self._ghost_profile <= 0.0:  
            return np.inf
        target = coupling ** self._power - np.log(np.random.uniform(0.0, 1.0)) * temperature / self._ghost_profile
        return target ** (1.0 / self._power) - coupling

    def get_ghost_wall(self):
        return np.inf  # No hard walls for soft sphere potential

    def choose_coupling_event_particle(self, positions):
        """d u_j / d lambda is proportional to u_j at full coupling, i.e. to r_j ** (- power)."""
        separations = self._get_minimum_image(positions[:-1, 0] - positions[-1, 0])
        weights = (np.min(np.abs(separations)) / np.abs(separations)) ** self._power
        index = int(np.random.choice(len(weights), p=weights / np.sum(weights)))
        return index, float(separations[index])

    def get_lift_ratio(self, distance, coupling):
        """u = epsilon * (lambda * sigma / r) ** power, so |du/dr| / (du/dlambda) = lambda / r."""
        return coupling / distance

    def get_compression_contact(self):
        return self._compression_contact

    def prepare_compression(self, positions, velocities, coupling, lowest, highest, temperature):
        """
        Caches the flow line through positions: every row moves as x + velocities * (lambda - coupling), so every
        pair distance is linear in lambda.  Near pairs (at most compression_pair_range ring neighbours apart, with at
        least one moving row) are kept with their distances and rates.  Every other pair with a moving row belongs to
        the tail, whose rate per pair is bounded by beta power epsilon sigma^power (1 / r^power + v_max / r^(power + 1))
        with r the shortest stretch of compression_pair_range + 1 ring spacings in the returned interval (linear in
        lambda, so checked at its ends) and v_max the largest rate of change of a pair distance.  Near pairs use the
        minimum image of their forward distance (linear in lambda), folded at L / 2.  The interval, at first the whole
        of [lowest, highest], is narrowed around coupling while a near pair or a stretch would close in it, or while
        the summed tail bound exceeds 5 per unit lambda (down to a half width of 0.02).
        """
        self._flow = None
        if self._epsilon == 0.0 or len(positions) < 2:
            return lowest, highest
        length = size_of_particle_space[0]
        coordinates = positions[:, 0].copy()
        number_of_rows = len(positions)
        ghost_index = number_of_rows - 1
        """Ring order starting at the ghost; particles never pass each other or the ghost while lambda > 0."""
        order = np.argsort(np.mod(coordinates - coordinates[ghost_index], length))
        rank = np.empty(number_of_rows, dtype=int)
        rank[order] = np.arange(number_of_rows)
        pair_range = min(self._compression_pair_range, (number_of_rows - 1) // 2)
        offsets = np.arange(1, pair_range + 1)
        first = np.repeat(order, pair_range)
        second = order[(np.arange(number_of_rows)[:, None] + offsets[None, :]).ravel() % number_of_rows]
        moving = velocities != 0.0
        moving[ghost_index] = True
        keep = moving[first] | moving[second]
        first, second = first[keep], second[keep]
        distances = np.mod(coordinates[second] - coordinates[first], length)  # Forward, folded when used
        rates = velocities[second] - velocities[first]
        """Stretches of pair_range + 1 ring spacings bound the distance of every tail pair from below."""
        stretch_ends = order[(np.arange(number_of_rows) + pair_range + 1) % number_of_rows]
        stretches = np.mod(coordinates[stretch_ends] - coordinates[order], length)
        stretch_rates = velocities[stretch_ends] - velocities[order]
        moving_rows = np.nonzero(moving)[0]
        speed = 2.0 * float(np.max(np.abs(velocities)))
        prefactor = self._power * self._epsilon * self._sigma ** self._power / temperature
        half_width = max(coupling - lowest, highest - coupling)
        while True:
            low, high = max(lowest, coupling - half_width), min(highest, coupling + half_width)
            shortest = min(float(np.min(stretches + stretch_rates * (low - coupling))),
                           float(np.min(stretches + stretch_rates * (high - coupling))))
            near_ends = np.concatenate([distances + rates * (low - coupling), distances + rates * (high - coupling),
                                        [0.5 * length]])
            if shortest > 0.0 and np.min(near_ends) > 0.0 and np.max(near_ends) < length:
                tail_bound = prefactor * (1.0 / shortest ** self._power + speed / shortest ** (self._power + 1.0))
                total_tail_bound = len(moving_rows) * (number_of_rows - 1) * tail_bound
                if total_tail_bound < 5.0 or half_width < 0.02:
                    break
            half_width *= 0.5
            if half_width < 1.0e-6:
                raise RuntimeError(f"The compression flow of {self.__class__.__name__} found no valid interval.")
        self._flow = {"coupling": coupling, "coordinates": coordinates, "velocities": velocities.copy(),
                      "distances": distances, "rates": rates, "ghost_pairs": (first == ghost_index) |
                      (second == ghost_index), "rank": rank, "pair_range": pair_range, "moving": moving,
                      "moving_rows": moving_rows, "tail_bound": tail_bound, "total_tail_bound": total_tail_bound,
                      "prefactor": prefactor}
        return low, high

    def _get_flow_terms(self, coupling):
        """Returns the near-pair (minimum-image) distances at coupling, d distance / d lambda and the ghost-pair
        mask."""
        flow = self._flow
        forward = flow["distances"] + flow["rates"] * (coupling - flow["coupling"])
        folded = forward > 0.5 * size_of_particle_space[0]
        return (np.where(folded, size_of_particle_space[0] - forward, forward), np.where(folded, -flow["rates"],
                                                                                         flow["rates"]),
                flow["ghost_pairs"])

    def get_compression_energy_rate(self, coupling, direction):
        """
        beta d U_near / d t: real pairs -power u (d r / dt) / r, ghost pairs power u (v_lambda / lambda - (d r / dt) /
        r) with u = epsilon (lambda sigma / r) ** power.
        """
        if self._flow is None:
            return 0.0
        distances, rates, ghost_pairs = self._get_flow_terms(coupling)
        if coupling <= 0.0:  # The ghost does not interact
            distances, rates, ghost_pairs = distances[~ghost_pairs], rates[~ghost_pairs], ghost_pairs[~ghost_pairs]
        scales = np.where(ghost_pairs, coupling, 1.0)
        energies = (scales * self._sigma / distances) ** self._power
        relative_rates = np.where(ghost_pairs, 1.0 / max(coupling, 1.0e-300), 0.0) - rates / distances
        return direction * self._flow["prefactor"] * float(np.sum(energies * relative_rates)) / self._sigma ** \
            self._power

    def get_compression_energy_rate_bound(self, coupling, step, direction):
        """
        Upper bound of get_compression_energy_rate while lambda moves from coupling a further distance step (at most
        get_compression_step_cap, inside the interval of prepare_compression).  Along the flow r and lambda are
        linear in t.  A real pair's d u / dt = -power epsilon sigma^power (d r / dt) / r^(power + 1) is largest at the
        end of the step.  A ghost pair's d u / dt = power epsilon sigma^power C f(t), with C = v_lambda r - lambda
        (d r / dt) constant and f = lambda^(power - 1) / r^(power + 1), is bounded exactly: f' changes sign at most
        once, at t* = ((power - 1) v_lambda r - (power + 1) lambda (d r / dt)) / (2 v_lambda (d r / dt)), so its
        extreme values on the step are among f(0), f(step) and f(t*).  A pair whose minimum image changes during the
        step (forward distance crossing L / 2, where it is far apart) is bounded by its modulus.
        """
        if self._flow is None:
            return 0.0
        half_length = 0.5 * size_of_particle_space[0]
        start, rates, ghost_pairs = self._get_flow_terms(coupling)
        distance_rates = direction * rates
        unfolded_end = start + distance_rates * step
        crossing = unfolded_end > half_length
        end = np.where(crossing, 2.0 * half_length - unfolded_end, unfolded_end)
        smallest = np.minimum(start, end)
        end_coupling = max(coupling + direction * step, 0.0)
        high = max(coupling, end_coupling)
        bounds = np.where(crossing, np.abs(distance_rates) / smallest ** (self._power + 1.0),
                          -distance_rates / end ** (self._power + 1.0))
        bounds = np.where(ghost_pairs, 0.0, bounds)
        if np.any(ghost_pairs) and high > 0.0:
            ghost_start, ghost_rates = start[ghost_pairs], distance_rates[ghost_pairs]
            ghost_end, ghost_crossing = end[ghost_pairs], crossing[ghost_pairs]
            constants = direction * ghost_start - coupling * ghost_rates
            values = [coupling ** (self._power - 1.0) / ghost_start ** (self._power + 1.0),
                      end_coupling ** (self._power - 1.0) / ghost_end ** (self._power + 1.0)]
            products = direction * ghost_rates
            with np.errstate(divide="ignore", invalid="ignore"):
                critical = (((self._power - 1.0) * direction * ghost_start - (self._power + 1.0) * coupling *
                             ghost_rates) / (2.0 * products))
            inside = (products != 0.0) & (critical > 0.0) & (critical < step)
            critical = np.where(inside, critical, 0.0)
            critical_value = ((coupling + direction * critical) ** (self._power - 1.0) /
                              (ghost_start + ghost_rates * critical) ** (self._power + 1.0))
            largest = np.maximum(np.maximum(values[0], values[1]), np.where(inside, critical_value, 0.0))
            smallest_value = np.minimum(np.minimum(values[0], values[1]), np.where(inside, critical_value, np.inf))
            exact = constants * np.where(constants >= 0.0, largest, smallest_value)
            modulus = ((half_length + high * np.abs(ghost_rates)) * high ** (self._power - 1.0) /
                       smallest[ghost_pairs] ** (self._power + 1.0))
            bounds[ghost_pairs] = np.where(ghost_crossing, modulus, exact)
        return self._flow["prefactor"] * float(np.sum(bounds))

    def get_compression_step_cap(self, coupling, direction):
        """Longest step in which no near pair closes by more than half its distance or moves by more than L / 4."""
        if self._flow is None or len(self._flow["rates"]) == 0:
            return np.inf
        distances, rates, _ = self._get_flow_terms(coupling)
        cap = 0.25 * size_of_particle_space[0] / max(float(np.max(np.abs(rates))), 1.0e-300)
        closing = direction * rates < 0.0
        if np.any(closing):
            cap = min(cap, float(np.min(0.5 * distances[closing] / np.abs(rates[closing]))))
        return cap

    def get_compression_tail_rate_bound(self):
        return 0.0 if self._flow is None else self._flow["total_tail_bound"]

    def accept_compression_tail_event(self, coupling, direction):
        """
        Thinning of a tail proposal: an ordered pair (a moving row, any other row) is drawn uniformly; pairs of two
        moving rows are drawn twice, so they are kept with probability 1/2; near pairs belong to the near factor and are
        rejected; otherwise lambda reverses with probability [beta d u / dt]^+ / tail_bound.
        """
        flow = self._flow
        number_of_rows = len(flow["coordinates"])
        first = int(flow["moving_rows"][np.random.randint(len(flow["moving_rows"]))])
        second = int(np.random.randint(number_of_rows - 1))
        second += second >= first
        if flow["moving"][second] and np.random.uniform(0.0, 1.0) < 0.5:
            return False
        separation = (flow["rank"][second] - flow["rank"][first]) % number_of_rows
        if min(separation, number_of_rows - separation) <= flow["pair_range"]:
            return False
        offset = coupling - flow["coupling"]
        positions = flow["coordinates"][[first, second]] + flow["velocities"][[first, second]] * offset
        difference = float(self._get_minimum_image(positions[1] - positions[0]))
        distance = abs(difference)
        distance_rate = np.sign(difference) * (flow["velocities"][second] - flow["velocities"][first])
        ghost_pair = number_of_rows - 1 in (first, second)
        if ghost_pair and coupling <= 0.0:
            return False
        scale = coupling if ghost_pair else 1.0
        relative_rate = (1.0 / coupling if ghost_pair else 0.0) - distance_rate / distance
        rate = direction * flow["prefactor"] * (scale / distance) ** self._power * relative_rate
        return np.random.uniform(0.0, 1.0) * flow["tail_bound"] < rate

    def get_description(self):
        return {"type": "soft_sphere", "epsilon": self._epsilon, "sigma": self._sigma, "power": self._power,
                "box": [float(size_of_particle_space[0])]}
