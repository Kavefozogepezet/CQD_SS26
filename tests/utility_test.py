import itertools

import numpy as np
import jax
import jax.numpy as jnp

import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.operators as ops
import Comp_Quant_Dynam.hamiltonians as ham


class Test_example_function:

    def test_example_func_zero(self):
        x = 0
        expected =  1 / np.pi ** (1 / 4)
        result = util.example_func(x)
        assert np.allclose(expected, result)

    def test_example_func_symmetry(self):
        x = np.array([-1, 1])
        result = util.example_func(x)
        assert np.allclose(result[0], result[1])


###################### Solution sheet 2 ######################


class Test_create_xvals:

    L = 10
    npoints = 101
    
    def test_create_xvals_length(self):
        
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert len(xvals) == self.npoints

    def test_create_xvals_range(self):

        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[0], -self.L/2)
        assert np.isclose(xvals[-1], self.L/2)

    def test_create_xvals_spacing(self):
        xvals, dx = util.create_xvals(self.L, self.npoints)
        expected_dx = self.L / (self.npoints - 1)
        assert np.isclose(dx, expected_dx)

    def test_create_xvals_zero_centered(self):
        # only works if npoints is odd
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[self.npoints // 2], 0) # check that the middle point is approximately zero


###################### Solution sheet 3 ######################


class Test_FT_iFT:

    L = 10
    npoints = 101
    xvals, dx = util.create_xvals(L, npoints, endpoint=False)
    kvals = np.fft.fftfreq(npoints, d=dx) * 2 * np.pi

    def test_FT_iFT_identity(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        phi = util.FT(psi, self.xvals, self.kvals)
        psi_reconstructed = util.iFT(phi, self.xvals, self.kvals)
        assert np.allclose(psi, psi_reconstructed)


    def test_FT_assertion(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        with np.testing.assert_raises(AssertionError):
            util.FT(psi, self.xvals, self.kvals[:-1]) # mismatch in length of k

    def test_iFT_assertion(self):
        phi = util.FT(util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1), self.xvals, self.kvals)
        with np.testing.assert_raises(AssertionError):
            util.iFT(phi, self.xvals[:-1], self.kvals) # mismatch in length of x

class Test_gaussian_wave_packet:

    L = 20
    npoints = 201
    xvals, dx = util.create_xvals(L, npoints)

    def test_gaussian_wave_packet_normalization(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=1)
        norm = np.sum(np.abs(psi)**2) * self.dx
        assert np.isclose(norm, 1)

    def test_gaussian_wave_packet_symmetry(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=0)
        assert np.allclose(psi[60], psi[80]) # check that the wave packet is symmetric around x0=-3

class Test_create_tvecs:

    def test_create_tvecs_length(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert len(tvec) == tsteps + 1

    def test_create_tvecs_values(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert np.isclose(tvec[0], 0)
        assert np.isclose(tvec[-1], tsteps * dt)

class Test_idx2state_state2idx:

    N1 = 3
    N2 = 4

    def test_idx2state_state2idx_consistency(self):
        idx_recon = []
        for i in range(self.N1 * self.N2):
            state = util.idx2state(self.N1, self.N2, i)
            idx_recon.append(util.state2idx(self.N1, self.N2, state))
        assert np.array_equal(np.arange(self.N1 * self.N2), idx_recon)

    def test_idx2state_state2idx_out_of_bounds(self):
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, -1) # negative index
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, self.N1 * self.N2) # index equal to dimension
        idx = util.state2idx(self.N1, self.N2, [self.N1, 0]) # n1 out of bounds
        assert idx == -1
        idx = util.state2idx(self.N1, self.N2, [0, self.N2]) # n2 out of bounds
        assert idx == -1

    def test_idx2state_specific_cases(self):
        # test some specific cases for idx2state and state2idx
        i = 0
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [0, 0]
        i = 5
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [1, 1]
        i = 11
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [2, 3]


###################### Solution sheet 4 ######################


class Test_create_coherent_state:

    N = 100

    def test_create_coherent_state_normalization(self):
        alpha = 1 + 1j
        state = util.create_coherent_state(self.N, alpha)
        norm = np.sum(np.abs(state)**2)
        assert np.isclose(norm, 1)

    def test_create_coherent_state_alpha_zero(self):
        alpha = 0
        state = util.create_coherent_state(self.N, alpha)
        expected = np.zeros(self.N)
        expected[0] = 1
        assert np.allclose(state, expected)

    def test_a_operator_sparse_consistency(self):
        alpha = 3
        a_op = ops.a_operator_sparse(self.N).toarray()
        init_state = util.create_coherent_state(self.N, alpha)
        applied_a_op = a_op @ init_state
        applied_a_op /= alpha # should be equal to the original state
        assert np.allclose(1, np.vdot(applied_a_op[:-1], init_state[:-1]), atol=1e-10)
        # check the coherent-state eigenvalue relation for the annihilation operator, up to truncation effects in the last basis element.

class Test_expectation_value:

    N = 100

    L = 20
    npoints = 2001
    xvals, dx = util.create_xvals(L, npoints)

    def test_expectation_value_hermitian(self):
        # test that the expectation value of a Hermitian operator is real
        alpha = 1 + 1j
        state = util.create_coherent_state(self.N, alpha)
        x_op = ops.x_operator_sparse(self.N).toarray()
        exp_val = util.expectation_value(state, x_op)
        assert np.isclose(np.imag(exp_val), 0)

    def test_expectation_value_known(self):
        # test the expectation value of the number operator in a coherent state, which should be |alpha|^2
        alpha = 2.0 + 1j
        state = util.create_coherent_state(self.N, alpha)
        print("state: ", state)
        n_op = ops.n_operator_sparse(self.N)
        exp_val = util.expectation_value(state, n_op)
        expected = np.abs(alpha)**2
        assert np.isclose(exp_val, expected, atol=1e-10)

    def test_expectation_value_iterable(self):
        # test that the function can handle an iterable of operators
        x0 = -1
        sigma = 1
        p0 = 1
        state = util.gaussian_wave_packet(self.xvals, x0=x0, sigma=sigma, p0=p0)
        print("x_prob = ", sum(np.abs(state)**2 * self.xvals) * self.dx)
        
        
        x_op = np.diag(self.xvals) # position operator in the x basis
        p_op = np.zeros((self.npoints, self.npoints), dtype=complex) # momentum operator in the x basis, using finite difference approximation
        for i in range(1, self.npoints - 1):
            p_op[i, i - 1] = 1j / (2 * self.dx)
            p_op[i, i + 1] = -1j / (2 * self.dx)

        exp_vals = util.expectation_value(state, [x_op, p_op])
        assert len(exp_vals) == 2
        assert np.isclose(np.imag(exp_vals[0]), 0) # expectation value of x should be real
        assert np.isclose(np.imag(exp_vals[1]), 0) # expectation value of p should be real
        expected = [x0, p0]
        exp_val_norm = np.real(exp_vals) * self.dx
        assert np.allclose(exp_val_norm, expected, atol=1e-4)


###################### Exercise sheet 7 ######################


class Test_Husimi_proj:

    N = 100
    #phi_test = np.pi / 3
    #theta_test = np.pi / 2
    ngrid = 101

    def test_husimi_front_back_symmetry(self):
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi / 2

        #psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_back
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_front_back_symmetry_theta_pi(self):
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi

        #psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_back
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_top_front_symmetry(self):
        # test that the Husimi functions for the top and front states are symmetric with respect to the theta axis
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi / 2

        psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        #psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_top = util.Husimi_top(self.N, psi_top, self.ngrid, self.ngrid)

        #Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_top
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_th_phi_symmetry(self):
        # test that the Husimi functions are symmetric with respect to theta -> pi - theta
        phi = np.pi / 3
        theta_1 = np.pi / 3
        theta_2 = np.pi - theta_1

        psi_1 = util.CSS(self.N, theta_1, phi)  # state of the CSS basis
        psi_2 = util.CSS(self.N, theta_2, phi)  # state of the CSS basis

        Theta, Phi, H1 = util.Husimi_th_ph(self.N, psi_1, self.ngrid, self.ngrid)
        Theta, Phi, H2 = util.Husimi_th_ph(self.N, psi_2, self.ngrid, self.ngrid)
        diff = H1 - np.flip(H2, axis=0) # flip H2 along the theta axis
        assert np.allclose(diff, 0, atol=1e-10)
    

    def test_husimi_z_phi_symmetry_phi(self):
        # test that the Husimi functions are symmetric with respect to z -> -z

        phi = np.pi / 3
        theta_1 = np.pi / 3
        theta_2 = np.pi - theta_1

        psi_1 = util.CSS(self.N, theta_1, phi)  # state of the CSS basis
        psi_2 = util.CSS(self.N, theta_2, phi)  # state of the CSS basis

        Z, Phi, H1 = util.Husimi_z_phi(self.N, psi_1, self.ngrid, self.ngrid)
        Z, Phi, H2 = util.Husimi_z_phi(self.N, psi_2, self.ngrid, self.ngrid)
        diff = H1 - np.flip(H2, axis=0) # flip H2 along the z axis
        assert np.allclose(diff, 0, atol=1e-10)
    

###################### Exercise sheet 8 ######################


class Test_partial_trace:

    def test_partial_trace_product(self):
        N = 3
        psi_full = np.eye(1, 2 ** N, 5)[0] # |101> state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_full, 1) # trace out the last spin
        expected_psi = np.eye(1, 2 ** (N - 1), 2)[0] # |10> state in the reduced Hilbert space
        expected_rho = np.outer(expected_psi, expected_psi.conj())
        assert np.allclose(rho_reduced, expected_rho)

    def test_partial_trace_entangled(self):
        N = 3
        psi_ghz = (1 / np.sqrt(2)) * (np.eye(1, 2 ** N, 0)[0] + np.eye(1, 2 ** N, 7)[0]) # GHZ state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_ghz, 2) # trace out the last two spins
        expected_rho = 0.5 * np.eye(2) # reduced density matrix for the first spin, which is maximally mixed
        assert np.allclose(rho_reduced, expected_rho)

class Test_entanglement_entropy:

    def test_entanglement_entropy_product(self):
        N = 3
        psi_full = np.eye(1, 2 ** N, 5)[0] # |101> state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_full, 1) # trace out the last spin
        S = util.entanglement_entropy(rho_reduced) # trace out the last spin
        expected_S = 0.0 # product state should have zero entanglement entropy
        assert np.isclose(S, expected_S)

    def test_entanglement_entropy_entangled(self):
        N = 3
        psi_ghz = (1 / np.sqrt(2)) * (np.eye(1, 2 ** N, 0)[0] + np.eye(1, 2 ** N, 7)[0]) # GHZ state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_ghz, 1) # trace out the last two spins
        S = util.entanglement_entropy(rho_reduced) # trace out the last two spins
        expected_S = 1 # reduced density matrix for the first spin is maximally mixed, so S should be log(2)
        assert np.isclose(S, expected_S)
        
    def test_entanglement_entropy_mixed_state(self):
        # test that the entanglement entropy of a mixed state is non-negative
        rho = np.diag([1/3, 2/3]) # mixed state for a single qubit
        S = util.entanglement_entropy(rho)
        S_expected = 1/3 * np.log2(3) +  2/3 * np.log2(3 / 2)
        assert np.isclose(S, S_expected)


###################### Exercise sheet 9 ######################

class Test_n_party_idx2state:

    def test_n_party_idx2state_first_state(self):
        # Test edge cases for n_party_idx2state
        N = 6
        local_dim = 3
        
        idx = 0
        expected_state = [-1] * N
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)
    
    def test_n_party_idx2state_last_state(self):
        N = 6
        local_dim = 3
        
        idx = local_dim ** N - 1
        expected_state = [1] * N
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)

    def test_n_party_idx2state_middle_state(self):
        N = 6
        local_dim = 3
        
        idx = 11 # corresponds to state [-1, -1, -1, 0, -1, 1]
        expected_state = [-1, -1, -1, 0, -1, 1]
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)


###################### Solution sheet 10 ######################


def _all_configs(N):
    # all 2^N spin configurations in the 0/1 (computational) convention
    return np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.int32)

def _config_to_index(s):
    # leftmost spin = most significant bit, matching operators.n_party_op_sparse ordering
    N = len(s)
    return int(sum(int(v) * 2 ** (N - 1 - k) for k, v in enumerate(s)))

def _model_state_vector(model, params, N):
    # dense wave function psi(s) over all configs, ordered to match build_H_TFIM_individual
    psi = np.zeros(2 ** N, dtype=complex)
    for s in _all_configs(N):
        psi[_config_to_index(s)] = complex(util.psi_theta(model, params, jnp.array(s)))
    return psi

def _exact_energy(psi, H):
    return float((psi.conj() @ (H @ psi) / (psi.conj() @ psi)).real)


class Test_variational_models:

    N = 5

    def test_jastrow_exact_formula(self):
        """Check the Jastrow formula against a hand-computed value for a fixed config and
        fixed J1, J2, mirroring exactly what Jastrow.__call__ does with jnp.roll. This
        confirms the periodic-boundary nearest/next-nearest-neighbor sum is wired up
        correctly -- no off-by-one in the roll direction, correct sign/coefficient placement.
        """

        model = util.Jastrow()
        params = {"params": {"j1": jnp.array([0.3]), "j2": jnp.array([-0.2])}}
        x = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        expected = np.sum(0.3 * x * np.roll(x, -1) - 0.2 * x * np.roll(x, -2))
        assert np.isclose(float(model.apply(params, jnp.array(x))), expected, atol=1e-6)

    def test_jastrow_batched_consistency(self):
        """flax models are typically called on batches of spin configs (batch, N) during
        MCMC/training but sometimes on a single config (N,). Verify the two call
        conventions agree: applying the model to a batch of 3 configs at once must match
        applying it to each row individually. Guards against broadcasting bugs in the
        jnp.sum(..., axis=-1) reduction.
        """

        model = util.Jastrow()
        params = {"params": {"j1": jnp.array([0.3]), "j2": jnp.array([-0.2])}}
        X = jnp.array([[0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.int32)
        batched = np.asarray(model.apply(params, X))
        single = np.array([float(model.apply(params, x)) for x in X])
        assert batched.shape == (3,)
        assert np.allclose(batched, single, atol=1e-6)

    def test_ffnn_output_shapes(self):
        """Shape/sanity check rather than a numerical-correctness one -- there's no simple
        closed-form to check an FFNN's output against, unlike Jastrow's explicit formula.
        Verifies a single spin string of shape (N,) produces a scalar output (shape ()),
        matching the convention Jastrow and the rest of the pipeline expect for a single
        sample; a batch of 3 spin strings, shape (3, N), produces shape (3,) -- one log psi
        per row, not something flattened or broadcast incorrectly; and all batch outputs
        are finite, catching a bad initializer or activation choice producing NaN/inf right
        out of the gate. Would catch e.g. a missing out[..., 0] (leaving a trailing size-1
        dimension that silently breaks broadcasting downstream in grad_E_theta_MC_TFIM) or
        a layer loop that fails to actually chain x through successive nn.Dense calls.
        """
        model = util.FFNN(features=(8, 8), out_dim=1, actfunc=jax.nn.tanh)
        params = model.init(jax.random.PRNGKey(1), jnp.ones((self.N,), jnp.float32))
        out_single = np.asarray(model.apply(params, jnp.ones((self.N,), jnp.float32)))
        out_batch = np.asarray(model.apply(params, jnp.ones((3, self.N), jnp.float32)))
        assert out_single.shape == ()
        assert out_batch.shape == (3,)
        assert np.all(np.isfinite(out_batch))

    def test_psi_p_logstar_consistency(self):
        """Check the small wrapper functions built on top of the model output are mutually
        consistent: p_theta(model, params, s) must equal |psi_theta(model, params, s)|^2,
        i.e. the Born probability is really the squared modulus of the amplitude, and
        logpsi_star_theta(model, params, s) must equal the complex conjugate of the raw
        model output (log psi). The latter is the O_k* term from the gradient formula in
        Exercise 2, so it's pinned down here even though it's only exercised for
        real-valued output in Exercise 1.
        """

        model = util.Jastrow()
        params = {"params": {"j1": jnp.array([0.3]), "j2": jnp.array([-0.2])}}
        s = jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32)
        assert np.isclose(float(util.p_theta(model, params, s)),
                          abs(complex(util.psi_theta(model, params, s))) ** 2, atol=1e-6)
        assert np.isclose(complex(util.logpsi_star_theta(model, params, s)),
                          np.conj(complex(model.apply(params, s))))


class Test_MCMC_Sampler:

    N = 4

    def _model_params(self, N, shift=0.0):
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((N,), jnp.int32))
        return model, jax.tree_util.tree_map(lambda x: x + shift, params)

    def test_samples_shape_and_binary(self):
        """Basic sanity check: requesting 50 samples for N=4 spins must return an array of
        shape (50, 4), with every entry in {0, 1} -- i.e. the sampler never produces
        something that isn't a valid spin configuration (no stray floats, no values
        outside the binary encoding).
        """

        model, params = self._model_params(self.N)
        init = jnp.ones((self.N,), dtype=jnp.int32)
        samples = np.asarray(util.MCMC_Sampler_Metropolis_Hastings(
            model, params, init, num_samples=50, PRNGkey=jax.random.PRNGKey(7)))
        assert samples.shape == (50, self.N)
        assert set(np.unique(samples).tolist()).issubset({0, 1})

    def test_sampler_deterministic(self):
        """Run the sampler twice with the identical PRNGKey(7) and check the two chains are
        bit-for-bit identical. Tests that the sampler is a pure function of its inputs (no
        hidden global RNG state, no accidental use of numpy.random instead of jax.random)
        -- important because JAX's reproducibility story depends on PRNG keys being
        threaded explicitly rather than mutated implicitly.
        """

        model, params = self._model_params(self.N)
        init = jnp.ones((self.N,), dtype=jnp.int32)
        kw = dict(num_samples=50, PRNGkey=jax.random.PRNGKey(7))
        s1 = util.MCMC_Sampler_Metropolis_Hastings(model, params, init, **kw)
        s2 = util.MCMC_Sampler_Metropolis_Hastings(model, params, init, **kw)
        assert np.array_equal(np.asarray(s1), np.asarray(s2))

    def test_samples_follow_born_distribution(self):
        """Validate that the whole point of Metropolis-Hastings actually holds: samples
        should be distributed according to p(s) = |psi(s)|^2 / Z. For a small N=3 system
        (8 configurations), draw 1000 MCMC samples and bin them into an empirical histogram
        (via _config_to_index), then compare against the exact Born distribution built by
        enumerating all 2^N configs and evaluating |psi_theta(s)|^2 directly (via
        _model_state_vector). Assert the L1 distance between empirical and exact
        distributions is below 0.15 -- loose enough for MC noise at 400 samples, but tight
        enough to catch a genuinely broken acceptance rule (e.g. a sign error in
        p_accept, or sampling uniformly instead of from p_theta). Shape/determinism tests
        alone would not catch this, since they don't check which distribution is sampled.
        """

        N = 3
        model, params = self._model_params(N, shift=0.5)
        samples = np.asarray(util.MCMC_Sampler_Metropolis_Hastings(
            model, params, jnp.ones((N,), jnp.int32),
            num_samples=1000, PRNGkey=jax.random.PRNGKey(21)))
        emp = np.bincount([_config_to_index(s) for s in samples], minlength=2 ** N) / len(samples)
        psi = _model_state_vector(model, params, N)
        p = np.abs(psi) ** 2
        p /= p.sum()
        assert np.abs(emp - p).sum() < 0.15  # L1 distance, loose for MC noise


class Test_local_energy_TFIM:

    def test_local_energy_matches_exact_hamiltonian(self):
        """Strongest correctness check for the local energy: compute E_loc(s) for every one
        of the 2^N configurations (feeding each as its own batch of size 1 makes
        grad_E_theta_MC_TFIM return exactly E_loc(s)), then average them weighted by the
        exact Born distribution |psi(s)|^2, and check this equals <psi|H|psi>/<psi|psi>
        computed by directly diagonalizing the true Hamiltonian matrix
        (build_H_TFIM_individual). Any sign or roll-direction error in get_Eloc would break
        this end-to-end identity even though the code runs without error -- it validates
        the local-energy formula against ground-truth quantum mechanics, not just shapes.
        """
        
        N, B = 4, 0.7
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((N,), jnp.int32))
        params = jax.tree_util.tree_map(lambda x: x + 0.35, params)  # non-uniform state

        Eloc = np.zeros(2 ** N)
        for s in _all_configs(N):
            Eloc[_config_to_index(s)] = float(
                util.grad_E_theta_MC_TFIM(B, model, params, jnp.array(s)[None, :])[0])
        psi = _model_state_vector(model, params, N)
        p = np.abs(psi) ** 2
        p /= p.sum()
        E_exact = _exact_energy(psi, ham.build_H_TFIM_individual(N, B).toarray())
        assert np.isclose(np.sum(p * Eloc), E_exact, atol=1e-4)

    def test_uniform_state_energy_is_minus_BN(self):
        """Closed-form special case that needs no exact diagonalization: with all params
        set to zero, log psi(s) = 0 for every config, so psi is constant, the Born
        distribution is exactly uniform, and every single-flip amplitude ratio in
        single_flip_energy is exactly 1 (since log psi(s') - log psi(s) = 0). That makes
        B_field_energy = -B*N deterministically for every configuration, while the
        interaction term averages to exactly 0 over the uniform ensemble (for each bond,
        the four equally-likely (s_i, s_i+1) combinations contribute +1, -1, -1, +1, which
        cancels). So the exact expected energy is provably -B*N -- this isolates and checks
        just the B-field term's sign and prefactor, independent of the interaction term.
        """

        N, B = 4, 0.9
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(1), jnp.ones((N,), jnp.int32))
        params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)  # log(psi)=0
        E, _ = util.grad_E_theta_MC_TFIM(B, model, params, jnp.array(_all_configs(N)))
        assert np.isclose(float(E), -B * N, atol=1e-4)

    def test_energy_gradient_matches_params_pytree(self):
        """Structural sanity check rather than a numerical one: confirms E is real, and
        that the returned gradient pytree has the same structure as params (matching
        keys/shapes), with all-finite values. Guards the ravel_pytree/unravel_fn round
        trip inside grad_E_theta_MC_TFIM -- if that broke, optimizer.update(grad_E,
        opt_state, params) in perform_gs_search would fail downstream with a cryptic
        pytree-mismatch error instead of a clear one here.
        """

        N, B = 4, 0.7
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((N,), jnp.int32))
        samples = jnp.array(_all_configs(N))
        E, grad = util.grad_E_theta_MC_TFIM(B, model, params, samples)
        assert np.isreal(float(E))
        assert jax.tree_util.tree_structure(grad) == jax.tree_util.tree_structure(params)
        assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree_util.tree_leaves(grad))


class Test_local_energy_tilted_TFIM:

    def test_tilted_local_energy_matches_exact_hamiltonian(self):
        """Same strategy as the plain-TFIM local-energy test, extended to the tilted
        model: compute E_loc(s) for every one of the 2^N configs (each fed as its own
        batch of size 1), Born-average them, and check the result against
        <psi|H|psi>/<psi|psi> from exact diagonalization of
        build_H_tilted_TFIM_individual(N, B, g).

        Unlike the ZZ and B-field terms, the longitudinal g term is odd under a global
        spin flip, so this test pins down the sigma_z sign convention: utility uses
        s_phys = 2*s - 1 (s=0 -> -1), matching operators.sigma_z_sparse (|0> -> -1).
        It fails if either side reverts to the opposite convention -- a mistake that
        the energy plots alone would not reveal, since H(g) and H(-g) are unitarily
        equivalent and share the same spectrum.
        """
        N, B, g = 4, 0.6, 0.3
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(2), jnp.ones((N,), jnp.int32))
        params = jax.tree_util.tree_map(lambda x: x + 0.25, params)

        Eloc = np.zeros(2 ** N)
        for s in _all_configs(N):
            Eloc[_config_to_index(s)] = float(
                util.grad_E_theta_MC_tilted_TFIM(B, g, model, params, jnp.array(s)[None, :])[0])
        psi = _model_state_vector(model, params, N)
        p = np.abs(psi) ** 2
        p /= p.sum()
        E_exact = _exact_energy(psi, ham.build_H_tilted_TFIM_individual(N, B, g).toarray())
        assert np.isclose(np.sum(p * Eloc), E_exact, atol=1e-4)


class Test_gs_search:

    def test_classical_limit_reaches_minus_N(self):
        """At B=0 the TFIM has no transverse field, so it reduces to a classical Ising
        chain whose exact ground state is the fully-aligned ferromagnet with energy
        exactly -N (N bonds, each contributing -1, periodic boundary conditions). Run
        perform_gs_search for 30 iterations and check both that the energy actually
        decreased from the first iteration to the last few, and that the final mean
        energy is within 0.1 of -N -- confirming the optimizer genuinely learns rather
        than being a no-op, using a case with a trivial, exactly-known answer.
        """

        N = 4
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((N,), jnp.int32))
        _, hist = util.perform_gs_search(model, N, params, B=0.0, num_iters=30,
                                         N_MC=25, lr=0.05, key=jax.random.PRNGKey(3))
        hist = np.asarray(hist)
        assert float(hist[0]) > float(np.mean(hist[-5:]))          # energy went down
        assert np.isclose(float(np.mean(hist[-5:])), -N, atol=0.1)  # reaches exact GS

    def test_variational_upper_bound(self):
        """Checks the variational principle itself: E_var(theta) >= E_gs must hold
        throughout optimization, up to a small MC/optimizer slack (here 0.1) since
        stochastic estimates can dip slightly below due to sampling noise, especially
        near convergence. Runs the ground state search for N=4, B=1 and asserts the mean
        of the last 10 logged energies doesn't fall below the exact ground state energy.
        This is the test that would catch a subtle sign error in get_Eloc that causes the
        reported energy to spuriously undershoot the true minimum -- a classic VMC bug
        pattern that shape/structure tests wouldn't notice.
        """

        N, B = 4, 1.0
        mc_slack = 0.1
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(6), jnp.ones((N,), jnp.int32))
        _, hist = util.perform_gs_search(model, N, params, B, num_iters=40,
                                         N_MC=40, lr=0.02, key=jax.random.PRNGKey(11))
        assert float(np.mean(np.asarray(hist)[-10:])) >= ham.E_TFIM_individual_exact(N, B) - mc_slack

    def test_plain_and_gpu_accelerated_agree(self):
        """Bridges the plain (Python for-loop) tilted ground-state search and its
        jax.lax.scan-based GPU-accelerated refactor: runs perform_gs_search_tilted and
        perform_gs_search_tilted_GPU_accelerated with identical hyperparameters and the
        same PRNGKey, and asserts the two energy histories match to 1e-5. Since the
        refactor is only supposed to make the loop JIT/GPU-friendly without changing its
        numerics, this is the regression test that would catch a bug where the PRNG key
        gets split/consumed in a different order inside scan_step than inside the Python
        loop -- a mistake that would silently produce a different but plausible-looking
        training trajectory instead of an outright crash.
        """
        N, B, g = 4, 0.8, 0.2
        model = util.Jastrow()
        params = model.init(jax.random.PRNGKey(5), jnp.ones((N,), jnp.int32))
        kw = dict(num_iters=6, N_MC=20, lr=0.05, key=jax.random.PRNGKey(9))
        _, h1 = util.perform_gs_search_tilted(model, params, N, B, g, **kw)
        _, h2 = util.perform_gs_search_tilted_GPU_accelerated(model, params, N, B, g, **kw)
        assert np.allclose(np.asarray(h1), np.asarray(h2), atol=1e-5)

