import numpy as np
from scipy.linalg import block_diag, inv
from scipy.optimize import fsolve
import itertools
from itertools import combinations_with_replacement, permutations
import sympy as sp
from sympy.physics.quantum import TensorProduct
from tqdm import tqdm
import scipy as scipy
import math
import matplotlib.pyplot as plt
import thewalrus as tw
import pandas as pd
import hickle as hkl
import copy
from joblib import Parallel, delayed
from sympy.polys import Poly, PolynomialRing
from sympy import symbols, I, expand, QQ
from numba import njit


TYP_PARAMS = {
    "mean_photon": 0.01,
    "schmidt_coeffs": [1],
    "detection_efficiency": 1,
    "bsm_efficiency": 1,
    "outcoupling_efficiency": 1,
    "dark_counts": 0,
    "visibility": 1,
}


def _wick_partitions(n):
    all_partitions = []
    indices = list(range(n))
    for partition in itertools.combinations(itertools.combinations(indices, 2), n//2):
        # Check if every index appears exactly once
        flat_partition = [idx for pair in partition for idx in pair]
        if len(set(flat_partition)) == n:
            all_partitions.append(partition)
    return np.array(all_partitions)

cache_wick_partitions = {
    2: _wick_partitions(2),
    4: _wick_partitions(4),
    6: _wick_partitions(6),
    8: _wick_partitions(8),
}

@njit(fastmath=True)
def wick_out_do_not_store_looping_pattern_numba_kernel(moment_vector, Anv, partitions):
    coeff_sum = 0.0
    for partition in partitions:
        # Convert index pairings to element pairings
        sum_factor = 1.0
        for i, j in partition:
            sum_factor *= Anv[moment_vector[i], moment_vector[j]]
        coeff_sum += sum_factor
    return coeff_sum


class tools:

    @staticmethod
    def permutation_matrix(permute):
        """
        Arguments
        - permute: The permutation array
        Output
        - The permutation matrix
        """
        n = len(permute)
        P = np.zeros((n, n))
        for i in range(n):
            P[i, int(permute[i])] = 1
        return P

    @staticmethod
    def expand_powers_to_symbols(array):
        """
        This function expands any basis state symbols which are raised to a power
        """
        expanded_array = []
        for item in array:
            if isinstance(item, sp.core.power.Pow):
                base, exp = item.as_base_exp()
                if exp.is_Integer:
                    expanded_array.extend([base] * int(exp))
                else:
                    expanded_array.append(item)
            else:
                expanded_array.append(item)
        return expanded_array

    @staticmethod
    def expand_powers_to_symbols_list(array):
        """
        This functin is used by the CvecZALM function to expand any basis state symbols which are raised to a power
        """

        expanded_array = []
        for item in array:
            if isinstance(item, sp.core.power.Pow):
                base, exp = item.as_base_exp()
                if exp.is_Integer:
                    expanded_array.extend([base] * exp)
                else:
                    expanded_array.append(item)
            else:
                expanded_array.append(item)
        return expanded_array

    @staticmethod
    def mcomb(rr):
        """
        Arguments:
            - rr = A desired length of permutations
        Outputs:
            - All permutations of 0 and 1 for that given length
        """

        def ordered_combinations_with_replacement(iterable, r):
            combinations = combinations_with_replacement(iterable, r)
            for combination in combinations:
                for perm in permutations(combination):
                    yield perm

        def clean(cb1):
            i = 0
            while i < len(cb1):
                j = i + 1
                while j < len(cb1):
                    if cb1[j] == cb1[i]:
                        cb1.pop(j)
                        i = 0
                    j += 1
                i += 1
            return cb1

        iterable = [0, 1]
        cb = []
        for combination in ordered_combinations_with_replacement(iterable, rr):
            cb.append(combination)

        return clean(clean(cb))  # Running it twice ensures absolutely no repeats

    @staticmethod
    def moment_vector_general(l):
        """
        Arguments
        - l: A vector defining the exponents of the alpha and beta variables for a general moment calculation
        Output
        - An array of all of the moments that are to be calculated. Generally, each class below has unique moment vectors for the quantities calculated. This funciton is unused and here for reference.
        """
        mds = len(l) # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        C = 1
        j = 0
        for i in l:
            C*= ((alp[j])**i)*((bet[j])**i)
            j += 1

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i))

        return Coutf

    @staticmethod
    def wick_pairings(moment_vector):
        """
        Generate all possible pairings (for Wick's theorem) of some moment vector element.
        """
        # TODO: urgent - cache this
        n = len(moment_vector)
        if n % 2 != 0:
            raise ValueError("Moment vector must have even length for Wick's theorem")

        # Generate all possible pairings of indices
        indices = list(range(n))
        all_pairings = []

        # TODO: Speed this up using Numba

        # Get all ways to partition the indices into pairs
        for partition in itertools.combinations(itertools.combinations(indices, 2), n//2):
            # Check if every index appears exactly once
            flat_partition = [idx for pair in partition for idx in pair]
            if len(set(flat_partition)) == n:
                all_pairings.append(partition)

        # Convert index pairings to element pairings
        element_pairings = []
        for pairing in all_pairings:
            element_pairing = []
            for i, j in pairing:
                element_pairing.append((moment_vector[i], moment_vector[j]))
            element_pairings.append(element_pairing)

        return element_pairings

    @staticmethod
    def wick_coupling_mat(input1, bv):
        """
        Arguments
            input1: The phase space input upon which we want to perform Wick coupling
            bv: The cooresponding basis vector, which will be used so that the output are matrix locations
        Outputs
            An array of matrix coordinates which will be added and multiplied to perform Wick's theorem
        """

        # Differentiate between the coefficients and the phase space variables, then save the coefficient so that it can be used later
        # TODO: later input1 should be a tuple (input1[0], input1[1:])
        coef = input1[0]
        input1a = input1[1:]

        # Create a dictionary mapping each basis vector element to its index (only once)
        # TODO: soon - precompute this reverse mapping when preparing the class - do this by calculating bv very early on, e.g. in __init__ and assigning it to a class property and never even bother passing it around as an argument to functions.
        # TODO: later - earlier do the transformation from symbols to indices
        bv_index_map = {element: idx for idx, element in enumerate(bv)}

        # Relate the phase coordinates to the basis vector using the map
        input = []
        for l in input1a:
            input.append(bv_index_map[l])  # O(1) lookup instead of O(n) .index()

        test = tools.wick_pairings(input)

        return (test,coef)  # The final array that contains all combinations which are valid for Wick coupling in terms of their location in the matrix, as well as the coeficient by which the result needs to be multiplied

    @staticmethod
    def wick_coupling_mat_list(input1, bv):
        """
        Arguments
            input1: The phase space input upon which we want to perform Wick coupling
            bv: The cooresponding basis vector, which will be used so that the output are matrix locations
        Outputs
            An array of matrix coordinates which will be added and multiplied to perform Wick's theorem
        """

        def all_combinations_of_two(elements):
            return list(set(itertools.combinations(elements, 2)))

        def all_combinations_of_n(elements, n):
            return list(set(itertools.combinations(elements, n)))

        # Differentiate between the coefficients and the phase space variables, then save the coefficient so that it can be used later
        input1a = input1[:]
        coef = 1 # TODO: Make it so that in the tuple the coefficient is the first element

        i = 0
        while i < len(input1a): # TODO: If structuring the input as a tuple, this loop will be free
            if type(input1a[i]) != sp.core.symbol.Symbol:
                coef *= input1a[i]
                input1a.pop(i)
            else:
                i += 1

        # Relate the phase coordinates to the basis vector
        input = []
        for l in input1a:
            input.append(bv.index(l)) # TODO: .index is itself a loop. Precompute this into a dictionary

        test = tools.wick_pairings(input)

        return [test,coef]  # The final array that contains all combinations which are valid for Wick coupling in terms of their location in the matrix, as well as the coeficient by which the result needs to be multiplied

    @staticmethod
    def wick_out(ar, Anv):
        """
        Input:
        - ar: tuple (The basis vector array, the coefficient)
        - Anv: The inverse of the A matrix
        Output:
        - A calculation of the wick element for each coupling
        """
        # TODO soon: use numba

        s = 0
        j = 0
        while j < len(ar[0]):
            si = 1
            for i in ar[0][j]:
                si *= Anv[i]
            s += si
            j += 1
        return s*ar[1]

    @staticmethod
    def wick_out_do_not_store_looping_pattern(Cni_instance, bv_index_map, Anv):
        """
        Input:
        - ar: tuple (The basis vector array, the coefficient)
        - Anv: The inverse of the A matrix
        Output:
        - A calculation of the wick element for each coupling
        """
        # TODO soon: use numba
        #####
        #test, coeff = tools.wick_coupling_mat(Cni_instance, xb)
        #####
        coef = Cni_instance[0]
        input1a = Cni_instance[1:]

        # Relate the phase coordinates to the basis vector using the map
        moment_vector = np.empty(len(input1a), dtype=int)
        for i, l in enumerate(input1a):
            moment_vector[i] = bv_index_map[l]

        #####
        #element_pairings = tools.wick_pairings(moment_vector)
        #####
        # Get all ways to partition the indices into pairs
        coeff_sum = wick_out_do_not_store_looping_pattern_numba_kernel(moment_vector, Anv, cache_wick_partitions[len(moment_vector)])

        return coeff_sum*coef

    @staticmethod
    def wick_out_do_not_store_looping_pattern_do_not_use_symbols(Cni_instance, Anv):
        """
        Input:
        - ar: tuple (The basis vector array, the coefficient)
        - Anv: The inverse of the A matrix
        Output:
        - A calculation of the wick element for each coupling
        """
        # TODO soon: use numba
        #####
        #test, coeff = tools.wick_coupling_mat(Cni_instance, xb)
        #####
        coef = Cni_instance[0]
        moment_vector = Cni_instance[1]

        #####
        #element_pairings = tools.wick_pairings(moment_vector)
        #####
        # Get all ways to partition the indices into pairs
        coeff_sum = wick_out_do_not_store_looping_pattern_numba_kernel(moment_vector, Anv, cache_wick_partitions[len(moment_vector)])

        return coeff_sum*coef

    @staticmethod
    def W(Cni, Amat, xb):
        """
        This is the function that we call the W function in the paper. It is what does the complete Wick coupling calculation
        """
        Anv = np.linalg.inv(Amat)
        # TODO: later - precompute Anv before calling W given that W is called many times

        elm = 0
        for i in Cni: # Can change Cni to tqdm(Cni) to get a progress bar
            a = tools.wick_coupling_mat_list(i, xb)
            elm += tools.wick_out(a, Anv)
        return elm

    """
    We can also simplify the calculation of Wick's theorem by using the Hafnian, instead of the approach used above.
    Hence, bellow are functions for performing calculations via the Hafnian
    """

    @staticmethod
    def multiply_location_vectors(v1,v2):
        """
        Define a function that multiplies two location vectors
        """
        v1v2 = []
        for i in v1:
            for j in v2:
                loc = np.append(i[1],j[1])
                coef = i[0]*j[0]
                v1v2.append([coef,loc])
        return v1v2

    @staticmethod
    def multiply_location_vectors_n(v):
        """
        Define a function that multiplies n location vectors
        """

        vm = []
        i = 0
        while i < len(v):
            if i == 0:
                vm = tools.multiply_location_vectors(v[i],v[i+1])
                i += 2
            else:
                vm = tools.multiply_location_vectors(vm,v[i])
                i += 1

        # Sorting the location vectors
        for i in vm:
            i[1] = np.sort(i[1])

        # To clean up so that we don't have multiples of vectors that are the same
        i = 0
        j = 1
        q = 1
        while q > 0:
            i = 0
            j = 1
            q = 0
            while i < len(vm):
                j = i + 1
                while j < len(vm):
                    if np.array_equal(vm[i][1],vm[j][1]):
                        vm[i][0] = vm[i][0] + vm[j][0]
                        vm.pop(j)
                        q += 1
                    j+=1
                i+=1

        return vm

    @staticmethod
    def construct_sub_A(Atst, unique_elements):
        """
        A function that constructs the sub matrix for given hafnian moment elements
        """
        list_of_combinations = list(set(itertools.combinations_with_replacement(unique_elements, 2)))

        # Differentiate between diagonal and off-diagonal elements
        diag = []
        diag_loc = []
        off_diag = []
        off_diag_loc = []
        for i in list_of_combinations:
            if i[0] == i[1]:
                diag_loc.append([int(i[0]-1),int(i[1]-1)])
                diag.append([i[0],i[1]])
            else:
                off_diag_loc.append([int(i[0]-1),int(i[1]-1)])
                off_diag.append([i[0],i[1]])

        # Making of vector of where each element is located in the final A matric, specificaly for the diagonal elements
        Aloc_diag = []
        for i in diag:
            loc = np.where(unique_elements == i[0])[0][0],np.where(unique_elements == i[1])[0][0]
            Aloc_diag.append(loc)

        # Now making the same vector for the off diagonal elements
        Aloc_off_diag = []
        for i in off_diag:
            loc = np.where(unique_elements == i[0])[0][0],np.where(unique_elements == i[1])[0][0]
            Aloc_off_diag.append(loc)

        # Setting the initial A matrix to zeros
        Afinal_diag = np.zeros((len(unique_elements),len(unique_elements)), dtype = complex)
        Afinal_off_diag = np.zeros((len(unique_elements),len(unique_elements)), dtype = complex)

        # Creating the diagonals of the final sub A matrix from the values of the big A matrix
        for i in range(0,len(diag_loc)):
            # print(Atst[diag_loc[i][0],diag_loc[i][1]])
            Afinal_diag[Aloc_diag[i][0],Aloc_diag[i][1]] = Atst[diag_loc[i][0],diag_loc[i][1]]

        # Doing the same for the off diagonals
        for i in range(0,len(off_diag_loc)):
            Afinal_off_diag[Aloc_off_diag[i][0],Aloc_off_diag[i][1]] = Atst[off_diag_loc[i][0],off_diag_loc[i][1]]

        # Putting it together to product the final sub A matrix
        Afinal = Afinal_diag + Afinal_off_diag + np.transpose(Afinal_off_diag)

        return Afinal

    @staticmethod
    def W_haf(Cni, Amat):
        """
        This is the function that we call the W function in the paper. It is what does the complete Wick coupling calculation
        """
        Anv = np.linalg.inv(Amat)

        elm = 0
        for i in Cni: # Can change Cni to tqdm(Cni) to get a progress bar
            unique_elements, counts = np.unique(i[1], return_counts=True)
            Asub = tools.construct_sub_A(Amat, unique_elements)
            elm += i[0]*tw.hafnian_repeated(Asub, counts)

        return elm

class TMSV:

    def __init__(self, param=TYP_PARAMS):

        self.params = param

        self.status = 0
        self.results = {
            "covariance_matrix": None,
            "k_function_matrix": None,
            "loss_matrix": None,
            "bsm_matrix": None,
            "output_state": None,
            "output_state_basis": None,
            "fidelity": None,
            "hashing_bound": None,
            "probability_success": None,
            "heralding_probability": None,
            "Gamma": None,
        }

    def run(self):
        self.calculate_covariance_matrix()

        # Calculate k function matrix
        self.calculate_k_function_matrix()
        self.calculate_loss_matrix()

    def calculate_covariance_matrix(self):
        """
        Calculates the covariance matrix for the TMSV state, and outputs it in qqpp form
        """
        # The initial TMSV covariance matrix, in the qpqp ordering
        covar = TMSV.tmsv_covar(self.params["mean_photon"])

        ## Go from qpqp to qqpp
        permutation_indices = np.array([0, 2, 1, 3])
        permute_matrix = tools.permutation_matrix(permutation_indices)
        covar2 = (
            permute_matrix @ covar @ np.transpose(permute_matrix)
        )

        self.results["covariance_matrix"] = covar2

    def calculate_k_function_matrix(self):
        """
        Calculates the portion of the matrix that arrises due to the k-function
        """
        # Calculating the K function portion of the A matrix
        Gamma = self.results["covariance_matrix"] + 0.5 * np.eye(
            self.results["covariance_matrix"].shape[0]
        )

        self.results["Gamma"] = Gamma
        Gamma_inverse = np.linalg.inv(Gamma)

        matrix_size = int(Gamma_inverse.shape[1] / 2)
        Ap = Gamma_inverse[:matrix_size, :matrix_size]
        Cp = Gamma_inverse[:matrix_size, matrix_size:]
        Bp = Gamma_inverse[matrix_size:, matrix_size:]

        script_B = (0.5)*np.block(
            [
                [Ap + 0.5*(1j)*(Cp + Cp.T), Cp - 0.5*(1j)*(Ap - Bp)],
                [Cp.T - 0.5*(1j)*(Ap - Bp),Bp - 0.5*(1j)*(Cp + Cp.T),
                ],
            ]
        )

        self.results["k_function_matrix"] = block_diag(script_B, script_B.conjugate())
        return block_diag(script_B, script_B.conjugate())

    def calculate_loss_matrix(self):
        """
        Calculates the portion of the A matrix that arrises due to incorporating loss
        """
        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss(eta_1, eta_2):
            G = np.zeros((8, 8), dtype=np.complex128)
            eta = np.array([eta_1, eta_2])

            for i in range(2):
                G[i, i+4] = (eta[i] - 1)
                G[i, i+6] = (-1j)*(eta[i] - 1)
                G[i+2, i+4] = (1j)*(eta[i] - 1)
                G[i+2, i+6] = (eta[i] - 1)
            return (1/2)*G + (1/2)*np.transpose(G) + (1/2)*np.eye(8)

        self.results["loss_matrix"] = sub_loss(self.params["detection_efficiency"], self.params["detection_efficiency"])

    def calculate_probability_success(self):
        """
        Calculates the probability of success, which is the probability of generating a photon-photon state with the given parameters
        """
        mds = 2  # Number of modes for our system

        # First, define our basis vector
        x = TMSV.basisvZ(mds)  # Because we corrected the basis vector of our covariance matrix, we need to use a different basis vector for the Wick coupling

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_matrix()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_matrix"])
        Gam = self.results["Gamma"]

        N1 = ((self.params["detection_efficiency"])**2)
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (N1)/(D1 * D2 * D3)

        C = TMSV.moment_vector(1)

        self.results["probability_success"] = Coef*tools.W(C,nA,x)
        return self.results["probability_success"]

    @staticmethod
    def moment_vector(n):
        """
        Arguments
        - n: The exponent of the moment to be calculate for the TMSV state
        Output
        - An array of all of the moments that are to be calculated
        """

        mds = 2  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        Ca = ((alp[0] * alp[1])**n)/(math.factorial(n))
        Cb = ((bet[0] * bet[1])**n)/(math.factorial(n))
        C = (Ca * Cb)

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i))

        return Coutf

    """
    Finally, just some generic functions that are needed for the calculations
    """

    # Ordering qpqp
    @staticmethod
    def A(mu):
        """
        Sub-matrix A for the TMSV covariance matrix
        Arguments
        - mu: The mean photon number of the TMSV state
        Output
        - The sub-matrix A for the TMSV covariance matrix
        """
        return np.array([[1 + 2 * mu, 0], [0, 1 + 2 * mu]])

    @staticmethod
    # Ordering qpqp
    def B(mu):
        """
        Sub-matrix B for the TMSV covariance matrix
        Arguments
        - mu: The mean photon number of the TMSV state
        Output
        - The sub-matrix B for the TMSV covariance matrix
        """
        return np.array(
            [[2 * np.sqrt(mu * (mu + 1)), 0], [0, -2 * np.sqrt(mu * (mu + 1))]]
        )

    @staticmethod
    # Ordering qpqp
    def tmsv_covar(mu):
        """
        Construct the covariance matrix for a TMSV state
        Arguments
        - mu: The mean photon number of the TMSV state
        Output
        - The covariance matrix for the TMSV state, in the qpqp ordering
        """
        # Now take the direct sum to construct the covariance matrix
        return (0.5)*np.block(
            [[TMSV.A(mu), TMSV.B(mu)], [TMSV.B(mu), TMSV.A(mu)]]
        )

    @staticmethod
    def basisvZ(mds):
        """
        Arguments
        - mds: the number of modes
        Output
        - The basis vector that cooresponds to that number of modes
        """
        qai = [sp.symbols('qa{}'.format(i)) for i in range(1, mds+1)]
        pai = [sp.symbols('pa{}'.format(i)) for i in range(1, mds+1)]
        qbi = [sp.symbols('qb{}'.format(i)) for i in range(1, mds+1)]
        pbi = [sp.symbols('pb{}'.format(i)) for i in range(1, mds+1)]

        x = qai + pai + qbi + pbi
        return x

    @staticmethod
    def moment_vector_n_signal_m_idler(na,nb,ma,mb):
        """
        Used to calculate the moment vector the any general click pattern of the TMSV state
        Arguments
        - na: The exponent of the alpha variable for the signal mode
        - nb: The exponent of the beta variable for the signal mode
        - ma: The exponent of the alpha variable for the idler mode
        - mb: The exponent of the beta variable for the idler mode
        Output
        - An array of all of the moments that are to be calculated
        """
        mds = 2  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        Ca0 = ((alp[0])**na)/np.sqrt(math.factorial(na))
        Ca1 = ((alp[1])**ma)/np.sqrt(math.factorial(ma))
        Cb0 = ((bet[0])**nb)/np.sqrt(math.factorial(nb))
        Cb1 = ((bet[1])**mb)/np.sqrt(math.factorial(mb))
        C = (Ca0 * Ca1 * Cb0 * Cb1)

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(ZALM.expand_powers_to_symbols(i))

        return Coutf

class SPDC:

    def __init__(self, param=TYP_PARAMS):

        self.params = param
        self.basisv = ZALM.basisvZ(4)  # Because we have 4 modes in the SPDC state
        self.status = 0
        self.results = {
            "covariance_matrix": None,
            "k_function_matrix": None,
            "loss_bsm_matrix": None,
            "bsm_matrix": None,
            "output_state": None,
            "output_state_basis": None,
            "fidelity": None,
            "fidelity_spin_spin": None,
            "hashing_bound": None,
            "probability_success": None,
            "Gamma": None,
            "density_matrix": None,
            "density_matrix_post_bsm": None,
            "fock_pgen": None,
        }

    def run(self):
        self.calculate_covariance_matrix()
        # Calculate k function matrix
        self.calculate_k_function_matrix()

    def calculate_covariance_matrix(self):
        """
        Calculates the covariance matrix for the SPDC state, and outputs it in qqpp form
        """
        # The initial SPDC covariance matrix, in the qpqp ordering
        covar = SPDC.spdc_covar(self.params["mean_photon"])

        ## Go from qpqp to qqpp
        idx_even = []
        idx_odd = []
        for ix in range(8):
            if ix % 2 == 0:
                idx_even = np.append(idx_even, int(ix))
            else:
                idx_odd = np.append(idx_odd, int(ix))

        permutation_indices = np.array(np.append(np.array(idx_even), np.array(idx_odd)))
        permute_matrix = tools.permutation_matrix(permutation_indices)
        covar2 = (
            permute_matrix @ covar @ np.transpose(permute_matrix)
        )

        self.results["covariance_matrix"] = covar2

    def calculate_k_function_matrix(self):
        """
        Calculating the K function portion of the A matrix
        """

        Gamma = self.results["covariance_matrix"] + 0.5 * np.eye(
            self.results["covariance_matrix"].shape[0]
        )

        self.results["Gamma"] = Gamma
        Gamma_inverse = np.linalg.inv(Gamma)

        matrix_size = int(Gamma_inverse.shape[1] / 2)
        Ap = Gamma_inverse[:matrix_size, :matrix_size]
        Cp = Gamma_inverse[:matrix_size, matrix_size:]
        Bp = Gamma_inverse[matrix_size:, matrix_size:]

        script_B = (0.5)*np.block(
            [
                [Ap + 0.5*(1j)*(Cp + Cp.T), Cp - 0.5*(1j)*(Ap - Bp)],
                [Cp.T - 0.5*(1j)*(Ap - Bp),Bp - 0.5*(1j)*(Cp + Cp.T),
                ],
            ]
        )

        self.results["k_function_matrix"] = block_diag(script_B, script_B.conjugate())
        return block_diag(script_B, script_B.conjugate())

    """
    Depending on the parameter that we are calculating, it will have a unique contribution to the A matrix, which is what we calulate in the following functions
    """

    def calculate_loss_matrix_fid(self):
        """
        Calculating the portion of the A matrix that arrises due to incorporating loss, specifically for fidelity calculations
        """

        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss_fid(eta_t, eta_d):
            G = np.zeros((16, 16), dtype=np.complex128)
            eta = np.array([eta_t*eta_d, eta_t*eta_d, eta_t*eta_d, eta_t*eta_d])

            for i in range(4):
                G[i, i+8] = (eta[i] - 1)
                G[i, i+12] = -1j*(eta[i] - 1)
                G[i+4, i+8] = 1j*(eta[i] - 1)
                G[i+4, i+12] = (eta[i] - 1)
            return (0.5)*G + (0.5)*np.transpose(G) + (0.5)*np.eye(16)


        self.results["loss_bsm_matrix"] = sub_loss_fid(self.params["outcoupling_efficiency"], self.params["detection_efficiency"])

    def calculate_loss_bsm_matrix_trace(self):
        """
        Calculating the portion of the A matrix that arrises due to incorporating loss, specifically for the trace of the BSM matrix
        """

        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss_trace(eta_t, eta_d, eta_b):
            G = np.zeros((16, 16), dtype=np.complex128)

            for i in range(4):
                G[i, i+8] = (-1)
                G[i, i+12] = (-1j)*(-1)
                G[i+4, i+8] = (1j)*(-1)
                G[i+4, i+12] = (-1)
            return (0.5)*G + (0.5)*np.transpose(G) + (0.5)*np.eye(16)

        self.results["loss_bsm_matrix"] = sub_loss_trace(self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])


    """
    Functions that calculate parameters of interest for the photon-photon state
    """

    def calculate_probability_success(self):
        """
        Calculates the probability of success, which is the probability of generating a photon-photon state with the given parameters
        """

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_bsm_matrix_trace()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_bsm_matrix"])

        Gam = self.results["Gamma"]

        N1 = ((self.params["bsm_efficiency"])**2)
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (N1)/(D1 * D2 * D3)

        C = 1

        self.results["probability_success"] = Coef #Coef*tools.W(C,nA,self.basisv) #4 * Coef * val(ZALM.moment_vector(self.params["schmidt_coeffs"], 0), nAinv, x)

    def calculate_fidelity(self):
        """
        Calculates the fidelity of the photon-photon state with respect to the Bell state
        """

        # Define the matrix element
        Cn1 = SPDC.moment_vector([1], 1)
        Cn2 = SPDC.moment_vector([1], 2)
        Cn3 = SPDC.moment_vector([1], 3)
        Cn4 = SPDC.moment_vector([1], 4)

        # The loss matrix will be unique for calculating the fidelity
        self.calculate_loss_matrix_fid()
        self.calculate_k_function_matrix()

        nA1 = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]
        Gam = self.results["Gamma"]

        F1 = tools.W(Cn1, nA1, self.basisv)
        F2 = tools.W(Cn2, nA1, self.basisv)
        F3 = tools.W(Cn3, nA1, self.basisv)
        F4 = tools.W(Cn4, nA1, self.basisv)

        N1 = ((self.params["detection_efficiency"]**2)*(self.params["outcoupling_efficiency"]**2))**2
        D1 = np.sqrt(np.linalg.det(nA1))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)

        Coef = (N1)/(2*D1*D2*D3)


        self.results["fidelity"] = Coef*(F1 + F2 + F3 + F4) # np.array([F1, F2, F3, F4, Trc])

    def calculate_rho_nv1_nv2(self, mA, nv1, nv2):
        """
        Arguments
        - mA: The A matrix
        - nv1: The row corresponding to the density matrix element of interest
        - nv2: The collumn corresponding to the density matrix element of interest
        Output
        - The unnormalized density matrix element corresponding to nv1 and nv2
        """
        if self.status == 0:
            self.run()

        nAnv = np.linalg.inv(mA)

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_matrix_fid()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_bsm_matrix"])

        Gam = self.results["Gamma"]

        etab = self.params["bsm_efficiency"]

        # For the case of no dark clicks
        N1 = ((etab)**2)
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (N1)/(D1 * D2 * D3)

        return  Coef*ZALM.dmijpp(nAnv, nv1, nv2) # This is the unnormalized density matrix element for the photon-photon density matrix

    """
    Functions for calculating the spin-spin state
    """

    def calculate_density_operator(self, nvec):
        """
        Arguments
        - nvec: The vector of the number of photons in each mode, which is used to calculate the density matrix
        Output
        - The element of the spin-spin density matrix cooresponding to the nvec
        """
        if self.status == 0:
            self.run()

        lmat = 4  # Number of modes for our system
        mat = np.zeros((lmat, lmat), dtype=np.complex128)

        # Set the A matrix
        self.calculate_loss_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]
        nAnv = np.linalg.inv(nA)

        for i in range(lmat):
            for j in range(lmat):
                mat[i, j] = SPDC.dmijZ(self, i, j, nAnv, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"])

        Gam = self.results["Gamma"]
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)

        Coef = (1)/(4*D1*D2*D3)

        self.results["output_state"] = Coef*mat # This is the unnormalized density matrix


    @staticmethod
    def dmijZ(self, dmi, dmj, nAinv, nvec, eta_t, eta_d):
        """
        Arguments:
        - nAinv: The numerical inverse of the A matrix
        - lamvec: The vectors of lambdas for the system
        - dmi: The row number for the cooresponding density matrix element
        - dmj: The collumn number for the cooresponding density matrix element
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        Output:
        - The density matrix element for the ZALM source
        """

        # Define the matrix element
        #Cn = ZALM.moment_vector_with_memory(lamvec, dmi, dmj, nvec, eta_t, eta_d, eta_b)
        #return ZALM.dmatval_do_not_store_looping_pattern(Cn, nAinv, x)
        Cn = SPDC.moment_vector_with_memory_do_not_convert_to_symbols(dmi, dmj, nvec, eta_t, eta_d)
        return SPDC.dmatval_do_not_store_looping_pattern_do_not_use_symbols(Cn, nAinv, self.basisv)

    @staticmethod
    def moment_vector_with_memory_do_not_convert_to_symbols(dmi, dmj, nvec, eta_t, eta_d):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        """
        C, all_qps = SPDC.moment_vector_with_memory_poly(dmi, dmj, nvec, eta_t, eta_d)
        # assert not any((2 in k) for k in v.as_dict().keys()) # making sure that no powers of 2 are present
        result = [(c,[i for (i,g) in enumerate(gens) if g == 1]) for (gens,c) in C.as_dict().items()]

        return result

    @staticmethod
    def moment_vector_with_memory_poly(dmi, dmj, nvec, eta_t, eta_d):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        """
        mds = 4 # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        _qai = [sp.Symbol("qa{}".format(i)) for i in range(1, mds + 1)]
        _pai = [sp.Symbol("pa{}".format(i)) for i in range(1, mds + 1)]
        _qbi = [sp.Symbol("qb{}".format(i)) for i in range(1, mds + 1)]
        _pbi = [sp.Symbol("pb{}".format(i)) for i in range(1, mds + 1)]
        all_qps = _qai + _pai + _qbi + _pbi
        qai = [sp.Poly(_qai[i], *all_qps, domain='CC') for i in range(mds)]
        pai = [sp.Poly(_pai[i], *all_qps, domain='CC') for i in range(mds)]
        qbi = [sp.Poly(_qbi[i], *all_qps, domain='CC') for i in range(mds)] # NB: we have already taken the complex conjugate
        pbi = [sp.Poly(_pbi[i], *all_qps, domain='CC') for i in range(mds)] # NB: we have already taken the complex conjugate

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + 1j * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - 1j * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_t*eta_d, eta_t*eta_d])

        if dmi == 0:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        if dmj == 0:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))*(1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))*(1/np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        C = Ca*Cb

        return C, all_qps

    @staticmethod
    def dmatval_do_not_store_looping_pattern_do_not_use_symbols(Cni, Anv, xb):
        elm = 0.0
        for i in Cni:
            elm += tools.wick_out_do_not_store_looping_pattern_do_not_use_symbols(i,Anv)
        return elm



    def calculate_fidelity_spin_spin_old(self):

        nvec = [0,1,0,1]
        # Define the matrix element, first the numerators
        Cn1n = SPDC.moment_vector_fid(2, 2, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)
        Cn2n = SPDC.moment_vector_fid(2, 3, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)
        Cn3n = SPDC.moment_vector_fid(3, 2, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)
        Cn4n = SPDC.moment_vector_fid(3, 3, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)

        # Now the denominators
        Cn1d = SPDC.moment_vector_fid(1, 1, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)
        Cn2d = Cn1n
        Cn3d = Cn4n
        Cn4d = SPDC.moment_vector_fid(4, 4, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], 1)


        # The loss matrix will be unique for calculating the fidelity
        self.calculate_loss_matrix_fid()
        self.calculate_k_function_matrix()

        nA1 = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]
        Gam = self.results["Gamma"]

        F1n = tools.W(Cn1n, nA1, self.basisv)
        F2n = tools.W(Cn2n, nA1, self.basisv)
        F3n = tools.W(Cn3n, nA1, self.basisv)
        F4n = tools.W(Cn4n, nA1, self.basisv)

        F1d = tools.W(Cn1d, nA1, self.basisv)
        F2d = F1n
        F3d = F4n
        F4d = tools.W(Cn4d, nA1, self.basisv)

        D1 = np.sqrt(np.linalg.det(nA1))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)

        Coef = (1)/(8*D1*D2*D3)


        self.results["fidelity_spin_spin"] = Coef*(F1n - F2n - F3n + F4n)/(F1d + F2d + F3d + F4d) # np.array([F1, F2, F3, F4, Trc])

    @staticmethod
    def dmijZ_old(self, dmi, dmj, nAinv, nvec, eta_t, eta_d):
        """
        Arguments:
        - nAinv: The numerical inverse of the A matrix
        - lamvec: The vectors of lambdas for the system
        - dmi: The row number for the cooresponding density matrix element
        - dmj: The collumn number for the cooresponding density matrix element
        - nvec: The vector of the number of photons in each mode
        - eta_t: The loss in transmission
        - eta_d: The detection efficiency
        Output:
        - The density matrix element for the SPDC source, not including coefficients
        """

        # Define the matrix element
        Cn = SPDC.moment_vector_with_memory(dmi, dmj, nvec, eta_t, eta_d)

        return SPDC.dmatval(Cn, nAinv, self.basisv)

    @staticmethod
    def moment_vector_with_memory_old(dmi, dmj, nvec, eta_t, eta_d):
        """
        Arguments
        - dmi: The row cooresponding to the density matrix element of interest
        - dmj: The collumn cooresponding to the density matrix element of interest
        - nvec: The vector of the number of photons in each mode, which is used to calculate the density matrix
        - eta_t: The loss in transmission
        - eta_d: The detection efficiency
        Output
        - An array of all of the moments that are to be calculated
        """
        mds = 4  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_t*eta_d, eta_t*eta_d])

        if dmi == 0:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        if dmj == 0:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        C = Ca*Cb

        # TODO: Change so that this matches the ZALM moment_vector_with_memory that uses tuples instead of lists

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols_list(i))

        return Coutf

    @staticmethod
    def moment_vector_fid(Ai, Bj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - Ai: The index of the alpha variable for the signal mode
        - Bj: The index of the beta variable for the idler mode
        - nvec: The vector of the number of photons in each mode, which is used to calculate the density matrix
        - eta_t: The loss in transmission
        - eta_d: The detection efficiency
        - eta_b: The BSM efficiency
        Output
        - The moment vector for calculating a specific term of the fidelity
        """
        mds = 4  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

        if Ai == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif Ai == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif Ai == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif Ai == 4:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[2]*np.sqrt(etav[2]) + alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Ca4 = ((alp[2]*np.sqrt(etav[2]) - alp[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        if Bj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif Bj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif Bj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif Bj == 4:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[2]*np.sqrt(etav[2]) + bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[2])
            Cb4 = ((bet[2]*np.sqrt(etav[2]) - bet[3]*np.sqrt(etav[3]))/(np.sqrt(2)))**(nvec[3])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        C = Ca*Cb

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i))

        return Coutf

    @staticmethod
    def dmatval_old(Cni, Anv, xb):
        """
        Arguments
        - Cni: The moment vector for the density matrix element
        - Anv: The inverse of the A matrix
        - xb: The basis vector that corresponds to the A matrix
        Output
        - The unnormalized density matrix element corresponding to the Cni and Anv
        """

        elm = 0
        for i in Cni:
            a = tools.wick_coupling_mat_list(i,xb)
            elm += tools.wick_out(a, Anv)
        return elm




    """
    Defining helper functions that are used
    """
    @staticmethod
    def spdc_covar(mu):
        """
        Arguments
        - mu: The mean photon number of the SPDC source
        Output
        - The covariance matrix of the SPDC source in the qpqp ordering
        """

        # Ordering qpqp
        # Mode swap for polarization entnaglement
        permutation_indices = [0, 1, 6, 7, 4, 5, 2, 3]
        permute_matrix = tools.permutation_matrix(permutation_indices)
        return (
            permute_matrix
            @ block_diag(TMSV.tmsv_covar(mu), TMSV.tmsv_covar(mu))
            @ np.transpose(permute_matrix)
        )

    @staticmethod
    def moment_vector(lambda_vector, l):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - l - An array of the exponents for the various moment variables
        Output
        - An array of all of the moments that are to be calculated
        """
        mds = 8 * len(lambda_vector)  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        ms = tools.mcomb(len(lambda_vector))

        Ca1 = alp[0] * alp[3]
        Ca2 = alp[1] * alp[2]
        Cb1 = bet[0] * bet[3]
        Cb2 = bet[1] * bet[2]


        if l == 0:
            C = alp[2] * alp[3] * bet[2] * bet[3]
        elif l == 1:
            C = Ca1 * Cb1
        elif l == 2:
            C = Ca1 * Cb2
        elif l == 3:
            C = Ca2 * Cb1
        elif l == 4:
            C = Ca2 * Cb2

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i))

        return Coutf

class ZALM:

    def __init__(self, param=TYP_PARAMS):

        self.params = param
        self.basisv = ZALM.basisvZ(8)
        self.status = 0
        self.results = {
            "covariance_matrix": None,
            "k_function_matrix": None,
            "loss_bsm_matrix": None,
            "bsm_matrix": None,
            "output_state": None,
            "output_state_basis": None,
            "fidelity": None,
            "hashing_bound": None,
            "probability_success": None,
            "Gamma": None,
            "density_matrix": None,
            "density_matrix_post_bsm": None,
            "fock_pgen": None,
        }

    def run(self):
        self.calculate_covariance_matrix()
        # Calculate k function matrix
        self.calculate_k_function_matrix()

    def calculate_covariance_matrix(self):
        """
        Calculating the covariance matrix of the single-mode ZALM source
        """
        # The initial ZALM covariance matrix, in the qpqp ordering
        covar = None
        covar = block_diag(SPDC.spdc_covar(self.params["mean_photon"]), SPDC.spdc_covar(self.params["mean_photon"]))

        ## Go from qpqp to qqpp
        idx_even = []
        idx_odd = []
        for ix in range(16):
            if ix % 2 == 0:
                idx_even = np.append(idx_even, int(ix))
            else:
                idx_odd = np.append(idx_odd, int(ix))

        permutation_indices = np.array(np.append(np.array(idx_even), np.array(idx_odd)))
        permute_matrix = tools.permutation_matrix(permutation_indices)
        covar2 = (
            permute_matrix @ covar @ np.transpose(permute_matrix)
        )

        # Applying the symplective matrices that represent 50/50 beampslitters between the bell state modes
        Id2 = np.identity(2)
        St35 = np.array([[np.sqrt(0.5), 0, np.sqrt(0.5), 0],[0,1,0,0],
                    [-np.sqrt(0.5), 0, np.sqrt(0.5),0],[0,0,0,1]])
        St46 = np.array([[1, 0, 0, 0],[0,np.sqrt(0.5),0,np.sqrt(0.5)],
                    [0, 0, 1, 0],[0,-np.sqrt(0.5),0,np.sqrt(0.5)]])

        S35 = block_diag(Id2, St35, Id2, Id2, St35, Id2)
        S46 = block_diag(Id2, St46, Id2, Id2, St46, Id2)

        self.results["covariance_matrix"] = S46@S35@covar2@np.transpose(S35)@np.transpose(S46)

    def calculate_k_function_matrix(self):
        """
        Calculating the K function portion of the A matrix for the single-mode ZALM source.
        """
        Gamma = self.results["covariance_matrix"] + 0.5 * np.eye(
            self.results["covariance_matrix"].shape[0]
        )

        self.results["Gamma"] = Gamma
        Gamma_inverse = np.linalg.inv(Gamma)

        matrix_size = int(Gamma_inverse.shape[1] / 2)
        Ap = Gamma_inverse[:matrix_size, :matrix_size]
        Cp = Gamma_inverse[:matrix_size, matrix_size:]
        Bp = Gamma_inverse[matrix_size:, matrix_size:]

        script_B = (0.5)*np.block(
            [
                [Ap + 0.5*(1j)*(Cp + Cp.T), Cp - 0.5*(1j)*(Ap - Bp)],
                [Cp.T - 0.5*(1j)*(Ap - Bp),Bp - 0.5*(1j)*(Cp + Cp.T),
                ],
            ]
        )

        self.results["k_function_matrix"] = block_diag(script_B, script_B.conjugate())
        return block_diag(script_B, script_B.conjugate())

    """
    Depending on the parameter that we are calculating, it will have a unique contribution to the A matrix, which is what we calulate in the following functions
    """

    def calculate_loss_bsm_matrix_pgen(self):
        """
        Calculating the loss portion of the A matrix, specifically when calculating the probability of generation.
        """
        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss_pgen(eta_t, eta_d, eta_b):
            G = np.zeros((32, 32), dtype=np.complex128)
            eta = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

            for i in range(8):
                if i==0 or i==1 or i==6 or i==7:
                    G[i, i+16] = (-1)
                    G[i, i+24] = (-1j)*(-1)
                    G[i+8, i+16] = (1j)*(-1)
                    G[i+8, i+24] = (-1)
                else:
                    G[i, i+16] = (eta[i] - 1)
                    G[i, i+24] = (-1j)*(eta[i] - 1)
                    G[i+8, i+16] = (1j)*(eta[i] - 1)
                    G[i+8, i+24] = (eta[i] - 1)
            return (0.5)*G + (0.5)*np.transpose(G) + (1/2)*np.eye(32)

        self.results["loss_bsm_matrix"] = sub_loss_pgen(self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])
        return sub_loss_pgen(self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])

    def calculate_loss_bsm_matrix_fid(self):
        """
        Calculating the loss portion of the A matrix, specifically when calculating the fidelity
        """

        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss_fid(eta_t, eta_d, eta_b):
            G = np.zeros((32, 32), dtype=np.complex128)
            eta = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

            for i in range(8):
                G[i, i+16] = (eta[i] - 1)
                G[i, i+24] = -1j*(eta[i] - 1)
                G[i+16, i+8] = 1j*(eta[i] - 1)
                G[i+24, i+8] = (eta[i] - 1)
            return (0.5)*G + (0.5)*np.transpose(G) + (0.5)*np.eye(32)


        self.results["loss_bsm_matrix"] = sub_loss_fid(self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])

    def calculate_loss_bsm_matrix_trace(self):
        """
        Calculating the loss portion of the A matrix, specifically when calculating the trace of the matrix
        """

        # Ordering is qqpp in every sub-block; compensated by appropriate basis vector later
        def sub_loss_trace(eta_t, eta_d, eta_b):
            G = np.zeros((32, 32), dtype=np.complex128)
            eta = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

            for i in range(8):
                G[i, i+16] = (-1)
                G[i, i+24] = (-1j)*(-1)
                G[i+8, i+16] = (1j)*(-1)
                G[i+8, i+24] = (-1)
            return G + np.transpose(G) + (1/2)*np.eye(32)

        self.results["loss_bsm_matrix"] = sub_loss_trace(self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])

    """
    Functions that calculate parameters of interest for the photon-photon state
    """

    def calculate_probability_success(self):
        """
        Calculate the probability of success for the photon-photon single-mode ZALM source
        """

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_bsm_matrix_pgen()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_bsm_matrix"])

        Gam = self.results["Gamma"]

        # # For the case of no dark clicks
        # N1 = ((self.params["bsm_efficiency"])**2)
        # D1 = np.sqrt(np.linalg.det(nA))
        # D2 = (np.linalg.det(Gam))**(0.25)
        # D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        # Coef = (N1)/(D1 * D2 * D3)

        # C = ZALM.moment_vector([1], 0)

        # self.results["probability_success"] = Coef*tools.W(C,nA,x) #4 * Coef * val(ZALM.moment_vector(self.params["schmidt_coeffs"], 0), nAinv, x)

        # Including dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        C1 = ZALM.moment_vector(self.params["schmidt_coeffs"], 0)
        C2 = ZALM.moment_vector(self.params["schmidt_coeffs"], 9)
        C3 = ZALM.moment_vector(self.params["schmidt_coeffs"], 10)
        C4 = ZALM.moment_vector(self.params["schmidt_coeffs"], 14)

        Term1 = ((self.params["bsm_efficiency"])**2)*((1 - self.params["dark_counts"])**4)*tools.W(C1,nA,self.basisv)
        Term2 = ((self.params["bsm_efficiency"]))*(self.params["dark_counts"])*((1 - (self.params["dark_counts"]))**3)*tools.W(C2,nA,self.basisv)
        Term3 = ((self.params["bsm_efficiency"]))*(self.params["dark_counts"])*((1 - (self.params["dark_counts"]))**3)*tools.W(C3,nA,self.basisv)
        Term4 = ((self.params["dark_counts"])**2)*((1 - self.params["dark_counts"])**2)*tools.W(C4,nA,self.basisv)

        self.results["probability_success"] = Coef*( Term1 + Term2 + Term3 + Term4 )

    def calculate_fidelity(self):
        """
        Calculate the fidelity with respect to the Bell state for the photon-photon single-mode ZALM source
        """

        # Define the matrix element
        Cn1 = ZALM.moment_vector([1], 1)
        Cn2 = ZALM.moment_vector([1], 2)
        Cn3 = ZALM.moment_vector([1], 3)
        Cn4 = ZALM.moment_vector([1], 4)
        Cn0 = ZALM.moment_vector([1], 0)

        # The loss matrix will be unique for calculating the fidelity
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()

        nA1 = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

        F1 = tools.W(Cn1, nA1, self.basisv)
        F2 = tools.W(Cn2, nA1, self.basisv)
        F3 = tools.W(Cn3, nA1, self.basisv)
        F4 = tools.W(Cn4, nA1, self.basisv)

        # Now calculate the trace of the state, which is equivalent to the probability of generation
        self.calculate_loss_bsm_matrix_pgen()
        self.calculate_k_function_matrix()
        nA2 = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

        N1 = ((self.params["detection_efficiency"])*(self.params["outcoupling_efficiency"]))**2
        N2 = np.sqrt(np.linalg.det(nA2))
        D1 = np.sqrt(np.linalg.det(nA1))
        Coef = (N1*N2)/(2*D1)

        Trc = tools.W(Cn0, nA2, self.basisv)

        self.results["fidelity"] = Coef*(F1 + F2 + F3 + F4)/(Trc) # np.array([F1, F2, F3, F4, Trc])

    def calculate_rho_nv1_nv2(self, mA, nv1, nv2):
        """
        Arguments
        - mA: The A matrix
        - nv1: The row corresponding to the density matrix element of interest
        - nv2: The collumn corresponding to the density matrix element of interest
        Output
        - The unnormalized density matrix element corresponding to nv1 and nv2
        """
        if self.status == 0:
            self.run()

        nAnv = np.linalg.inv(mA)

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_bsm_matrix"])

        Gam = self.results["Gamma"]

        etab = self.params["bsm_efficiency"]

        # For the case of no dark clicks
        N1 = ((etab)**2)
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (N1)/(D1 * D2 * D3)

        return  Coef*ZALM.dmijpp([1], nAnv, nv1, nv2) # This is the unnormalized density matrix element for the photon-photon density matrix


    """
    Functions that use the hafnian approach instead of directly calculating via Wick's theorem

    **Need to re-verify their functionality
    """

    @staticmethod
    def moment_vector_haf(nvec1, nvec2):
        """
        Defining the moment vector, to be used for the Hafnian calculation
        """
        # Define the coherent state variables
        a1 = np.array([np.array([1/np.sqrt(2),1]),np.array([1j/np.sqrt(2),9])])
        a2 = np.array([np.array([1/np.sqrt(2),2]),np.array([1j/np.sqrt(2),10])])
        a3 = np.array([np.array([1/np.sqrt(2),3]),np.array([1j/np.sqrt(2),11])])
        a4 = np.array([np.array([1/np.sqrt(2),4]),np.array([1j/np.sqrt(2),12])])
        a5 = np.array([np.array([1/np.sqrt(2),5]),np.array([1j/np.sqrt(2),13])])
        a6 = np.array([np.array([1/np.sqrt(2),6]),np.array([1j/np.sqrt(2),14])])
        a7 = np.array([np.array([1/np.sqrt(2),7]),np.array([1j/np.sqrt(2),15])])
        a8 = np.array([np.array([1/np.sqrt(2),8]),np.array([1j/np.sqrt(2),16])])
        b1s = np.array([np.array([1/np.sqrt(2),17]),np.array([-1j/np.sqrt(2),25])])
        b2s = np.array([np.array([1/np.sqrt(2),18]),np.array([-1j/np.sqrt(2),26])])
        b3s = np.array([np.array([1/np.sqrt(2),19]),np.array([-1j/np.sqrt(2),27])])
        b4s = np.array([np.array([1/np.sqrt(2),20]),np.array([-1j/np.sqrt(2),28])])
        b5s = np.array([np.array([1/np.sqrt(2),21]),np.array([-1j/np.sqrt(2),29])])
        b6s = np.array([np.array([1/np.sqrt(2),22]),np.array([-1j/np.sqrt(2),30])])
        b7s = np.array([np.array([1/np.sqrt(2),23]),np.array([-1j/np.sqrt(2),31])])
        b8s = np.array([np.array([1/np.sqrt(2),24]),np.array([-1j/np.sqrt(2),32])])

        mvec1 = [a1, a2, a3, a4, a5, a6, a7, a8]
        mvec2 = [b1s, b2s, b3s, b4s, b5s, b6s, b7s, b8s]
        mlv = []
        for i in range(0,len(nvec1)):
            if nvec1[i] > 0:
                j = 0
                while j < nvec1[i]:
                    mlv.append(mvec1[i])
                    j += 1
        for i in range(0,len(nvec2)):
            if nvec2[i] > 0:
                j = 0
                while j < nvec2[i]:
                    mlv.append(mvec2[i])
                    j += 1

        mlv_tst = tools.multiply_location_vectors_n(mlv)

        return mlv_tst

    def calculate_rho_nv1_nv2_haf(self, Am, nv1, nv2):
        """
        Arguments
        - mA: The A matrix
        - nv1: The row corresponding to the density matrix element of interest
        - nv2: The collumn corresponding to the density matrix element of interest
        Output
        - The unnormalized density matrix element corresponding to nv1 and nv2, calculated via the hafnian approach
        """
        if self.status == 0:
            self.run()

        # Set the C coefficient
        Cn = ZALM.moment_vector_haf(nv1, nv2)

        # Calculate the detection-pattern dependent coefficient
        def Cof(et, ed, eb, n1, n2):
            etv = np.array([np.sqrt(et*ed), np.sqrt(et*ed), np.sqrt(eb), np.sqrt(eb), np.sqrt(eb), np.sqrt(eb), np.sqrt(et*ed), np.sqrt(et*ed)])

            ca = 1
            cb = 1
            for i in range(0,8):
                ca *= ((etv[i])**n1[i])/(np.sqrt(math.factorial(n1[i])))
                cb *= ((etv[i])**n2[i])/(np.sqrt(math.factorial(n2[i])))
            return ca*cb

        etat = self.params["outcoupling_efficiency"]
        etad = self.params["detection_efficiency"]
        etab = self.params["bsm_efficiency"]

        # The loss matrix will be unique for calculating the probability of generation
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()

        nA = (self.results["k_function_matrix"] + self.results["loss_bsm_matrix"])

        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        N1 = ((etab)**2)
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (N1)/(D1 * D2 * D3)

        return  Coef*Cof(etat, etad, etab, nv1, nv2)*tools.W_haf(Cn, Am) # This is the unnormalized density matrix element


    """
    Functions for calculating the complete photon-photon density matrix up to 4 photons
    """
    @staticmethod
    def n_photons_in_m_modes(n, m):
        """
        This function calculates the basis elements for a basis vector of n photons in m modes
        """
        def combinations_with_replacement(n, k):
            return list(itertools.combinations_with_replacement(range(n), k))

        n_bins = m
        n_balls = n
        result = combinations_with_replacement(n_bins, n_balls)
        b = []
        for combo in result:
            vec = np.zeros(8, dtype=int)
            for bin_idx in combo:
                vec[bin_idx] += 1
            b.append(vec)
        return b

    @staticmethod
    def density_matrix_elms(n):
        """
        Creates a density matrix basis vector for n photons in 8 modes
        """

        for i in range(0,n+1):
            if i == 0:
                # Configurations with 0 photons in 8 modes
                b = ZALM.n_photons_in_m_modes(0, 8)
            else:
                # Appending vector for Configurations with i photons in 8 modes
                b = b + ZALM.n_photons_in_m_modes(i, 8)

        bra_indices = []
        ket_indices = []
        bra_states = []
        ket_states = []

        # Iterate through all combinations of basis states
        for i in range(len(b)):
            for j in range(len(b)):
                bra_indices.append(i)
                ket_indices.append(j)
                bra_states.append(b[i])
                ket_states.append(b[j])

        # Create the DataFrame
        dm = pd.DataFrame({
            'bra_index': bra_indices,
            'ket_index': ket_indices,
            'bra_state': bra_states,
            'ket_state': ket_states
        })

        return dm

    @staticmethod
    def density_matrix_elms_post_detection(n):
        """
        Creates a density matrix basis vector for n photons in 8 modes following BSM
        """

        # Constructing the complete basis vector with only the BSM successes
        for i in range(0,n+1):
            if i == 0:
                # Configurations with 0 photons in 8 modes
                bi = ZALM.n_photons_in_m_modes(0, 8)
                b = []
                for j in range(0,len(bi)):
                    if bi[j][2] == 1 and bi[j][3] == 1 and bi[j][4] == 0 and bi[j][5] == 0:
                        b.append(bi[j])
            else:
                # Configurations with 0 photons in 8 modes
                bi = ZALM.n_photons_in_m_modes(i, 8)
                bg = []
                for j in range(0,len(bi)):
                    if bi[j][2] == 1 and bi[j][3] == 1 and bi[j][4] == 0 and bi[j][5] == 0:
                        bg.append(bi[j])

                # Appending vector for Configurations with i photons in 8 modes
                b = b + bg

        bra_indices = []
        ket_indices = []
        bra_states = []
        ket_states = []

        # Iterate through all combinations of basis states
        for i in range(len(b)):
            for j in range(len(b)):
                bra_indices.append(i)
                ket_indices.append(j)
                bra_states.append(b[i])
                ket_states.append(b[j])

        # Create the DataFrame
        dm = pd.DataFrame({
            'bra_index': bra_indices,
            'ket_index': ket_indices,
            'bra_state': bra_states,
            'ket_state': ket_states
        })

        return dm

    @staticmethod
    def get_density_matrix_bv_element(dmat, bra_idx, ket_idx):
        """
        Extract a specific density matrix element from the DataFrame

        Parameters:
        dmat (DataFrame): DataFrame containing the density matrix elements
        bra_idx (int): Index of the bra state
        ket_idx (int): Index of the ket state

        Returns:
        tuple: (bra_state, ket_state) for the requested indices
        """
        element = dmat[(dmat['bra_index'] == bra_idx) & (dmat['ket_index'] == ket_idx)]
        if len(element) == 0:
            return None
        return (element['bra_state'].iloc[0], element['ket_state'].iloc[0])

    def calculate_density_matrix(self):
        """
        Calculate the photon-photon density matrix up to 4 photons
        """
        rho = np.zeros((495, 495), dtype=complex)

        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

         # Calculating the coefficients
        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        for i in tqdm(range(0,495)):
            for j in tqdm(range(0,495)):
                n1 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, j)[0]
                n2 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, j)[1]

                if (np.sum(n1) + np.sum(n2))%2 == 1:
                    rho[i,j] = 0
                else:
                    rho[i,j] = Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

        self.results["density_matrix"] = rho
        return rho

    def calculate_density_matrix_post_bsm(self, n):
        """
        Calculate the photon-photon density matrix up to n photons
        """

        dmat_size = np.max(ZALM.density_matrix_elms_post_detection(n)['bra_index'])

        rho = np.zeros((dmat_size+1, dmat_size+1), dtype=complex)

        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

         # Calculating the coefficients
        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        for i in range(0,dmat_size):
            for j in range(0,dmat_size):
                n1 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms_post_detection(n), i, j)[0]
                n2 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms_post_detection(n), i, j)[1]

                if (np.sum(n1) + np.sum(n2))%2 == 1:
                    rho[i,j] = 0
                else:
                    rho[i,j] = Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

        self.results["density_matrix_post_bsm"] = rho
        return rho

    def calculate_fock_trace_haf(self):
        """
        Calculate the trace of the photon-photon density matrix up to 4 photons using the Hafnian
        """
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

         # Calculating the coefficients
        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        Pgen = 0

        for i in tqdm(range(0,495)):
            n1 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[0]
            n2 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[1]
            Pgen += Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

            # if (np.sum(n1) + np.sum(n2))%2 == 1:
            #     Pgen += 0
            # else:
            #     Pgen += Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

        self.results["fock_pgen"] = Pgen
        return Pgen

    def calculate_fock_trace(self):
        """
        Calculate the trace of the photon-photon density matrix up to 4 photons using our built Wick functions
        """
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

         # Calculating the coefficients
        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        Pgen = 0

        for i in tqdm(range(0,495)):
            n1 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[0]
            n2 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[1]
            Pgen += Coef*self.calculate_rho_nv1_nv2(nA, n1, n2)

            # if (np.sum(n1) + np.sum(n2))%2 == 1:
            #     Pgen += 0
            # else:
            #     Pgen += Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

        self.results["fock_pgen"] = Pgen
        return Pgen

    def calculate_fock_diag_haf(self):
        """
        Calculate the diagonal elements of the photon-photon density matrix up to 4 photons using the Hafnian and place them in an array
        """
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]

         # Calculating the coefficients
        Gam = self.results["Gamma"]

        # For the case of no dark clicks
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        diag = []
        click = []

        for i in tqdm(range(0,495)):
            n1 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[0]
            n2 = ZALM.get_density_matrix_bv_element(ZALM.density_matrix_elms(4), i, i)[1]
            diag_lm = Coef*self.calculate_rho_nv1_nv2_haf(nA, n1, n2)

            diag.append(diag_lm)
            click.append([n1,n2])

        return [diag,click]

    """
    Functions that calculate parameters of interest for the spin-spin state
    """

    def calculate_density_operator(self, nvec):
        """
        Arguments
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        Output
        - The numerical complete spin density matrix
        """
        if self.status == 0:
            self.run()

        lmat = 4  # Number of modes for our system
        mat = np.zeros((lmat, lmat), dtype=np.complex128)

        # Set the A matrix
        self.calculate_loss_bsm_matrix_fid()
        self.calculate_k_function_matrix()
        nA = self.results["k_function_matrix"] + self.results["loss_bsm_matrix"]
        nAnv = np.linalg.inv(nA)

        Gam = self.results["Gamma"]
        D1 = np.sqrt(np.linalg.det(nA))
        D2 = (np.linalg.det(Gam))**(0.25)
        D3 = (np.linalg.det(np.conjugate(Gam)))**(0.25)
        Coef = (1)/(D1 * D2 * D3)

        for i in range(lmat):
            for j in range(lmat):
                mat[i, j] = ZALM.dmijZ(self, i, j, nAnv, nvec, self.params["outcoupling_efficiency"], self.params["detection_efficiency"], self.params["bsm_efficiency"])

        self.results["output_state"] = Coef*mat # This is the unnormalized density matrix

    @staticmethod
    def dmijZ(self, dmi, dmj, nAinv, nvec, eta_t, eta_d, eta_b):
        """
        Arguments:
        - nAinv: The numerical inverse of the A matrix
        - lamvec: The vectors of lambdas for the system
        - dmi: The row number for the cooresponding density matrix element
        - dmj: The collumn number for the cooresponding density matrix element
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        Output:
        - The density matrix element for the ZALM source
        """

        # Define the matrix element
        #Cn = ZALM.moment_vector_with_memory(lamvec, dmi, dmj, nvec, eta_t, eta_d, eta_b)
        #return ZALM.dmatval_do_not_store_looping_pattern(Cn, nAinv, x)
        Cn = ZALM.moment_vector_with_memory_do_not_convert_to_symbols(dmi, dmj, nvec, eta_t, eta_d, eta_b)
        return ZALM.dmatval_do_not_store_looping_pattern_do_not_use_symbols(Cn, nAinv, self.basisv)


    @staticmethod
    def dmijZ_old(self, lamvec, dmi, dmj, nAinv, nvec, eta_t, eta_d, eta_b):
        """
        Arguments:
        - nAinv: The numerical inverse of the A matrix
        - lamvec: The vectors of lambdas for the system
        - dmi: The row number for the cooresponding density matrix element
        - dmj: The collumn number for the cooresponding density matrix element
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        Output:
        - The density matrix element for the ZALM source
        """

        # Define the matrix element
        Cn = ZALM.moment_vector_with_memory(lamvec, dmi, dmj, nvec, eta_t, eta_d, eta_b)

        #return ZALM.dmatval(Cn, nAinv, x)
        return ZALM.dmatval(Cn, nAinv, self.basisv)

    @staticmethod
    def moment_vector_with_memory_old(lambda_vector,dmi, dmj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        Output
        - A tuple of all moments to be calculated when using a Duan-Kimble style quantum memory
        """
        mds = 8 * len(lambda_vector)  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

        # Calculate Ca based on dmi value
        if dmi == 0:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        # Calculate Cb based on dmj value
        if dmj == 0:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        # Calculate Cd terms
        Cd3 = (alp[2]*bet[2]*etav[2])**(nvec[2])/math.factorial(nvec[2])
        Cd4 = (alp[3]*bet[3]*etav[3])**(nvec[3])/math.factorial(nvec[3])
        Cd5 = (alp[4]*bet[4]*etav[4])**(nvec[4])/math.factorial(nvec[4])
        Cd6 = (alp[5]*bet[5]*etav[5])**(nvec[5])/math.factorial(nvec[5])
        C = Cd3*Cd4*Cd5*Cd6*Ca*Cb

        # # Process terms and convert to tuples
        # C = sp.expand(C)
        # Cv = tuple(C.as_ordered_terms())

        # # Create result with structured tuples
        # result = []
        # for term in Cv:
        #     # Extract the numeric coefficient from the term
        #     coef, symbolic_part = term.as_coeff_mul()

        #     # Process symbolic factors and combine all numeric elements
        #     numeric_coef = coef
        #     symbolic_factors = []

        #     for factor in symbolic_part:
        #         if factor.is_number or factor == sp.I or factor == -sp.I:
        #             numeric_coef *= factor
        #         else:
        #             symbolic_factors.append(factor)

        #     # Expand powers in symbolic factors
        #     expanded_symbols = tools.expand_powers_to_symbols(symbolic_factors)

        #     # Create tuple with numeric coefficient first
        #     result_tuple = (numeric_coef,) + tuple(expanded_symbols)
        #     result.append(result_tuple)

            # Convert polynomial expression to regular expression and expand
        C_expanded = expand(C)

        # Process terms to create the structured tuples
        result = []
        for term in C_expanded.as_ordered_terms():
            coef, rest = term.as_coeff_mul()

            # Collect all numeric factors into coefficient
            symbolic_factors = []
            for factor in rest:
                if factor.is_number:
                    coef *= factor
                else:
                    symbolic_factors.append(factor)

            # Expand powers
            expanded_symbols = []
            for factor in symbolic_factors:
                if isinstance(factor, sp.Pow) and factor.exp.is_Integer:
                    expanded_symbols.extend([factor.base] * int(factor.exp))
                else:
                    expanded_symbols.append(factor)

            # Create result tuple with numeric coefficient first
            result_tuple = (coef,) + tuple(expanded_symbols)
            result.append(result_tuple)

        return tuple(result)

    @staticmethod
    def moment_vector_with_memory(dmi, dmj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        """
        C, all_qps = ZALM.moment_vector_with_memory_poly(dmi, dmj, nvec, eta_t, eta_d, eta_b)
        # assert not any((2 in k) for k in v.as_dict().keys()) # making sure that no powers of 2 are present
        result = [(c,*[s for (g,s) in zip(gens,all_qps) if g == 1]) for (gens,c) in C.as_dict().items()]

        return result

    @staticmethod
    def moment_vector_with_memory_do_not_convert_to_symbols(dmi, dmj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        """
        C, all_qps = ZALM.moment_vector_with_memory_poly(dmi, dmj, nvec, eta_t, eta_d, eta_b)
        # assert not any((2 in k) for k in v.as_dict().keys()) # making sure that no powers of 2 are present
        result = [(c,[i for (i,g) in enumerate(gens) if g == 1]) for (gens,c) in C.as_dict().items()]

        return result

    @staticmethod
    def moment_vector_with_memory_poly(dmi, dmj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row corresponding to the density matrix element of interest
        - dmj: The column corresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        """
        mds = 8 # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        _qai = [sp.Symbol("qa{}".format(i)) for i in range(1, mds + 1)]
        _pai = [sp.Symbol("pa{}".format(i)) for i in range(1, mds + 1)]
        _qbi = [sp.Symbol("qb{}".format(i)) for i in range(1, mds + 1)]
        _pbi = [sp.Symbol("pb{}".format(i)) for i in range(1, mds + 1)]
        all_qps = _qai + _pai + _qbi + _pbi
        qai = [sp.Poly(_qai[i], *all_qps, domain='CC') for i in range(mds)]
        pai = [sp.Poly(_pai[i], *all_qps, domain='CC') for i in range(mds)]
        qbi = [sp.Poly(_qbi[i], *all_qps, domain='CC') for i in range(mds)] # NB: we have already taken the complex conjugate
        pbi = [sp.Poly(_pbi[i], *all_qps, domain='CC') for i in range(mds)] # NB: we have already taken the complex conjugate

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + 1j * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - 1j * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

        # Calculate Ca based on dmi value
        if dmi == 0:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        # Calculate Cb based on dmj value
        if dmj == 0:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))* (1/np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))* (1/np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        # Calculate Cd terms
        Cd3 = (alp[2]*bet[2]*etav[2])**(nvec[2])/math.factorial(nvec[2])
        Cd4 = (alp[3]*bet[3]*etav[3])**(nvec[3])/math.factorial(nvec[3])
        Cd5 = (alp[4]*bet[4]*etav[4])**(nvec[4])/math.factorial(nvec[4])
        Cd6 = (alp[5]*bet[5]*etav[5])**(nvec[5])/math.factorial(nvec[5])
        C = Cd3*Cd4*Cd5*Cd6*Ca*Cb
        return C, all_qps

    @staticmethod
    def moment_vector_with_memory_list(lambda_vector, dmi, dmj, nvec, eta_t, eta_d, eta_b):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - dmi: The row cooresponding to the density matrix element of interest
        - dmj: The collumn cooresponding to the density matrix element of interest
        - nvec: The vector of n_i's for the system, where n_i is the number of photons in mode i
        - eta_t: The transmission efficiency
        - eta_d: The detection efficiency
        - eta_b: The Bell state measurement efficiency
        Output
        - An array of all of the moments that are to be calculated when using a Duan-Kimble style quantum memory

        """
        mds = 8 * len(lambda_vector)  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        etav = np.array([eta_t*eta_d, eta_t*eta_d, eta_b, eta_b, eta_b, eta_b, eta_t*eta_d, eta_t*eta_d])

        if dmi == 0:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 1:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 2:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        elif dmi == 3:
            Ca1 = ((alp[0]*np.sqrt(etav[0]) + alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Ca2 = ((alp[0]*np.sqrt(etav[0]) - alp[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Ca3 = ((alp[6]*np.sqrt(etav[6]) + alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Ca4 = ((alp[6]*np.sqrt(etav[6]) - alp[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Ca = Ca1*Ca2*Ca3*Ca4
        else:
            Ca = 1

        if dmj == 0:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 1:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 2:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        elif dmj == 3:
            Cb1 = ((bet[0]*np.sqrt(etav[0]) + bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[0])
            Cb2 = ((bet[0]*np.sqrt(etav[0]) - bet[1]*np.sqrt(etav[1]))/(np.sqrt(2)))**(nvec[1])
            Cb3 = ((bet[6]*np.sqrt(etav[6]) + bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[6])
            Cb4 = ((bet[6]*np.sqrt(etav[6]) - bet[7]*np.sqrt(etav[7]))/(np.sqrt(2)))**(nvec[7])
            Cb = Cb1*Cb2*Cb3*Cb4
        else:
            Cb = 1

        Cd3 = (alp[2]*bet[2]*etav[2])**(nvec[2])/math.factorial(nvec[2])
        Cd4 = (alp[3]*bet[3]*etav[3])**(nvec[3])/math.factorial(nvec[3])
        Cd5 = (alp[4]*bet[4]*etav[4])**(nvec[4])/math.factorial(nvec[4])
        Cd6 = (alp[5]*bet[5]*etav[5])**(nvec[5])/math.factorial(nvec[5])
        C = Cd3*Cd4*Cd5*Cd6*Ca*Cb

        ## The old Approach

        # # Currently only set up for the single-zalm case
        # ms = ZALM.mcomb(2)

        # # Presently, this only works for the 2 schmidt coefficient case
        # Ca = alp[2]*alp[3]*(alp[0] - ((-1)**(ms[dmi][0]))*alp[1])*(alp[6] - ((-1)**(ms[dmi][1]))*alp[7])
        # Cb = bet[2]*bet[3]*(bet[0] - ((-1)**(ms[dmj][0]))*bet[1])*(bet[6] - ((-1)**(ms[dmj][1]))*bet[7])
        # C = Ca*Cb

        # Seperating the coefficients in a way that can be used by the Wick coupling function

        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors()) # TODO: Should be a list of tuples, starting with the number then the rest of the list
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i)) # Should be a tuple as well

        return Coutf


    """
    Helper functions
    """

    @staticmethod
    def moment_vector(lambda_vector, l):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - l: The index of the moment to be calculated
        Output
        - An array of all of the moments that are to be calculated
        """
        mds = 8 * len(lambda_vector)  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1

        ms = tools.mcomb(len(lambda_vector))

        Ca1 = alp[0] * alp[2] * alp[3] * alp[7]
        Ca2 = alp[1] * alp[2] * alp[3] * alp[6]
        Cb1 = bet[0] * bet[2] * bet[3] * bet[7]
        Cb2 = bet[1] * bet[2] * bet[3] * bet[6]

        # For calculating the normalization constant
        Ca3 = alp[0] * alp[2] * alp[3] * alp[6]
        Ca4 = alp[1] * alp[2] * alp[3] * alp[7]
        Cb3 = bet[0] * bet[2] * bet[3] * bet[6]
        Cb4 = bet[1] * bet[2] * bet[3] * bet[7]

        if l == 0:
            C = alp[2] * alp[3] * bet[2] * bet[3]
        elif l == 1:
            C = Ca1 * Cb1
        elif l == 2:
            C = Ca1 * Cb2
        elif l == 3:
            C = Ca2 * Cb1
        elif l == 4:
            C = Ca2 * Cb2
        if l == 5:
            C = Ca3 * Cb3
        elif l == 6:
            C = Ca3 * Cb4
        elif l == 7:
            C = Ca4 * Cb3
        elif l == 8:
            C = Ca4 * Cb4
        elif l == 9:
            C = alp[2] * bet[2]
        elif l == 10:
            C = alp[3] * bet[3]
        elif l == 11:
            C = alp[2] * alp[3] * bet[2] * bet[3]
        elif l == 12:
            C = alp[0] * alp[0] * bet[0] * bet[0] # for testing the other vector approach
        elif l == 14:
            C = 1

        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(tools.expand_powers_to_symbols(i))

        return Coutf

    @staticmethod
    def moment_vector_nvec(nvec1, nvec2):
        """
        Arguments
        - lambda_vector: The vector of lambda_i's
        - nvec1: The click pattern on the alpha side of the density matrix calculation
        - nvec2: The click pattern on the beta side of the density matrix calculation
        Output
        - An array of all of the moments that are to be calculated for the given density matrix element
        """
        mds = 8  # Number of modes for our system

        # For the number of modes desired, create a vector of (q/p)_{\alphas / \beta}'s
        qai = [sp.symbols("qa{}".format(i)) for i in range(1, mds + 1)]
        pai = [sp.symbols("pa{}".format(i)) for i in range(1, mds + 1)]
        qbi = [sp.symbols("qb{}".format(i)) for i in range(1, mds + 1)]
        pbi = [sp.symbols("pb{}".format(i)) for i in range(1, mds + 1)]

        # Define the alpha and beta vectors
        alp = []
        bet = []
        j = 0
        while j < mds:
            alp.append((1 / np.sqrt(2)) * (qai[j] + sp.I * pai[j]))
            bet.append(
                (1 / np.sqrt(2)) * (qbi[j] - sp.I * pbi[j])
            )  # We have already taken the complex conjugate
            j += 1


        Ca = (alp[0]**nvec1[0]) * (alp[1]**nvec1[1]) * (alp[2]**nvec1[2]) * (alp[3]**nvec1[3]) * (alp[4]**nvec1[4]) * (alp[5]**nvec1[5]) * (alp[6]**nvec1[6]) * (alp[7]**nvec1[7])
        Cb = (bet[0]**nvec2[0]) * (bet[1]**nvec2[1]) * (bet[2]**nvec2[2]) * (bet[3]**nvec2[3]) * (bet[4]**nvec2[4]) * (bet[5]**nvec2[5]) * (bet[6]**nvec2[6]) * (bet[7]**nvec2[7])
        C = Ca*Cb


        # Seperating the coefficients in a way that can be used by the Wick coupling function
        C = sp.expand(C)
        Cv = C.as_ordered_terms()

        # Change the format just slightly
        Cout = []
        i = 0
        while i < len(Cv):
            Cout.append(Cv[i].as_ordered_factors())
            i += 1

        # Doing a little bit extra to handle the powers
        Coutf = []
        for i in Cout:
            # print(i)
            Coutf.append(ZALM.expand_powers_to_symbols(i))

        return Coutf

    @staticmethod
    def basisvZ(mds):
        """
        Arguments
        - mds: the number of modes
        Output
        - The basis vector that cooresponds to that number of modes
        """
        qai = [sp.symbols('qa{}'.format(i)) for i in range(1, mds+1)]
        pai = [sp.symbols('pa{}'.format(i)) for i in range(1, mds+1)]
        qbi = [sp.symbols('qb{}'.format(i)) for i in range(1, mds+1)]
        pbi = [sp.symbols('pb{}'.format(i)) for i in range(1, mds+1)]

        x = qai + pai + qbi + pbi
        return x

    @staticmethod
    def dmijpp(self, nAinv, nv1, nv2):
        """
        Arguments:
        - nAinv: The numerical inverse of the A matrix
        - lamvec: The vectors of lambdas for the system
        - dmi: The row number for the cooresponding density matrix element
        - dmj: The column number for the cooresponding density matrix element

        Output:
        - The density matrix element for the ZALM source
        """

        # Define the matrix element
        Cn = ZALM.moment_vector_nvec(nv1, nv2)

        return ZALM.dmatval(Cn, nAinv, self.basisv)

    @staticmethod
    def dmatval(Cni, Anv, xb):
        elm = 0
        for i in Cni:
            a = tools.wick_coupling_mat(i,xb)
            elm += tools.wick_out(a, Anv)
        return elm

    @staticmethod
    def dmatval_do_not_store_looping_pattern(Cni, Anv, xb):
        elm = 0.0
        bv_index_map = {element: idx for idx, element in enumerate(xb)}
        for i in Cni:
            elm += tools.wick_out_do_not_store_looping_pattern(i,bv_index_map,Anv)
        return elm

    @staticmethod
    def dmatval_do_not_store_looping_pattern_do_not_use_symbols(Cni, Anv, xb):
        elm = 0.0
        for i in Cni:
            elm += tools.wick_out_do_not_store_looping_pattern_do_not_use_symbols(i,Anv)
        return elm

    """
    Functions for analysis purposes
    """
    @staticmethod
    def Pgenrange(etat, etad, etab, muvr):
        """
        Calculates the probability of generation for a range of mean photon numbers
        """
        Pgenv = np.array([])

        zex = ZALM()

        zex.params["bsm_efficiency"] = etab
        zex.params["outcoupling_efficiency"] = etat
        zex.params["detection_efficiency"] = etad

        for i in tqdm(muvr):
            zex.params["mean_photon"] = i
            zex.run()
            zex.calculate_probability_success()
            Pgenv = np.append(Pgenv, zex.results["probability_success"])
        return Pgenv

    @staticmethod
    def psucc_value(etat, etad, etab, muv, Pd):
        zalm_example = ZALM()
        zalm_example.params["bsm_efficiency"] = etab # 1 dB of loss in the BSM
        zalm_example.params["outcoupling_efficiency"] = etat # 1 dB of loss in the transmission
        zalm_example.params["detection_efficiency"] = etad # So that each mode has equal loss
        zalm_example.params["mean_photon"] = muv
        zalm_example.params["dark_counts"] = Pd
        zalm_example.run()
        zalm_example.calculate_probability_success()
        return zalm_example.results["probability_success"]

    @staticmethod
    def eta_pgen_3d(mu):
        """
        Arguments
        - mu: The mean photon number vector
        Output
        Calculates the probability of generation for a range of transmission and detection efficiencies
        """
        #mu = np.array([10**(-4), 10**(-2), 0.5])
        eta_b = np.linspace(0,3,15)
        eta_t = np.linspace(0,10,15)

        # Create the meshgrid for the eta values
        EB, ET = np.meshgrid(eta_b, eta_t)

        # Vectorize the function that calculates the probability of error for the depolarizing channel
        pgen_vectorized = np.vectorize(ZALM.pgen_value)

        # Compute the values of Pef_depol for each combination of a, EPS, and L
        Z = pgen_vectorized(10**(-ET/10), 1, 10**(-EB/10), mu)

        return [EB, ET, Z]

    @staticmethod
    def Fidrange_full(etat, etad, etab, muv1, muv2, muvs):
        """
        Calculate the fidelity for a range of mean photon numbers
        """

        Fidl3 = np.array([])
        muv = np.linspace(muv1, muv2, muvs) #np.array([10**(-4), 0.001, 0.01, 0.05, 0.1, 0.2])

        zalm_fid_loss_3 = ZALM()
        zalm_fid_loss_3.params["bsm_efficiency"] = etab
        zalm_fid_loss_3.params["outcoupling_efficiency"] = etat
        zalm_fid_loss_3.params["detection_efficiency"] = etad
        for i in tqdm(muv):
            zalm_fid_loss_3.params["mean_photon"] = i
            zalm_fid_loss_3.run()
            zalm_fid_loss_3.calculate_fidelity_full()
            Fidl3 = np.append(Fidl3, zalm_fid_loss_3.results["fidelity"])
        return [muv, 4*Fidl3] # The 4 comes from the fact that the pgen in the numerator is off by a factor of 4

    @staticmethod
    def fidelity_value(etat, etad, etab, muv):
        """
        Calculate the fidelity for the given values
        """
        zalm_fid = ZALM()
        zalm_fid.params["bsm_efficiency"] = etab
        zalm_fid.params["outcoupling_efficiency"] = etat
        zalm_fid.params["detection_efficiency"] = etad
        zalm_fid.params["mean_photon"] = muv
        zalm_fid.run()
        zalm_fid.calculate_fidelity_full()

        return zalm_fid.results["fidelity"]

    @staticmethod
    def eta_fid_3d(mu):
        """
        Arguments
        - mu: The mean photon number vector
        Output
        Calculates the fidelity for a range of transmission and detection efficiencies
        """
        #mu = np.array([10**(-4), 10**(-2), 0.5])
        eta_b = np.linspace(0,3,10)
        eta_t = np.linspace(0,10,10)

        # Create the meshgrid for the eta values
        EB, ET = np.meshgrid(eta_b, eta_t)

        # Vectorize the function that calculates the probability of error for the depolarizing channel
        fid_vectorized = np.vectorize(ZALM.fidelity_value)

        # Compute the values of Pef_depol for each combination of a, EPS, and L
        Z = fid_vectorized(10**(-ET/10), 1, 10**(-EB/10), mu)

        return [EB, ET, Z]


    """
    Schmidt Analysis Function
    """
    @staticmethod
    def solve_squeezing_parameter(ns):
        """
        Solve for the squeezing parameter xi given a mean photon number value
        """

        # Define the equation to solve
        def equation(xi, ns):
            return np.sinh(np.abs(xi))**2 - ns

        # Initial guess for xi
        initial_guess = 1

        # Solve for xi
        xi_solution = fsolve(equation, initial_guess, args=(ns))

        return xi_solution

    @staticmethod
    def Ns_Schmidt(ns, lam_n):
        """
        For a given mean photon number value, calculate the resulting mean photon number for for a Schmidt value
        """

        xi_s = ZALM.solve_squeezing_parameter(ns)
        Ns_m = np.sinh(np.abs(np.sqrt(lam_n)*xi_s))**2
        return Ns_m[0]

    @staticmethod
    def closest_calc(sNs, muvv):
        """
        Given an array of mean photon number values, find the element closest to a specific value, i.e. a value that results from a Schmidt modulation
        """
        # Find the index of the element closest to the specific value
        index = np.abs(muvv - sNs).argmin()

        # Get the element closest to the specific value
        closest_element = muvv[index]

        return [closest_element, index]
