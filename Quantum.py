from IPython.display import display, Latex
import sympy as sym
from sympy.physics.quantum import TensorProduct
from sympy.core.numbers import I
import numpy as np
import copy


class State:
    def __init__(
        self,
        coeffs=[],
        states=[]
    ):
        assert len(coeffs) == len(states)
        sum_coeffs = 0
        for coeff in coeffs:
            sum_coeffs += sym.conjugate(coeff) * coeff
        sum_coeffs = sym.simplify(sum_coeffs)
        if not sum_coeffs == sym.Integer(1):
            print(f"Warning: State not normalized. Probabilities sum to {sum_coeffs}")
            assert sum_coeffs == sym.Integer(1), "Sum of probabilities must be 1."
        self.coeffs = coeffs
        self.states = states

    def __repr__(self) -> str:
        return f"(Coeff: {self.coeffs}, State: {self.states})"

    def __str__(self):
        return f"({self.coeffs}, {self.states})"

    def show(
        self
    ):
        state_str = []
        for coeff, state in zip(self.coeffs, self.states):
            coeff = sym.simplify(sym.nsimplify(coeff))
            if not sym.re(coeff) == sym.Integer(0):
                if not sym.im(coeff) == sym.Integer(0):
                    state_str.append(
                        f"{sym.latex(coeff)}"
                        + r"\left|" + sym.latex(state) + r"\right\rangle"
                    )
                if sym.im(coeff) == sym.Integer(0):
                    if sym.re(coeff) == sym.Integer(1):
                        state_str.append(
                            r"\left|" + sym.latex(state) + r"\right\rangle"
                        )
                    if sym.re(coeff) == -sym.Integer(1):
                        state_str.append(
                            r"-\left|" + sym.latex(state) + r"\right\rangle"
                        )
                    if not sym.re(coeff) == sym.Integer(1) and not sym.re(coeff) == -sym.Integer(1):
                        state_str.append(
                            f"{sym.latex(coeff)}"
                            + r"\left|" + sym.latex(state) + r"\right\rangle"
                        )
            if sym.re(coeff) == sym.Integer(0):
                if not sym.im(coeff) == sym.Integer(0):
                    state_str.append(
                        f"{sym.latex(coeff)}"
                        + r"\left|" + sym.latex(state) + r"\right\rangle"
                    )

        display(
            Latex(
                ("$" + "+".join(state_str) +
                 "$").replace("+-", "-").replace("+ -", "-")
            )
        )

    def __call__(
        self
    ):
        s = [0 for _ in range(2 ** len(str(self.states[0])))]

        for i, state in enumerate(self.states):
            s[int(str(state), 2)] = self.coeffs[i]

        return s

    def __mul__(
        self,
        other
    ):
        result_coeffs = [0 for _ in range(
            2 ** (len(str(self.states[0])) + len(str(other.states[0]))))]

        for sc, ss in zip(self.coeffs, self.states):
            for oc, os in zip(other.coeffs, other.states):
                result_coeffs[int(str(ss) + str(os), 2)] = sc * oc

        return state_from_array(result_coeffs)

    def entangled(self):
        entangled = "Unknown"
        if len(str(self.states[0])) == 1:
            entangled = "Not Entangled"

        if len(str(self.states[0])) == 2:
            A = np.zeros((2, 2))
            for coeff, state in zip(self.coeffs, self.states):
                A[int(str(state)[0]), int(str(state)[1])] = coeff
            if np.isclose(np.linalg.det(A), 0):
                entangled = "Not Entangled"
            else:
                entangled = "Entangled"

        print(f"Entanglement Status: {entangled}")

    def measure(
        self,
        *qubits
    ):
        def recurse(coeffs, states, *qubits):

            available_states = copy.deepcopy(states)
            index, *qubits = qubits

            probs = []
            for coeff in coeffs:
                probs.append(
                    float((sym.conjugate(coeff) * coeff).evalf())
                )
            selection = np.random.choice(
                states, p=np.abs(np.array(probs, dtype=float))
            )

            for state in states:
                if state[index] != selection[index] and state in available_states:
                    available_states.remove(state)

            available_coeffs = sym.Array(
                [
                    coeff for coeff, state in zip(self.coeffs, self.states)
                    if state in available_states
                ]
            )



            normalization = 1 / sym.sqrt(
                np.sum(
                    [
                        sym.conjugate(available_coeff) * available_coeff
                        for available_coeff in available_coeffs
                    ]
                )
            )
            available_coeffs = normalization * available_coeffs

            if not qubits:
                return available_coeffs, available_states
            else:
                return recurse(available_coeffs, available_states, *qubits)

        remaining_coeffs, remaining_states = recurse(
            self.coeffs, self.states, *qubits)

        return State(
            coeffs=remaining_coeffs,
            states=remaining_states
        )


def random_state():
    a = sym.Rational(np.random.randint(0, 1000), 1000) + I * sym.Rational(np.random.randint(0, 1000), 1000)
    b = sym.Rational(np.random.randint(0, 1000), 1000) + I * sym.Rational(np.random.randint(0, 1000), 1000)
    norm = sym.sqrt(a * a.conjugate() + b * b.conjugate())
    a, b = sym.simplify(a / norm), sym.simplify(b / norm)
    return State(
        coeffs=[a, b],
        states=["0", "1"]
    )


def bits_to_send(state):
    expanded = [value for value in expand(state).values()]

    alice = [(1, s[0:2]) for (c, s) in expanded]
    alice = np.prod([s for s in alice[0][1]])

    bob = [(c, s[2]) for (c, s) in expanded]
    bob = State(
        coeffs=[c for (c, s) in bob],
        states=["".join(s.states) for (c, s) in bob]
    )

    return alice, bob


def receive_state(alice, bob):
    bits = np.where(np.isclose(alice.coeffs, 1))[0][0]
    return {
        0: bob,
        1: NOT(bob, 0),
        2: Z(bob, 0),
        3: Z(NOT(bob, 0), 0)
    }[bits]


def state_n_zeros(n):
    return State(
        coeffs=[1],
        states=["".zfill(n)]
    )


def state_from_array(array):
    return State(
        coeffs=array,
        states=[
            str(str(bin(i))[2:].zfill(int(np.log2(len(array)))))
            for i in range(len(array))
        ]
    )


def expand(
    state
):
    expanded = {}
    for i, composite_state in enumerate(state.states):
        qubits = [
            State(
                coeffs=[1],
                states=[s]
            ) for s in composite_state
        ]
        expanded[i] = (state.coeffs[i], qubits)
    return expanded


zero = State(
    coeffs=[1],
    states=["0"]
)

one = State(
    coeffs=[1],
    states=[1]
)

minus = State(
    coeffs=[1/sym.sqrt(2), -1/sym.sqrt(2)],
    states=["0", "1"]
)

plus = State(
    coeffs=[1/sym.sqrt(2), 1/sym.sqrt(2)],
    states=["0", "1"]
)

plusi = State(
    coeffs=[1/sym.sqrt(2), I/sym.sqrt(2)],
    states=["0", "1"]
)

minusi = State(
    coeffs=[1/sym.sqrt(2), -I/sym.sqrt(2)],
    states=["0", "1"]
)

B00 = State(
    coeffs=[1/sym.sqrt(2), 1/sym.sqrt(2)],
    states=["00", "11"]
)

B01 = State(
    coeffs=[1/sym.sqrt(2), 1/sym.sqrt(2)],
    states=["01", "10"]
)

B10 = State(
    coeffs=[1/sym.sqrt(2), -1/sym.sqrt(2)],
    states=["00", "11"]
)

B11 = State(
    coeffs=[1/sym.sqrt(2), -1/sym.sqrt(2)],
    states=["01", "10"]
)


def CNOT(
    state,
    qubit1,
    qubit2
):

    op = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ to_transform()))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def H(
    state, *qubits
):

    h = (1/sym.sqrt(2)) * sym.Matrix(
        [
            [1, 1],
            [1, -1]
        ]
    )

    operators = []
    for i in range(len(str(state.states[0]))):
        if i in qubits:
            operators.append(h)
        else:
            operators.append(sym.eye(2))

    op = TensorProduct(*operators)

    return state_from_array(op * sym.Matrix(state()))


def H_transform(
    state, n
):

    qubits = [i for i in range(n)]

    return H(state, *qubits)


def X(
    state, *qubits
):

    h = sym.Matrix(
        [
            [0, 1],
            [1, 0]
        ]
    )

    operators = []
    for i in range(len(str(state.states[0]))):
        if i in qubits:
            operators.append(h)
        else:
            operators.append(sym.eye(2))

    op = TensorProduct(*operators)

    return state_from_array(op * sym.Matrix(state()))


def NOT(
    state, *qubits
):
    return X(state, *qubits)


def Y(
    state, *qubits
):

    h = sym.Matrix(
        [
            [0, -I],
            [I, 0]
        ]
    )

    operators = []
    for i in range(len(str(state.states[0]))):
        if i in qubits:
            operators.append(h)
        else:
            operators.append(sym.eye(2))

    op = TensorProduct(*operators)

    return state_from_array(op * sym.Matrix(state()))


def Z(
    state, *qubits
):

    h = sym.Matrix(
        [
            [1, 0],
            [0, -1]
        ]
    )

    operators = []
    for i in range(len(str(state.states[0]))):
        if i in qubits:
            operators.append(h)
        else:
            operators.append(sym.eye(2))

    op = TensorProduct(*operators)

    return state_from_array(op * sym.Matrix(state()))


def CH(
    state,
    qubit1,
    qubit2
):

    op = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/sym.sqrt(2), 1/sym.sqrt(2)],
            [0, 0, -1/sym.sqrt(2), 1/sym.sqrt(2)]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ sym.Matrix(to_transform())))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def CX(
    state,
    qubit1,
    qubit2
):

    return CNOT(state, qubit1, qubit2)


def CY(
    state,
    qubit1,
    qubit2
):

    op = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -I],
            [0, 0, I, 0]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ sym.Matrix(to_transform())))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def CZ(
    state,
    qubit1,
    qubit2
):

    op = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ sym.Matrix(to_transform())))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def SWAP(
    state,
    qubit1,
    qubit2
):

    op = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ sym.Matrix(to_transform())))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def P(
    state, *qubits, phase
):

    h = sym.Matrix(
        [
            [1, 0],
            [0, sym.exp(I * phase)]
        ]
    )

    operators = []
    for i in range(len(str(state.states[0]))):
        if i in qubits:
            operators.append(h)
        else:
            operators.append(sym.eye(2))

    op = TensorProduct(*operators)

    return state_from_array(op * sym.Matrix(state()))


def CP(
    state,
    qubit1,
    qubit2,
    phase
):

    op = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, sym.exp(I * phase)]
        ]
    )

    expanded = expand(state)

    to_construct = []
    for i, (coeff, qubits) in expanded.items():
        q1 = qubits[qubit1]
        q2 = qubits[qubit2]
        to_transform = q1 * q2
        result = expand(state_from_array(op @ sym.Matrix(to_transform())))
        for j, (res_coeff, res_qubits) in result.items():
            state_coeff = coeff * res_coeff
            if state_coeff != 0:
                state_states = copy.deepcopy(qubits)
                state_states[qubit1] = res_qubits[0]
                state_states[qubit2] = res_qubits[1]
                res_state = np.prod(state_states)
                res_state.coeffs = list(
                    state_coeff * np.array(res_state.coeffs))
                to_construct.append(res_state)

    result_coeffs = []
    result_states = []
    for s in to_construct:
        for i, c in enumerate(s.coeffs):
            if not sym.Abs(c) == sym.Integer(0):
                result_coeffs.append(c)
                result_states.append(str(s.states[i]))

    return State(
        coeffs=result_coeffs,
        states=result_states
    )


def DJU(
    state,
    f
):
    def balanced_f(state):
        result_state = copy.deepcopy(state)
        sign = np.random.choice([-1, 1])
        for i, (c, s) in iter(enumerate(zip(state.coeffs, state.states))):
            fx = sign if int(s[:-1], 2) % 2 == 0 else -sign
            result_state.coeffs[i] = c * fx
        return result_state

    def constant_f(state):
        fx = (-1)**np.random.randint(2)
        result_state = copy.deepcopy(state)
        for i, c in iter(enumerate(state.coeffs)):
            result_state.coeffs[i] = c * fx
        return result_state

    def random_f(state):
        result_state = copy.deepcopy(state)
        return np.random.choice([balanced_f, constant_f])(result_state)

    return {
        "balanced": balanced_f,
        "constant": constant_f,
        "random": random_f
    }[f](state)


def DJU_result(state):
    if int(state.states[0][:-1], 2) == 0:
        print("DJU Algorithm Result: f(x) is Constant")
    else:
        print("DJU Algorithm Result: f(x) is Balanced")
