#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    s = np.pi/2

    N = 5
    shift_grads = weights.copy()
    mid = circuit(shift_grads)
    mid *= 2
    for i in range(N):
        shift_grads = weights.copy()
        
        
        shift_grads[i] += s
        forward = circuit(shift_grads)
        
        shift_grads[i] -= 2* s
        backward = circuit(shift_grads)
        
        
        hessian[i,i] = (forward+backward-mid) /(2 * np.sin(s)**2)
        gradient[i]= (forward-backward) /(2* np.sin(s)**2 )
 
        for j in range(i+1,N):
            shifts = weights.copy()
            shifts[i] += s
            shifts[j] += s
            forward1 = circuit(shifts)

            shifts[i] -= s * 2
            shifts[j] -= s * 2
            forward2 = circuit(shifts)

            shifts[i] += s * 2
            backward1 = circuit(shifts)

            shifts[i] -= s * 2
            shifts[j] += s * 2
            backward2 = circuit(shifts)

            hessian[i,j] = (forward1+forward2 - (backward1+backward2))/ ( 4 * (np.sin(s) ** 2))
            hessian[j,i] = (forward1+forward2 - (backward1+backward2))/ ( 4 * (np.sin(s) ** 2))
            
    # QHACK #
    
    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
