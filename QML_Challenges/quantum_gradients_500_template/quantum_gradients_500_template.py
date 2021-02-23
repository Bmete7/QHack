#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np
from copy import deepcopy
# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    N = 6
    @qml.qnode(dev)
    def qcirc(params):
        """A PennyLane QNode that pairs the variational_circuit with probabilistic measurement."""
        variational_circuit(params)
        return qml.probs(range(0,3))
    
    # shifting amount for the gradients
    twist  = np.pi/2
    gradient = np.zeros([N] , dtype = np.float64)
    
    # Fubini-Study metric
    F = np.zeros([N,N] , dtype = np.float64)
    
    initial_measurement = qcirc(params)
    initial_state = deepcopy(dev.state)
    
    

        
    for i in range(N):
         twisted_params = params.copy()
         twisted_params[i] += twist
        
         grad_measurement_1 = qnode(twisted_params)
         twisted_params[i] -= (2 * twist)
        
         grad_measurement_2 = qnode(twisted_params)
         gradient[i] = (grad_measurement_1 - grad_measurement_2)/(2 * np.sin(twist))
         for j in range(N):
            twisted_params = params.copy()
            
            twisted_params[i] += twist
            twisted_params[j] += twist
            qcirc(twisted_params)
            
            stat_vec_1 = deepcopy(dev.state)
            
            twisted_params = params.copy()
            twisted_params[i] -= twist
            twisted_params[j] += twist
            qcirc(twisted_params)
            
            stat_vec_2 = deepcopy(dev.state)
            
            twisted_params = params.copy()
            
            twisted_params[i] += twist
            twisted_params[j] -= twist
            qcirc(twisted_params)
            stat_vec_3 = deepcopy(dev.state)
            twisted_params = params.copy()
            
            twisted_params[i] -= twist
            twisted_params[j] -= twist           
            qcirc(twisted_params)
            stat_vec_4 = deepcopy(dev.state)
            # inner product of the acftual state and the pi/2 shifted state
            metric1 = abs(  np.array(np.matrix(stat_vec_1).H).T.dot(initial_state))**2
            metric2 = abs(  np.array(np.matrix(stat_vec_2).H).T.dot(initial_state))**2
            metric3   = abs(  np.array(np.matrix(stat_vec_3).H).T.dot(initial_state))**2
            metric4 =abs(  np.array(np.matrix(stat_vec_4).H).T.dot(initial_state))**2
            
            F[i,j] = -metric1+metric2 + metric3 -  metric4
            F[i,j] /= 8
            
         
    natural_grad = np.linalg.inv(F) @ gradient

    # compare with the pennylane implementation
    met_fn=qml.metric_tensor(qcirc)
    met_fn(params)
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
