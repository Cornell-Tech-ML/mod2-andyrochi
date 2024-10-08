from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.

    vals_list = list(vals)
    vals_list[arg] += epsilon
    f_forward = f(*vals_list)
    vals_list[arg] -= 2 * epsilon
    f_backward = f(*vals_list)
    derivative = (f_forward - f_backward) / (2 * epsilon)
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    topological_sorted = []
    visited = set()

    def dfs(node: Variable) -> None:
        if node.unique_id in visited:
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            dfs(parent)
        if not node.is_constant():
            topological_sorted.append(node)

    dfs(variable)

    return reversed(topological_sorted)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv   : Its derivative that we want to propagate backward to the leaves.


    Returns:
    -------
        None. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # Perform a topological sort on the computational graph
    topological_sorted = topological_sort(variable)

    # Initialize a dictionary to store the derivatives of the scalar variables
    derivatives = {variable.unique_id: deriv}

    for node in topological_sorted:
        if node.is_leaf():
            node.accumulate_derivative(derivatives[node.unique_id])
        else:
            current_deriv = derivatives[node.unique_id]
            for parent, local_grad in node.chain_rule(current_deriv):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += local_grad
                else:
                    derivatives[parent.unique_id] = local_grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for use in backpropagation."""
        return self.saved_values
