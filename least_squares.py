from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

numeric_sequence = Sequence[int | float]
symbolic_sequence = Sequence[sp.Symbol]


class LeastSquares:
    def __init__(self, x: numeric_sequence, y: numeric_sequence, functions: symbolic_sequence) -> None:
        r"""
        Args:
            x (Sequence[int | float]): The x values.
            y (Sequence[int | float]): The true y values.
            functions (Sequence[sp.Symbol]): The functions to be used in the least squares method.
        """
        self.x = np.array(x)
        self.g_functions = functions
        self.f_vector = np.array(y)
        self._get_g_vectors()

    def _get_g_vectors(self) -> None:
        r"""Construct an array containing \(n\) vectors \(g_i(x_k)\) of the form:
                    $$[[g_1(x_0), g_1(x_1),...,g_1(x_m)],\\
                    [g_2(x_0), g_2(x_1),...,g_2(x_m)],\\
                    [g_n(x_0), g_n(x_1),...,g_n(x_m)]]$$
            for $n$ functions $g$ and $m$ points $x$ passed to the constructor.
        """
        x = sp.Symbol('x')
        g_vectors = []
        for g_function in self.g_functions:
            current_g_vector = []
            for point in self.x:
                current_value = g_function.subs(x, point)  # type: ignore
                current_g_vector.append(current_value)     # type: ignore
            g_vectors.append(current_g_vector)             # type: ignore

        self.g_vectors = np.array(g_vectors)               # type: ignore

    def solve(self) -> None:
        r"""Calculate $\phi (x)$ and its coefficients by solving the system of linear equations:
            $$\left\(
                \begin{matrix}
                    \alpha_1 (\vec{g_1} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_1} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_1} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_1} \\\\
                    \alpha_1 (\vec{g_2} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_2} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_2} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_2} \\\\
                    \vdots & + & \vdots & + & \vdots & = & \vdots \\\\
                    \alpha_1 (\vec{g_n} \cdot \vec{g_1}) & + & (\alpha_2 \vec{g_n} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_n} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_n} \\\\
                \end{matrix}
            \right.$$
        """
        number_of_functions = len(self.g_functions)
        coeffs_matrix = []
        indexes = range(number_of_functions)
        new_shape = (number_of_functions, number_of_functions + 1)

        for i in indexes:
            for j in indexes:
                coeffs_matrix.append(self.g_vectors[i] @ self.g_vectors[j])          # type: ignore
            coeffs_matrix.append(self.f_vector @ self.g_vectors[i])                  # type: ignore

        coeffs_matrix = np.array(coeffs_matrix).reshape(new_shape)                   # type: ignore
        unsolved_coeffs = [sp.Symbol(f'{i}') for i in indexes]
        self.coefficients = sp.solve_linear_system(coeffs_matrix, *unsolved_coeffs)  # type: ignore

        g_functions_with_coeff = []
        for i in indexes:
            current_coeff = self.coefficients[unsolved_coeffs[i]]                     # type: ignore
            current_g_function = self.g_functions[i]
            current_g_function_with_coeff = current_coeff * current_g_function        # type: ignore
            g_functions_with_coeff.append(current_g_function_with_coeff)              # type: ignore

        self.phi = np.sum(g_functions_with_coeff)                                     # type: ignore

    def evaluate_error(self) -> None:
        """Calculate the absolute, mean square and relative errors of the fitted function."""
        x = sp.Symbol('x')
        Y = self.f_vector
        predicted_y = [self.phi.subs(x, i) for i in self.x]                           # type: ignore
        predicted_y = np.array(predicted_y).astype(float)                             # type: ignore

        error = predicted_y - Y
        self.absolute_error = np.abs(error)
        squared_error = (predicted_y - Y) ** 2
        mean_squared_error = np.sum(squared_error) / len(self.x)                      # type: ignore

        self.mean_square_error = np.sqrt(mean_squared_error)
        self.relative_error = self.absolute_error / Y

    def predict(self, *args: numeric_sequence) -> list[float]:
        """Return a list of predicted values for the given points.

        Args:
            args (Sequence[int | float]): The points to predict.

        Returns:
            list[float]: The predicted values.
        """
        x = sp.Symbol('x')
        return [self.phi.subs(x, i) for i in args[0]]                                  # type: ignore

    def plot(self) -> None:
        """Plot the fitted function."""
        x_sample = np.linspace(self.x[0], self.x[-1], 1000)                                 # type: ignore
        y_predict = self.predict(x_sample)                                                  # type: ignore
        plt.scatter(self.x, self.f_vector, marker='x', color='black', label='Actual data')  # type: ignore
        plt.plot(x_sample, y_predict, color='r', label='Fitted model')                      # type: ignore
        plt.title('Least Squares Fit', weight='bold', y=1.05)                               # type: ignore
        plt.legend()                                                                        # type: ignore
        plt.show()                                                                          # type: ignore
