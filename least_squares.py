from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class LeastSquares:
    def __init__(self, x: Sequence[float | int], y: Sequence[float | int], functions: Sequence['sp.Symbol']) -> None:
        r"""
        Args:
            x (Sequence[float | int]): _The x values_
            y (Sequence[float | int]): _The true y values_
            functions (Sequence['sp.Symbol']): _The functions to be used in the least squares method_

        Attributes:
            x (np.ndarray): _The x values_
            f_vector (np.ndarray): _The true y values_
            g_functions (Sequence['sp.Symbol']): _The functions to be used in the least squares method_
            g_vectors (list[np.ndarray]): _A list of each g function evaluated in each x value_
            phi (sp.Symbol): _The least squares fitted function_
            coefficients (dict): _The coefficients of the g functions_
            mean_square_error (float): _The mean square error of the fitted function_
            relative_errors (np.ndarray): _The relative error of each point_

        """
        self.x = np.array(x)
        self.g_functions = functions
        self.f_vector = np.array(y)
        self.get_g_vectors()

    def get_g_vectors(self) -> None:
        r"""_Constructs an array containing \(n\) vectors \(g_i(x_k)\) of the form:
                    $$[[g_1(x_0), g_1(x_1),...,g_1(x_m)],\\
                    [g_2(x_0), g_2(x_1),...,g_2(x_m)],\\
                    [g_n(x_0), g_n(x_1),...,g_n(x_m)]]$$
            for \(n\) functions \(g\) and \(m\) points \(x\) passed to the constructor.

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
        r"""Returns \(\phi (x)\) and its coefficients by solving the system of linear equations:
            $$\left\{
                \begin{matrix}
                    \alpha_1 (\vec{g_1} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_1} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_1} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_1}\\
                    \alpha_1 (\vec{g_2} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_2} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_2} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_2}\\
                    \vdots & + & \vdots & + & \vdots & = & \vdots\\
                    \alpha_1 (\vec{g_n} \cdot \vec{g_1}) & + & (\alpha_2 \vec{g_n} \cdot \vec{g_2})
                    & + & \alpha_3 (\vec{g_n} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_n}\\
                \end{matrix}
            \right.$$_
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
        r"""_Calculates the mean square error and the relative error of the fitted function_"""
        x = sp.Symbol('x')
        Y = self.f_vector
        predicted_y = [self.phi.subs(x, i) for i in self.x]                           # type: ignore
        predicted_y = np.array(predicted_y).astype(float)                             # type: ignore

        error = predicted_y - Y
        absolute_error = np.abs(error)
        squared_error = (predicted_y - Y) ** 2
        mean_squared_error = np.sum(squared_error) / len(self.x)                      # type: ignore

        self.mean_square_error = np.sqrt(mean_squared_error)
        self.relative_error = absolute_error / Y

    def predict(self, *args: Sequence[int | float]) -> list[float]:
        r"""_Returns a list of predicted values for the given points_

        Args:
            *args (Sequence[float | int]): _The points to predict_

        Returns:
            list[float]: _The predicted values_
        """
        x = sp.Symbol('x')
        return [self.phi.subs(x, i) for i in args[0]]                                 # type: ignore

    @staticmethod
    def plot(X: Sequence[int | float], y: Sequence[int | float]) -> None:
        r"""_Plots the given points_

        Args:
            X (Sequence[float | int]): _The x values_
            y (Sequence[float | int]): _The y values_
        """
        plt.scatter(X, y, marker='x', color='r', s=25)                               # type: ignore
        plt.title('Least Squares Fit', weight='bold', y=1.05)                        # type: ignore
        plt.xlabel('x')                                                              # type: ignore
        plt.show()                                                                   # type: ignore
