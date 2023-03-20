# Least Squares Fit
This is a Python module that implements the least squares algorithm to, given a $\varphi(x)=\alpha_1g_1(x)+\alpha_2g_2(x)+\cdots+\alpha_ng_n(x)$ function that seems to fit the data, find all the $\alpha_i$ that minimize the error

$\displaystyle \sum_{i=1}^n\left(y_i-\varphi(x_i)\right)^2$.

The algorithm works by solving the equations 

$\displaystyle \sum_{i=1}^n\left(y_i-\varphi(x_i)\right)g_j(x_i)=0$

for all $j=1,2,\cdots,n$, which results in solving the following system of linear equations:

$$\left(
\begin{matrix}
\alpha_1 (\vec{g_1} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_1} \cdot \vec{g_2})
& + & \alpha_3 (\vec{g_1} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_1}\\
\alpha_1 (\vec{g_2} \cdot \vec{g_1}) & + & \alpha_2 (\vec{g_2} \cdot \vec{g_2})
& + & \alpha_3 (\vec{g_2} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_2}\\
\vdots & + & \vdots & + & \vdots & = & \vdots\\
\alpha_1 (\vec{g_n} \cdot \vec{g_1}) & + & (\alpha_2 \vec{g_n} \cdot \vec{g_2})
& + & \alpha_3 (\vec{g_n} \cdot \vec{g_3}) & = & \vec{f} \cdot \vec{g_n}\\
\end{matrix}
\right.$$

Where $\vec{f}$ is the vector of $y_i$ values, and every $\vec{g_j}$ vector are the $x_i$ values evaluated in the $g_j(x)$ functions.

## Requirements
-   Python 3.10 or above to avoid any errors related to Type Hints from PEP 585 and PEP 604;
-   Sympy;
-   Numpy;
-   Matplotlib;

You can run the following command to install all the requirements:

```bash
pip install -r requirements.txt
```

## Usage
To start, download or clone the repository, and navigate to the project directory. Create a new  .py file, and import the LeastSquares class from the least_squares module.

```python
from least_squares import LeastSquares
```

Create a new LeastSquares object, passing all $g_j(x)$ functions in a sequence of symbolic expressions, and the $x_i$ and $y_i$ values as sequences of int or float. **Be sure to use the symbol _x_ to represent the independent variable.**

An example of usage is shown below:

```python
import sympy as sp


x_values = [0, 1, 2, 3, 4]
y_values = [1, -1, 0, 20, 100]

x = sp.Symbol('x')
g_functions = [x, x ** 2, x ** 3]

ls = LeastSquares(x=x_values, y=y_values, functions=g_functions)
```

From the instance, call the solve() method to find the $\alpha_i$ that minimize the error. The $\varphi(x)$ can then be accessed from the phi attribute.

After calling solve(), you may now can call other methods such as

-   plot: plots the fitted function $\varphi(x)$ and the data points $(x_i, y_i)$.
-   evaluate_error: define the attributes absolute_error, mean_square_error and relative_error available.
-   predict: returns the value of $\varphi(x)$ for a given sequence of value of $x$.

```python
ls.solve()
ls.evaluate_error()

ls.phi
#(221/46)*x**3 - (729/46)*x**2 + (263/23)*x
ls.mean_square_error
#1.36732045161012

ls.plot()
```
![plot of the fitted function](https://i.ibb.co/hLSsDx9/baixados.png)

## Documentation
Further documentation about the implementation can be found [here](https://andrey-rv.github.io/LeastSquares/)
