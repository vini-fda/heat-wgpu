# PDE discretization with finite difference methods

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

## Crank-Nicolson

$$ \frac{U_{i,j}^{n+1} - U_{i,j}^{n}}{\Delta t} = \frac{\alpha}{2} \left(\frac{U_{i+1,j}^{n+1} - 2 U_{i,j}^{n+1} + U_{i-1,j}^{n+1}}{\Delta x^2} + \frac{U_{i,j+1}^{n+1} - 2 U_{i,j}^{n+1} + U_{i,j-1}^{n+1}}{\Delta y^2} \right) + \frac{\alpha}{2} \left(\frac{U_{i+1,j}^{n} - 2 U_{i,j}^{n} + U_{i-1,j}^{n}}{\Delta x^2} + \frac{U_{i,j+1}^{n} - 2 U_{i,j}^{n} + U_{i,j-1}^{n}}{\Delta y^2} \right)$$

With $\Delta x = \Delta y = h$ and $\Delta t = k$:

$$ U_{i,j}^{n+1} = U_{i,j}^{n} + \frac{\alpha k}{2 h^2} \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} - 4 U_{i,j}^{n+1} + U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} - 4 U_{i,j}^{n} \right)$$

Let's define $\gamma = \frac{\alpha k}{2 h^2}$. We then have:

$$ U_{i,j}^{n+1} = U_{i,j}^{n} + \gamma \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} - 4 U_{i,j}^{n+1} + U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} - 4 U_{i,j}^{n} \right)$$

We now pass all $n+1$ terms to the left side and all $n$ terms to the right side:

$$ (1 + 4 \gamma) U_{i,j}^{n+1} - \gamma \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} \right) = (1 - 4 \gamma) U_{i,j}^{n} + \gamma \left( U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} \right)$$

Which can be represented as a matrix equation:

$$ A U^{n+1} = B U^{n} $$

Where $A$ is a matrix with $1 + 4 \gamma$ on the diagonal and $-\gamma$ on the off-diagonals, $U^{n+1}$ is a vector with all $U_{i,j}^{n+1}$ terms, $B$ is a matrix with $1 - 4 \gamma$ on the diagonal and $\gamma$ on the off-diagonals, and $U^{n}$ is a vector with all $U_{i,j}^{n}$ terms.


## Conjugate-Gradient method

This is an algorithm to solve linear systems of equations. It is an iterative method, which means that it will converge to the solution after a number of iterations. The algorithm is as follows:

[//]: # "load from CPU to GPU memory"

1. Choose an initial guess $x_0$ and set $r_0 = b - A x_0$.
2. If $r_0$ is sufficiently small, return $x_0$.
3. Set $p_0 = r_0$.
4. For $k = 0, 1, 2, \dots k_\text{max}$:
    1. Set $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$.
    2. Set $x_{k+1} = x_k + \alpha_k p_k$.
    3. Set $r_{k+1} = r_k - \alpha_k A p_k$.
    4. If $r_{k+1}$ is sufficiently small, return $x_{k+1}$.
    5. Set $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$.
    6. Set $p_{k+1} = r_{k+1} + \beta_k p_k$.
5. Return $x_{k_\text{max}}$.
