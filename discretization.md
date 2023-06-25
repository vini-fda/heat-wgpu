# PDE discretization with finite difference methods

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

## Crank-Nicolson

$$ \frac{U_{i,j}^{n+1} - U_{i,j}^{n}}{\Delta t} = \frac{\alpha}{2} \left(\frac{U_{i+1,j}^{n+1} - 2 U_{i,j}^{n+1} + U_{i-1,j}^{n+1}}{\Delta x^2} + \frac{U_{i,j+1}^{n+1} - 2 U_{i,j}^{n+1} + U_{i,j-1}^{n+1}}{\Delta y^2} \right) + \frac{1}{2} \left(\frac{U_{i+1,j}^{n} - 2 U_{i,j}^{n} + U_{i-1,j}^{n}}{\Delta x^2} + \frac{U_{i,j+1}^{n} - 2 U_{i,j}^{n} + U_{i,j-1}^{n}}{\Delta y^2} \right)$$

With $\Delta x = \Delta y = h$ and $\Delta t = k$:

$$ U_{i,j}^{n+1} = U_{i,j}^{n} + \frac{\alpha k}{2 h^2} \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} - 4 U_{i,j}^{n+1} + U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} - 4 U_{i,j}^{n} \right)$$

Let's define $\gamma = \frac{\alpha k}{2 h^2}$. We then have:

$$ U_{i,j}^{n+1} = U_{i,j}^{n} + \gamma \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} - 4 U_{i,j}^{n+1} + U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} - 4 U_{i,j}^{n} \right)$$

We now pass all $n+1$ terms to the left side and all $n$ terms to the right side:

$$ (1 + 4 \gamma) U_{i,j}^{n+1} - \gamma \left( U_{i+1,j}^{n+1} + U_{i-1,j}^{n+1} + U_{i,j+1}^{n+1} + U_{i,j-1}^{n+1} \right) = (1 - 4 \gamma) U_{i,j}^{n} + \gamma \left( U_{i+1,j}^{n} + U_{i-1,j}^{n} + U_{i,j+1}^{n} + U_{i,j-1}^{n} \right)$$

Which can be represented as a matrix equation:

$$ A U^{n+1} = B U^{n} $$

Where $A$ is a matrix with $1 + 4 \gamma$ on the diagonal and $-\gamma$ on the off-diagonals, $U^{n+1}$ is a vector with all $U_{i,j}^{n+1}$ terms, $B$ is a matrix with $1 - 4 \gamma$ on the diagonal and $\gamma$ on the off-diagonals, and $U^{n}$ is a vector with all $U_{i,j}^{n}$ terms.
