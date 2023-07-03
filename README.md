# heat-wgpu

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/vini-fda/heat-wgpu?include_prereleases)
![GitHub](https://img.shields.io/github/license/vini-fda/heat-wgpu)

This is a Rust program which leverages computing and rendering capabilities of modern GPUs to efficiently solve the [heat equation](https://en.wikipedia.org/wiki/Heat_equation) and display the result in real time. It is based on the [`wgpu`](https://lib.rs/crates/wgpu) crate, which is a Rust wrapper around the WebGPU API.

The heat equation is a partial differential equation which describes the flow of heat in a given domain. For a 2D domain, it is given by the following equation:

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

where $u$ is the temperature, $t$ is time, $\alpha$ is the thermal diffusivity and $x$ and $y$ are the spatial coordinates.

The equation is solved numerically using the finite difference method, which consists in discretizing the domain into a grid and approximating the derivatives by finite differences. The discretization scheme used here is the [Crank-Nicolson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method), which is unconditionally stable and second-order accurate in time and space.

The resulting system of equations is then solved using the [Conjugate Gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method), which is written as a series of WGSL compute shaders and runs on the GPU through `wgpu`.

The mathematical formulation, including explanations of the finite difference scheme and the conjugate gradient method, is described in [discretization](docs/discretization.md).

## Sample output

![Sample output](images/example.png)

## Usage

### From binary

Download the latest executable from [the GitHub release page](https://github.com/vini-fda/heat-wgpu/releases). Supported platforms are Windows, Linux (mainly Ubuntu) and macOS. You can start the binary on its own by running it with no arguments. On all systems, this can be done by double-clicking the executable.

On Linux and macOS, you can also run it from the terminal:

```bash
./heat-wgpu
```

On Windows, you can run it from the command prompt:

```cmd
> heat-wgpu.exe
```

### From source

You need to have the Rust toolchain installed. Check out the [official language website](https://www.rust-lang.org/) for instructions in how to install. Then, run the following command:

```shell
cargo run --release
```

## References and useful resources

- LeVeque, R. J. (2007). *Finite difference methods for ordinary and partial differential equations: steady-state and time-dependent problems. Society for Industrial and Applied Mathematics*.
- Shewchuk, J. R. (1994). [*An introduction to the conjugate gradient method without the agonizing pain*](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf).
- Bell, N., & Garland, M. (2008). *Efficient sparse matrix-vector multiplication on CUDA (Vol. 2, No. 5). Nvidia Technical Report NVR-2008-004, Nvidia Corporation*.
- Harris, M. (2007). *Optimizing parallel reduction in CUDA. Nvidia developer technology, 2*(4), 70.
- Mikhailov, A. (2019). [Turbo, An Improved Rainbow Colormap for Visualization](https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html).
- [WebGPU specification](https://gpuweb.github.io/gpuweb/).
