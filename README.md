# heat-wgpu

This is a Rust program which leverages computing and rendering capabilities of modern GPUs to solve the heat equation PDE and display results in real time. It is based on the `wgpu` crate, which is a Rust wrapper around the WebGPU API.

The heat equation is a partial differential equation which describes the flow of heat in a given domain. For a 2D domain, it is given by the following equation:

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

where $u$ is the temperature, $t$ is time, $\alpha$ is the thermal diffusivity and $x$ and $y$ are the spatial coordinates. The equation is solved numerically using the finite difference method, which consists in discretizing the domain into a grid and approximating the derivatives by finite differences. The resulting system of equations is then solved using a Forward Euler method, which is written as a WGSL compute shader and runs on the GPU.

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
