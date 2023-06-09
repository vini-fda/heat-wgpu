# heat-wgpu

This is a Rust program which leverages computing and rendering capabilities of modern GPUs to solve the heat equation PDE and display results in real time. It is based on the `wgpu` crate, which is a Rust wrapper around the WebGPU API.

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
