# Bevy SPH Simulator

A Smoothed Particle Hydrodynamics (SPH) fluid simulation implemented in Rust using the Bevy game engine. The simulation includes airfoil aerodynamics and real-time visualization capabilities. I built this to test Bevy's parallel processing capabilities.

![Demo](assets/sph.gif)

## Features

- Real-time SPH fluid simulation
- Interactive airfoil with configurable parameters
- Multiple visualization modes:
  - Velocity field
  - Pressure distribution
  - Density field
- Camera controls for panning and zooming
- Particle pooling for efficient memory management
- Spatial partitioning for optimized collision detection

## Requirements

- [Rust](https://www.rust-lang.org/tools/install)
- Cargo, installed with above Rust install.

## Installation

Clone the repository and build with Cargo:

```bash
git clone https://github.com/yourusername/bevy-sph
cd bevy-sph
cargo build --release
```

## Running Examples

The project includes two example simulations:

1. Basic ball physics:
```bash
cargo run --example 1-ball_physics
```

2. SPH Airfoil simulation:
```bash
cargo run --example 2-SPH_Airfoil
```

## Controls

- Arrow keys: Pan camera
- +/- keys: Zoom in/out
- Tab: Cycle through visualization modes (Velocity → Pressure → Density)

## Configuration

The simulation parameters can be customized in `lib.rs`. Key parameters include:

- Fluid properties (density, viscosity, etc.)
- Particle properties (size, count, spawn rate)
- Domain size and freestream velocity
- Airfoil properties (angle of attack, chord length)

## License

MIT OR Apache-2.0

## Performance Notes

The simulation uses spatial partitioning and parallel processing for optimal performance. For best results, run in release mode:

```bash
cargo run --release --example 2-SPH_Airfoil
```
