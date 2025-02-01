use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

mod components;
mod math;
mod resources;
mod systems;

pub use components::*;
pub use math::*;
pub use resources::*;
pub use systems::*;

use std::f32::consts::PI;

/// Physics timestep configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct TimeConfig {
    /// Physics update rate in Hz
    pub physics_rate: f32,
    /// Derived fixed timestep in seconds
    pub fixed_timestep: f32,
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            physics_rate: 240.0,
            fixed_timestep: 1.0 / 240.0,
        }
    }
}

/// Wind tunnel and domain configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct DomainConfig {
    /// Size of the simulation domain [mm]
    pub box_size: Vec2,
    /// Free stream velocity [m/s]
    pub freestream_velocity: Vec2,
    /// Gravity vector [m/s²]
    pub gravity: Vec2,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            box_size: Vec2::new(200.0, 200.0),
            freestream_velocity: Vec2::new(50.0, 0.0),
            gravity: Vec2::new(0.0, 0.0),
        }
    }
}

/// Particle-specific configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct ParticleConfig {
    /// Particle radius [mm]
    pub radius: f32,
    /// Initial number of particles
    pub initial_count: usize,
    /// Number of particles to spawn per update
    pub spawn_rate: usize,
    /// Grid cell size for spatial partitioning
    pub grid_cell_size: f32,
    /// Particle mass [kg]
    pub mass: f32,
}

impl ParticleConfig {
    pub fn new(radius: f32, initial_count: usize, spawn_rate: usize, rest_density: f32) -> Self {
        Self {
            radius,
            initial_count,
            spawn_rate,
            grid_cell_size: radius * 4.0,
            mass: rest_density * (4.0 / 3.0) * PI * radius.powi(3),
        }
    }
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self::new(0.5, 600, 50, 1.225)
    }
}

/// SPH fluid properties configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct FluidConfig {
    /// Rest density of the fluid [kg/m³]
    pub rest_density: f32,
    /// Gas constant for the fluid [J/(kg·K)]
    pub gas_constant: f32,
    /// Smoothing length relative to particle radius
    pub smoothing_length: f32,
    /// Dynamic viscosity coefficient [Pa·s]
    pub viscosity: f32,
}

impl FluidConfig {
    pub fn new(particle_radius: f32) -> Self {
        Self {
            rest_density: 1.225,
            gas_constant: 287.05,
            smoothing_length: particle_radius * 16.0,
            viscosity: 1.81e-5,
        }
    }
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Visualization and rendering configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct RenderConfig {
    /// Scale factor for rendering
    pub render_scale: f32,
    /// Camera zoom limits
    pub min_zoom: f32,
    pub max_zoom: f32,
    /// Camera pan speed
    pub pan_speed: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            render_scale: 4.0,
            min_zoom: 0.1,
            max_zoom: 10.0,
            pan_speed: 150.0,
        }
    }
}

/// Airfoil configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct AirfoilConfig {
    /// Angle of attack [degrees]
    pub aoa: f32,
    /// Chord length [mm]
    pub chord_length: f32,
    /// Maximum thickness ratio
    pub thickness_ratio: f32,
    /// Position in the domain
    pub position: Vec2,
}

impl Default for AirfoilConfig {
    fn default() -> Self {
        Self {
            aoa: 5.0,
            chord_length: 50.0,
            thickness_ratio: 0.12,
            position: Vec2::ZERO,
        }
    }
}

/// Combined simulation configuration
#[derive(Resource, Clone, Debug, Copy)]
pub struct SPHConfig {
    pub time: TimeConfig,
    pub domain: DomainConfig,
    pub particle: ParticleConfig,
    pub fluid: FluidConfig,
    pub render: RenderConfig,
    pub airfoil: AirfoilConfig,
}

impl Default for SPHConfig {
    fn default() -> Self {
        Self {
            time: TimeConfig::default(),
            domain: DomainConfig::default(),
            particle: ParticleConfig::default(),
            fluid: FluidConfig::default(),
            render: RenderConfig::default(),
            airfoil: AirfoilConfig::default(),
        }
    }
}

pub struct SPHPlugin {
    pub config: SPHConfig,
}

impl Default for SPHPlugin {
    fn default() -> Self {
        Self {
            config: SPHConfig::default(),
        }
    }
}

impl Plugin for SPHPlugin {
    fn build(&self, app: &mut App) {
        // Insert configuration and derived constants
        app.insert_resource(self.config)
            .insert_resource(ParticlePool::new(self.config.particle.initial_count))
            .insert_resource(SpatialGrid::new(
                self.config.particle.grid_cell_size,
                self.config.particle.radius,
            ))
            .init_resource::<SimulationStats>()
            // Setup systems
            .add_systems(Startup, setup)
            // Fixed update systems
            .add_systems(
                FixedUpdate,
                (
                    update_spatial_grid,
                    compute_density_pressure,
                    apply_sph_forces,
                )
                    .chain()
                    .run_if(on_timer(Duration::from_secs_f32(
                        self.config.time.fixed_timestep,
                    ))),
            )
            .add_systems(Update, (spawn_particles, print_fps, pan_camera))
            .init_resource::<VisualizationMode>()
            .add_systems(
                Update,
                (handle_visualization_toggle, update_particle_colors),
            );
    }
}

// Re-export everything needed for the public API
pub mod prelude {
    pub use crate::{components::*, math::*, resources::*, SPHConfig, SPHPlugin};
}
