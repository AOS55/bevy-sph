use bevy::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;

#[derive(Resource)]
pub struct ParticlePool {
    pub available: Vec<(Entity, Handle<Mesh>, Handle<ColorMaterial>)>,
    pub capacity: usize,
}

impl ParticlePool {
    pub fn new(particle_count: usize) -> Self {
        Self {
            available: Vec::with_capacity(particle_count * 2),
            capacity: particle_count * 2,
        }
    }
}

#[derive(Resource)]
pub struct SpatialGrid {
    pub cells: HashMap<(i32, i32), Vec<(Entity, Vec2)>>,
    pub expected_particles_per_cell: usize,
}

impl SpatialGrid {
    pub fn new(grid_cell_size: f32, particle_radius: f32) -> Self {
        // Calculate expected particles per cell based on density and cell size
        let cell_area = grid_cell_size * grid_cell_size;
        let particle_area = PI * particle_radius * particle_radius;
        let expected_per_cell = (cell_area / particle_area * 0.5) as usize; // 50% packing density

        Self {
            cells: HashMap::new(),
            expected_particles_per_cell: expected_per_cell.max(50), // At least 50 particles
        }
    }
}

#[derive(Resource, Default)]
pub enum VisualizationMode {
    #[default]
    Velocity,
    Pressure,
    Density,
}

#[derive(Resource)]
pub struct SimulationStats {
    pub particle_count: usize,
    pub fps: f32,
    pub last_update: f32,
}

impl Default for SimulationStats {
    fn default() -> Self {
        Self {
            particle_count: 0,
            fps: 0.0,
            last_update: 0.0,
        }
    }
}
