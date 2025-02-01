use crate::components::*;
use crate::math::*;
use crate::resources::*;
use crate::SPHConfig;
use crate::*;

use bevy::color::palettes::css::RED;
use rand::Rng;
use std::{collections::HashMap, sync::Mutex};

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    config: Res<SPHConfig>,
) {
    // Spawn the camera
    commands.spawn((
        Camera2d,
        Camera {
            hdr: true, // Enable HDR rendering for better visual quality
            ..default()
        },
        Transform {
            scale: Vec3::splat(1.0 / config.render.render_scale), // Apply uniform scaling to zoom in/out
            ..default()
        },
    ));

    let mut rng = rand::thread_rng();

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(
            config.domain.box_size.x,
            config.domain.box_size.y,
        ))),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(Color::WHITE))),
        Transform::from_xyz(0.0, 0.0, -1.0),
    ));

    let airfoil = SolidBoundary::new_airfoil(
        config.airfoil.chord_length,      // chord length [mm]
        config.airfoil.position,          // position
        -config.airfoil.aoa.to_radians(), // angle of attack [rad]
    );

    let num_points_per_surface = airfoil.points.len() / 2;

    // Draw upper surface
    for i in 0..num_points_per_surface - 1 {
        let start = airfoil.points[i * 2]; // Skip odd indices (lower surface points)
        let end = airfoil.points[(i + 1) * 2];
        let line_length = start.distance(end);
        let angle = (end - start).y.atan2((end - start).x);

        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(line_length, 0.2))),
            MeshMaterial2d(materials.add(ColorMaterial::from(Color::BLACK))),
            Transform::from_translation(((start + end) / 2.0).extend(0.1))
                .with_rotation(Quat::from_rotation_z(angle)),
        ));
    }

    // Draw lower surface
    for i in 0..num_points_per_surface - 1 {
        let start = airfoil.points[i * 2 + 1]; // Odd indices for lower surface
        let end = airfoil.points[(i + 1) * 2 + 1];
        let line_length = start.distance(end);
        let angle = (end - start).y.atan2((end - start).x);

        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(line_length, 0.2))),
            MeshMaterial2d(materials.add(ColorMaterial::from(Color::BLACK))),
            Transform::from_translation(((start + end) / 2.0).extend(0.1))
                .with_rotation(Quat::from_rotation_z(angle)),
        ));
    }

    // Connect trailing edge
    let last_upper = airfoil.points[(num_points_per_surface - 1) * 2];
    let last_lower = airfoil.points[(num_points_per_surface - 1) * 2 + 1];
    let line_length = last_upper.distance(last_lower);
    let angle = (last_lower - last_upper)
        .y
        .atan2((last_lower - last_upper).x);

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(line_length, 0.2))),
        MeshMaterial2d(materials.add(ColorMaterial::from(Color::BLACK))),
        Transform::from_translation(((last_upper + last_lower) / 2.0).extend(0.1))
            .with_rotation(Quat::from_rotation_z(angle)),
    ));

    commands.spawn(airfoil); // Add airfoil boundary
    commands.spawn(SolidBoundary::new_wind_tunnel(config.domain.box_size)); // Add wind tunnel walls

    for _ in 0..config.particle.initial_count {
        let y = rng.gen_range(-config.domain.box_size.y / 2.0..config.domain.box_size.y / 2.0);
        let position = Vec3::new(-config.domain.box_size.x / 2.0 + 0.0, y, 0.0);
        let velocity =
            wind_tunnel_velocity(y, config.domain.freestream_velocity, config.domain.box_size);

        commands.spawn((
            FluidParticle {
                velocity,
                density: config.fluid.rest_density,
                pressure: 0.0,
            },
            Transform::from_translation(position),
            Mesh2d(meshes.add(Circle::new(config.particle.radius))),
            MeshMaterial2d(materials.add(ColorMaterial::from_color(RED))),
        ));
    }
}

pub fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform)>,
    config: Res<SPHConfig>,
) {
    let mut new_cells: HashMap<(i32, i32), Vec<(Entity, Vec2)>> = HashMap::new();

    // Pre-calculate required cells and initialize them
    for (_, transform) in query.iter() {
        let pos = transform.translation.truncate();
        let cell = get_grid_cell(pos, config.particle.grid_cell_size);
        new_cells
            .entry(cell)
            .or_insert_with(|| Vec::with_capacity(grid.expected_particles_per_cell));
    }

    // Fill grid
    for (entity, transform) in query.iter() {
        let pos = transform.translation.truncate();
        let cell = get_grid_cell(pos, config.particle.grid_cell_size);
        if let Some(cell_vec) = new_cells.get_mut(&cell) {
            cell_vec.push((entity, pos));
        }
    }

    grid.cells = new_cells;
}

pub fn compute_density_pressure(
    grid: Res<SpatialGrid>,
    mut query: Query<(Entity, &Transform, &mut FluidParticle)>,
    config: Res<SPHConfig>,
) {
    let particle_data: Vec<_> = query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();

    query.par_iter_mut().for_each(|(entity, _, mut particle)| {
        let pos = particle_data.iter().find(|(e, _)| *e == entity).unwrap().1;
        let cell = get_grid_cell(pos, config.particle.grid_cell_size);

        let mut density = config.particle.mass * w_poly6(0.0, config.fluid.smoothing_length);

        // Check neighboring cells
        for &neighbor_cell in &get_neighboring_cells(cell) {
            if let Some(neighbors) = grid.cells.get(&neighbor_cell) {
                for &(other_entity, other_pos) in neighbors {
                    if entity != other_entity {
                        let r = pos.distance(other_pos);
                        if r < config.fluid.smoothing_length {
                            density +=
                                config.particle.mass * w_poly6(r, config.fluid.smoothing_length);
                        }
                    }
                }
            }
        }

        let min_density = config.fluid.rest_density * 0.01;
        particle.density = density.max(min_density);

        // Calculate pressure using the equation of state for an ideal gas
        // P = ρRT where R is the gas constant
        particle.pressure = config.fluid.gas_constant
            * particle.density
            * ((particle.density / config.fluid.rest_density).powf(7.0) - 1.0); // Modified equation of state

        if particle.pressure.abs() < 1e-6 {
            particle.pressure = 0.0; // Clean up numerical noise
        }
    });
}

fn apply_boundary_forces(
    particle_pos: Vec2,
    particle_vel: Vec2,
    particle_density: f32,
    boundary: &SolidBoundary,
    config: &SPHConfig,
) -> Vec2 {
    let mut force = Vec2::ZERO;
    let ds = config.particle.radius * 0.25; // Finer sampling

    const BOUNDARY_STIFFNESS: f32 = 500.0; // Increased stiffness
    const BOUNDARY_DAMPING: f32 = 1.0; // Increased damping
    let min_distance: f32 = config.particle.radius * 0.5; // Prevent complete penetration

    for i in 0..boundary.points.len() - 1 {
        let p1 = boundary.points[i];
        let p2 = boundary.points[i + 1];
        let segment_length = p1.distance(p2);
        let steps = (segment_length / ds).ceil() as usize;

        for step in 0..steps {
            let t = step as f32 / steps as f32;
            let boundary_point = p1.lerp(p2, t);
            let normal = boundary.normals[i];

            let r = particle_pos.distance(boundary_point);

            if r < config.fluid.smoothing_length {
                // Strong repulsion for very close particles
                let close_repulsion = if r < min_distance {
                    let penetration = (min_distance - r) / min_distance;
                    normal * BOUNDARY_STIFFNESS * 10.0 * penetration * penetration
                } else {
                    Vec2::ZERO
                };

                // Normal boundary force
                let q = r / config.fluid.smoothing_length;
                let pressure_force =
                    normal * BOUNDARY_STIFFNESS * (1.0 - q) * (1.0 - q) * (1.0 - q);

                // Enhanced damping near boundary
                let relative_vel = particle_vel;
                let damping_force = -relative_vel * BOUNDARY_DAMPING * (1.0 - q);

                // Combine all forces
                let force_contribution = (pressure_force + damping_force + close_repulsion) * ds;
                force += force_contribution * (particle_density / config.fluid.rest_density);
            }
        }
    }

    force * config.particle.mass
}

pub fn apply_sph_forces(
    mut commands: Commands,
    grid: Res<SpatialGrid>,
    boundary_query: Query<&SolidBoundary>,
    mut query: Query<(Entity, &mut Transform, &mut FluidParticle)>,
    mut particle_pool: ResMut<ParticlePool>,
    mesh_query: Query<(&Mesh2d, &MeshMaterial2d<ColorMaterial>)>,
    config: Res<SPHConfig>,
) {
    // Collect all particle data first to avoid borrow checker issues
    let particle_data: Vec<_> = query
        .iter()
        .map(|(entity, transform, particle)| {
            (
                entity,
                transform.translation.truncate(),
                particle.density,
                particle.pressure,
                particle.velocity,
            )
        })
        .collect();

    let forces = Mutex::new(Vec::with_capacity(particle_data.len()));
    let config = config.clone();

    // Parallel force computation
    query.par_iter().for_each(|(entity, _, _)| {
        let (pos, density, pressure, velocity) = particle_data
            .iter()
            .find(|(e, ..)| *e == entity)
            .map(|(_, pos, density, pressure, velocity)| (*pos, *density, *pressure, *velocity))
            .unwrap();

        let cell = get_grid_cell(pos, config.particle.grid_cell_size);
        let mut pressure_force = Vec2::ZERO;
        let mut viscosity_force = Vec2::ZERO;
        let mut boundary_force = Vec2::ZERO;

        for boundary in boundary_query.iter() {
            boundary_force += apply_boundary_forces(pos, velocity, density, boundary, &config);
        }

        // Compute forces with neighbors
        for &neighbor_cell in &get_neighboring_cells(cell) {
            if let Some(neighbors) = grid.cells.get(&neighbor_cell) {
                for &(other_entity, other_pos) in neighbors {
                    if entity != other_entity {
                        let r = pos.distance(other_pos);
                        if r < config.fluid.smoothing_length && r > 0.0 {
                            if let Some((_, _, other_density, other_pressure, other_velocity)) =
                                particle_data.iter().find(|(e, ..)| e == &other_entity)
                            {
                                let direction = (other_pos - pos) / r;

                                // Pressure force
                                let pressure_factor =
                                    (pressure + other_pressure) / (2.0 * density * other_density);
                                pressure_force -= direction
                                    * config.particle.mass
                                    * pressure_factor
                                    * w_spiky_gradient(r, config.fluid.smoothing_length);

                                // Viscosity force
                                let velocity_diff = *other_velocity - velocity;
                                viscosity_force += config.fluid.viscosity
                                    * config.particle.mass
                                    * (velocity_diff / other_density)
                                    * w_viscosity_laplacian(r, config.fluid.smoothing_length);
                            }
                        }
                    }
                }
            }
        }

        let total_force =
            pressure_force + viscosity_force + boundary_force + config.domain.gravity * density;
        forces.lock().unwrap().push((entity, total_force));
    });

    let forces = forces.lock().unwrap();
    let to_despawn = Mutex::new(Vec::new());

    // Apply forces and update positions in parallel
    query
        .par_iter_mut()
        .for_each(|(entity, mut transform, mut particle)| {
            if let Some(&(_, force)) = forces.iter().find(|(e, _)| *e == entity) {
                let dt = config.time.fixed_timestep;

                // Update velocity
                let density = particle.density;
                particle.velocity += force * dt / density;

                // Maintain inlet conditions
                let inlet_zone = -config.domain.box_size.x / 2.0;
                if transform.translation.x < inlet_zone {
                    particle.velocity = wind_tunnel_velocity(
                        transform.translation.y,
                        config.domain.freestream_velocity,
                        config.domain.box_size,
                    );
                }

                // Update position
                let mut new_pos = transform.translation.truncate() + particle.velocity * dt;

                let half_height = config.domain.box_size.y / 2.0 - config.particle.radius;
                new_pos.y = new_pos.y.clamp(-half_height, half_height);

                // Despawn particle if it goes out of bounds
                if new_pos.x + config.particle.radius > config.domain.box_size.x / 2.0 {
                    to_despawn.lock().unwrap().push(entity);
                } else {
                    transform.translation = new_pos.extend(0.0);
                }
            }
        });

    for entity in to_despawn.lock().unwrap().iter() {
        if let Ok((mesh, material)) = mesh_query.get(*entity) {
            particle_pool
                .available
                .push((*entity, mesh.0.clone(), material.0.clone()));
            commands
                .entity(*entity)
                .remove::<(Transform, FluidParticle)>();
        }
    }
}

pub fn spawn_particles(
    mut commands: Commands,
    mut pool: ResMut<ParticlePool>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    visualization_mode: Res<VisualizationMode>,
    config: Res<SPHConfig>,
) {
    let mut rng = rand::thread_rng();

    for _ in 0..config.particle.spawn_rate {
        let y = rng.gen_range(-config.domain.box_size.y / 2.0..config.domain.box_size.y / 2.0);
        let position = Vec3::new(-config.domain.box_size.x / 2.0 + 0.0, y, 0.0);
        let velocity =
            wind_tunnel_velocity(y, config.domain.freestream_velocity, config.domain.box_size);
        let particle = FluidParticle {
            velocity,
            density: config.fluid.rest_density,
            pressure: 0.0,
        };

        // Calculate initial color based on visualization mode
        let color = match *visualization_mode {
            VisualizationMode::Velocity => Color::hsl(240.0, 1.0, 0.5), // Blue
            VisualizationMode::Pressure => Color::hsl(120.0, 1.0, 0.5), // Green
            VisualizationMode::Density => Color::hsl(270.0, 1.0, 0.5),  // Purple
        };

        if let Some((entity, _mesh, _material)) = pool.available.pop() {
            commands
                .entity(entity)
                .insert((particle, Transform::from_translation(position)));
        } else if pool.available.len() < pool.capacity {
            let mesh = meshes.add(Circle::new(config.particle.radius));
            let material = materials.add(ColorMaterial::from_color(color));

            commands.spawn((
                particle,
                Transform::from_translation(position),
                Mesh2d(mesh.clone()),
                MeshMaterial2d(material.clone()),
            ));
        }
    }
}

pub fn update_particle_colors(
    visualization_mode: Res<VisualizationMode>,
    mut query: Query<(&FluidParticle, &mut MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    config: Res<SPHConfig>,
) {
    for (particle, material) in query.iter_mut() {
        let color = match *visualization_mode {
            VisualizationMode::Velocity => {
                let velocity_magnitude = particle.velocity.length();
                let freestream = config.domain.freestream_velocity.length();
                // Normalize velocity relative to freestream velocity
                let normalized = (velocity_magnitude / (3.0 * freestream)).clamp(0.0, 1.0);
                // HSL: Blue (240°) to Red (0°), full saturation, 50% lightness
                Color::hsl(240.0 * (1.0 - normalized), 1.0, 0.5)
            }
            VisualizationMode::Pressure => {
                let particle_dynamic_pressure =
                    0.5 * particle.density * particle.velocity.length_squared();
                let total_pressure = particle.pressure + particle_dynamic_pressure;

                // Normalize pressure from -1.0 to 1.0 relative to max q
                let max_q = 0.5
                    * config.fluid.rest_density
                    * config.domain.freestream_velocity.length_squared();
                let normalized = (total_pressure / (max_q * 0.2)).clamp(-1.0, 1.0);
                // Map -1 to 1 to hue values: blue -> white -> red
                let hue = if normalized < 0.0 {
                    // Negative pressure: blue to white
                    240.0 + (normalized * 240.0)
                } else {
                    // Positive pressure: white to red
                    240.0 - (normalized * 240.0)
                };
                Color::hsl(hue.clamp(0.0, 360.0), 1.0, 0.5)
            }
            VisualizationMode::Density => {
                // Normalize density relative to rest density
                let normalized =
                    ((particle.density / config.fluid.rest_density) - 1.0).clamp(-0.1, 0.1) * 2.0; // Scale to -1.0 to 1.0
                let base_hue = 270.0;
                let hue = base_hue - (normalized * 210.0); // HSL: Purple (270°) to Yellow (60°), full saturation, 50% lightness
                Color::hsl(hue.clamp(60.0, 270.0), 1.0, 0.5)
            }
        };
        materials.get_mut(&material.0).unwrap().color = color;
    }
}

pub fn handle_visualization_toggle(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut visualization_mode: ResMut<VisualizationMode>,
) {
    if keyboard.just_pressed(KeyCode::Tab) {
        *visualization_mode = match *visualization_mode {
            VisualizationMode::Velocity => VisualizationMode::Pressure,
            VisualizationMode::Pressure => VisualizationMode::Density,
            VisualizationMode::Density => VisualizationMode::Velocity,
        };
    }
}

pub fn print_fps(
    time: Res<Time>,
    query: Query<&FluidParticle>,
    mut stats: ResMut<SimulationStats>,
) {
    if time.elapsed_secs() - stats.last_update >= 1.0 {
        stats.fps = 1.0 / time.delta_secs();
        stats.particle_count = query.iter().count();
        stats.last_update = time.elapsed_secs();

        info!("FPS: {:.0}, Particles: {}", stats.fps, stats.particle_count);
    }
}

pub fn pan_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
) {
    let (mut transform, mut projection) = query.single_mut();
    let mut direction = Vec3::ZERO;
    let speed = 150.0;

    if keyboard.pressed(KeyCode::ArrowRight) {
        direction.x += 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowLeft) {
        direction.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        direction.y += 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        direction.y -= 1.0;
    }

    if direction != Vec3::ZERO {
        transform.translation +=
            direction.normalize() * speed * time.delta_secs() * projection.scale;
    }

    // Zoom controls with limits
    let min_scale = 0.1;
    let max_scale = 10.0;

    if keyboard.pressed(KeyCode::Equal) {
        projection.scale = (projection.scale * 0.95).max(min_scale);
    }
    if keyboard.pressed(KeyCode::Minus) {
        projection.scale = (projection.scale * 1.05).min(max_scale);
    }
}
