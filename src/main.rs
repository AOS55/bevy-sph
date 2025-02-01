use bevy::color::palettes::css::RED;
use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use rand::Rng;
use std::time::Duration;
use std::{collections::HashMap, f32::consts::PI, sync::Mutex};

// Simulation Constants
const FIXED_TIMESTEP: f32 = 1.0 / 240.0; // 240 Hz physics update
const BOX_SIZE: Vec2 = Vec2::new(200.0, 200.0); // [mm]
const FREESTREAM_VELOCITY: Vec2 = Vec2::new(50.0, 0.0); // [m/s]
const PARTICLE_RADIUS: f32 = 0.5; // [mm]
const INITIAL_PARTICLE_COUNT: usize = 600;
const PARTICLES_PER_UPDATE: usize = 50; // This is essentially the flow rate
const GRID_CELL_SIZE: f32 = PARTICLE_RADIUS * 4.0; // Size of each grid cell for grid cell

// Rendering Constants
const RENDER_SCALE: f32 = 4.0;

// SPH Constants
const REST_DENSITY: f32 = 1.225; // Resting air density
const GAS_CONSTANT: f32 = 287.05; // Gas constant for air
const SMOOTHING_LENGTH: f32 = 16.0 * PARTICLE_RADIUS; // Influence Radius
const VISCOSITY: f32 = 1.81e-5; // Viscosity coefficient
const GRAVITY: Vec2 = Vec2::new(0.0, -0.0); // Gravity
const MASS: f32 =
    REST_DENSITY * (4.0 / 3.0) * PI * (PARTICLE_RADIUS * PARTICLE_RADIUS * PARTICLE_RADIUS);

// Airfoil Constants
const AOA: f32 = 5.0; // Angle of attack [deg]

#[derive(Component)]
struct FluidParticle {
    velocity: Vec2,
    density: f32,
    pressure: f32,
}

impl Default for FluidParticle {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            density: REST_DENSITY,
            pressure: 0.0,
        }
    }
}

#[derive(Resource)]
struct ParticlePool {
    available: Vec<(Entity, Handle<Mesh>, Handle<ColorMaterial>)>,
    capacity: usize,
}

impl Default for ParticlePool {
    fn default() -> Self {
        Self {
            available: Vec::with_capacity(INITIAL_PARTICLE_COUNT * 2),
            capacity: INITIAL_PARTICLE_COUNT * 2,
        }
    }
}

#[derive(Resource)]
struct SpatialGrid {
    cells: HashMap<(i32, i32), Vec<(Entity, Vec2)>>,
    expected_particles_per_cell: usize,
}

impl Default for SpatialGrid {
    fn default() -> Self {
        // Calculate expected particles per cell based on density and cell size
        let cell_area = GRID_CELL_SIZE * GRID_CELL_SIZE;
        let particle_area = PI * PARTICLE_RADIUS * PARTICLE_RADIUS;
        let expected_per_cell = (cell_area / particle_area * 0.5) as usize; // 50% packing density

        Self {
            cells: HashMap::new(),
            expected_particles_per_cell: expected_per_cell.max(50), // At least 50 particles
        }
    }
}

#[derive(Component)]
struct SolidBoundary {
    points: Vec<Vec2>,  // Boundary points defining the shape
    normals: Vec<Vec2>, // Normal vectors at each boundary point
}

impl SolidBoundary {
    fn new_airfoil(chord: f32, position: Vec2, angle_of_attack: f32) -> Self {
        // NACA 0012 airfoil coordinates (simplified)
        let num_points = 50;
        let mut points = Vec::with_capacity(num_points);
        let mut normals = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let x = (i as f32) / (num_points as f32 - 1.0);
            // NACA 0012 equation
            let t = 0.12; // maximum thickness
            let y = 5.0
                * t
                * chord
                * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x.powi(2) + 0.2843 * x.powi(3)
                    - 0.1015 * x.powi(4));

            let point = Vec2::new(x * chord - chord / 2.0, y);

            // Add points for both upper and lower surface
            points.push(rotate_point(point, angle_of_attack) + position);
            points.push(rotate_point(Vec2::new(point.x, -point.y), angle_of_attack) + position);

            // Calculate normals (perpendicular to surface)
            let dx = 0.01;
            let dy = if i < num_points - 1 {
                let next_y = 5.0
                    * t
                    * chord
                    * (0.2969 * (x + dx).sqrt() - 0.1260 * (x + dx) - 0.3516 * (x + dx).powi(2)
                        + 0.2843 * (x + dx).powi(3)
                        - 0.1015 * (x + dx).powi(4));
                (next_y - y) / dx
            } else {
                let prev_x = x - dx;
                let prev_y = 5.0
                    * t
                    * chord
                    * (0.2969 * prev_x.sqrt() - 0.1260 * prev_x - 0.3516 * prev_x.powi(2)
                        + 0.2843 * prev_x.powi(3)
                        - 0.1015 * prev_x.powi(4));
                (y - prev_y) / dx
            };

            let normal = Vec2::new(-dy, 1.0).normalize();
            normals.push(rotate_vector(normal, angle_of_attack));
            normals.push(rotate_vector(
                Vec2::new(-dy, -1.0).normalize(),
                angle_of_attack,
            ));
        }

        Self { points, normals }
    }

    fn new_wind_tunnel(box_size: Vec2) -> Self {
        let half_width = box_size.x / 2.0;
        let half_height = box_size.y / 2.0;

        // Define the walls
        let points = vec![
            // Top wall
            Vec2::new(-half_width, half_height),
            Vec2::new(half_width, half_height),
            // Bottom wall
            Vec2::new(-half_width, -half_height),
            Vec2::new(half_width, -half_height),
        ];

        // Define normals (pointing into the fluid)
        let normals = vec![
            // Top wall normal (pointing down)
            Vec2::new(0.0, -1.0),
            Vec2::new(0.0, -1.0),
            // Bottom wall normal (pointing up)
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        Self { points, normals }
    }
}

fn rotate_point(point: Vec2, angle_rad: f32) -> Vec2 {
    let cos = angle_rad.cos();
    let sin = angle_rad.sin();
    Vec2::new(point.x * cos - point.y * sin, point.x * sin + point.y * cos)
}

fn rotate_vector(vector: Vec2, angle_rad: f32) -> Vec2 {
    rotate_point(vector, angle_rad)
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (BOX_SIZE.x * RENDER_SCALE, BOX_SIZE.y * RENDER_SCALE).into(),
                title: "SPH Simulation".to_string(),
                resizable: false,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<SpatialGrid>()
        .init_resource::<ParticlePool>()
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            (
                update_spatial_grid,
                compute_density_pressure,
                apply_sph_forces,
            )
                .chain()
                .run_if(on_timer(Duration::from_secs_f32(FIXED_TIMESTEP))),
        )
        .add_systems(Update, (spawn_particles, print_fps, pan_camera))
        .init_resource::<VisualizationMode>()
        .add_systems(
            Update,
            (handle_visualization_toggle, update_particle_colors),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn the camera
    commands.spawn((
        Camera2d,
        Camera {
            hdr: true, // Enable HDR rendering for better visual quality
            ..default()
        },
        Transform {
            scale: Vec3::splat(1.0 / RENDER_SCALE), // Apply uniform scaling to zoom in/out
            ..default()
        },
    ));

    let mut rng = rand::thread_rng();

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(BOX_SIZE.x, BOX_SIZE.y))),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(Color::WHITE))),
        Transform::from_xyz(0.0, 0.0, -1.0),
    ));

    let airfoil = SolidBoundary::new_airfoil(
        50.0,                // chord length [mm]
        Vec2::new(0.0, 0.0), // position
        -AOA.to_radians(),   // angle of attack [rad]
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
    commands.spawn(SolidBoundary::new_wind_tunnel(BOX_SIZE)); // Add wind tunnel walls

    for _ in 0..INITIAL_PARTICLE_COUNT {
        let y = rng.gen_range(-BOX_SIZE.y / 2.0..BOX_SIZE.y / 2.0);
        let position = Vec3::new(-BOX_SIZE.x / 2.0 + 0.0, y, 0.0);
        let velocity = wind_tunnel_velocity(y);

        commands.spawn((
            FluidParticle {
                velocity,
                density: REST_DENSITY,
                pressure: 0.0,
            },
            Transform::from_translation(position),
            Mesh2d(meshes.add(Circle::new(PARTICLE_RADIUS))),
            MeshMaterial2d(materials.add(ColorMaterial::from_color(RED))),
        ));
    }
}

fn wind_tunnel_velocity(y: f32) -> Vec2 {
    let max_speed = FREESTREAM_VELOCITY.x;
    let normalized_y = (y + BOX_SIZE.y / 2.0) / BOX_SIZE.y;
    let velocity_x = max_speed * (1.0 - (2.0 * normalized_y - 1.0).powi(2));
    Vec2::new(velocity_x, 0.0)
}

fn get_grid_cell(position: Vec2) -> (i32, i32) {
    (
        (position.x / GRID_CELL_SIZE).floor() as i32,
        (position.y / GRID_CELL_SIZE).floor() as i32,
    )
}

fn update_spatial_grid(mut grid: ResMut<SpatialGrid>, query: Query<(Entity, &Transform)>) {
    let mut new_cells: HashMap<(i32, i32), Vec<(Entity, Vec2)>> = HashMap::new();

    // Pre-calculate required cells and initialize them
    for (_, transform) in query.iter() {
        let pos = transform.translation.truncate();
        let cell = get_grid_cell(pos);
        new_cells
            .entry(cell)
            .or_insert_with(|| Vec::with_capacity(grid.expected_particles_per_cell));
    }

    // Fill grid
    for (entity, transform) in query.iter() {
        let pos = transform.translation.truncate();
        let cell = get_grid_cell(pos);
        if let Some(cell_vec) = new_cells.get_mut(&cell) {
            cell_vec.push((entity, pos));
        }
    }

    grid.cells = new_cells;
}

fn compute_density_pressure(
    grid: Res<SpatialGrid>,
    mut query: Query<(Entity, &Transform, &mut FluidParticle)>,
) {
    let particle_data: Vec<_> = query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();

    query.par_iter_mut().for_each(|(entity, _, mut particle)| {
        let pos = particle_data.iter().find(|(e, _)| *e == entity).unwrap().1;
        let cell = get_grid_cell(pos);

        let mut density = MASS * w_poly6(0.0, SMOOTHING_LENGTH);

        // Check neighboring cells
        for &neighbor_cell in &get_neighboring_cells(cell) {
            if let Some(neighbors) = grid.cells.get(&neighbor_cell) {
                for &(other_entity, other_pos) in neighbors {
                    if entity != other_entity {
                        let r = pos.distance(other_pos);
                        if r < SMOOTHING_LENGTH {
                            density += MASS * w_poly6(r, SMOOTHING_LENGTH);
                        }
                    }
                }
            }
        }

        let min_density = REST_DENSITY * 0.01;
        particle.density = density.max(min_density);

        // Calculate pressure using the equation of state for an ideal gas
        // P = ρRT where R is the gas constant
        particle.pressure =
            GAS_CONSTANT * particle.density * ((particle.density / REST_DENSITY).powf(7.0) - 1.0); // Modified equation of state

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
) -> Vec2 {
    let mut force = Vec2::ZERO;
    let ds = PARTICLE_RADIUS * 0.25; // Finer sampling

    const BOUNDARY_STIFFNESS: f32 = 500.0; // Increased stiffness
    const BOUNDARY_DAMPING: f32 = 1.0; // Increased damping
    const MIN_DISTANCE: f32 = PARTICLE_RADIUS * 0.5; // Prevent complete penetration

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

            if r < SMOOTHING_LENGTH {
                // Strong repulsion for very close particles
                let close_repulsion = if r < MIN_DISTANCE {
                    let penetration = (MIN_DISTANCE - r) / MIN_DISTANCE;
                    normal * BOUNDARY_STIFFNESS * 10.0 * penetration * penetration
                } else {
                    Vec2::ZERO
                };

                // Normal boundary force
                let q = r / SMOOTHING_LENGTH;
                let pressure_force =
                    normal * BOUNDARY_STIFFNESS * (1.0 - q) * (1.0 - q) * (1.0 - q);

                // Enhanced damping near boundary
                let relative_vel = particle_vel;
                let damping_force = -relative_vel * BOUNDARY_DAMPING * (1.0 - q);

                // Combine all forces
                let force_contribution = (pressure_force + damping_force + close_repulsion) * ds;
                force += force_contribution * (particle_density / REST_DENSITY);
            }
        }
    }

    force * MASS
}

fn apply_sph_forces(
    mut commands: Commands,
    grid: Res<SpatialGrid>,
    boundary_query: Query<&SolidBoundary>,
    mut query: Query<(Entity, &mut Transform, &mut FluidParticle)>,
    mut particle_pool: ResMut<ParticlePool>,
    mesh_query: Query<(&Mesh2d, &MeshMaterial2d<ColorMaterial>)>,
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

    // Parallel force computation
    query.par_iter().for_each(|(entity, _, _)| {
        let (pos, density, pressure, velocity) = particle_data
            .iter()
            .find(|(e, ..)| *e == entity)
            .map(|(_, pos, density, pressure, velocity)| (*pos, *density, *pressure, *velocity))
            .unwrap();

        let cell = get_grid_cell(pos);
        let mut pressure_force = Vec2::ZERO;
        let mut viscosity_force = Vec2::ZERO;
        let mut boundary_force = Vec2::ZERO;

        for boundary in boundary_query.iter() {
            boundary_force += apply_boundary_forces(pos, velocity, density, boundary);
        }

        // Compute forces with neighbors
        for &neighbor_cell in &get_neighboring_cells(cell) {
            if let Some(neighbors) = grid.cells.get(&neighbor_cell) {
                for &(other_entity, other_pos) in neighbors {
                    if entity != other_entity {
                        let r = pos.distance(other_pos);
                        if r < SMOOTHING_LENGTH && r > 0.0 {
                            if let Some((_, _, other_density, other_pressure, other_velocity)) =
                                particle_data.iter().find(|(e, ..)| e == &other_entity)
                            {
                                let direction = (other_pos - pos) / r;

                                // Pressure force
                                let pressure_factor =
                                    (pressure + other_pressure) / (2.0 * density * other_density);
                                pressure_force -= direction
                                    * MASS
                                    * pressure_factor
                                    * w_spiky_gradient(r, SMOOTHING_LENGTH);

                                // Viscosity force
                                let velocity_diff = *other_velocity - velocity;
                                viscosity_force += VISCOSITY
                                    * MASS
                                    * (velocity_diff / other_density)
                                    * w_viscosity_laplacian(r, SMOOTHING_LENGTH);
                            }
                        }
                    }
                }
            }
        }

        let total_force = pressure_force + viscosity_force + boundary_force + GRAVITY * density;
        forces.lock().unwrap().push((entity, total_force));
    });

    let forces = forces.lock().unwrap();
    let to_despawn = Mutex::new(Vec::new());

    // Apply forces and update positions in parallel
    query
        .par_iter_mut()
        .for_each(|(entity, mut transform, mut particle)| {
            if let Some(&(_, force)) = forces.iter().find(|(e, _)| *e == entity) {
                let dt = FIXED_TIMESTEP;

                // Update velocity
                let density = particle.density;
                particle.velocity += force * dt / density;

                // Maintain inlet conditions
                let inlet_zone = -BOX_SIZE.x / 2.0;
                if transform.translation.x < inlet_zone {
                    particle.velocity = wind_tunnel_velocity(transform.translation.y);
                }

                // Update position
                let mut new_pos = transform.translation.truncate() + particle.velocity * dt;

                let half_height = BOX_SIZE.y / 2.0 - PARTICLE_RADIUS;
                new_pos.y = new_pos.y.clamp(-half_height, half_height);

                // Despawn particle if it goes out of bounds
                if new_pos.x + PARTICLE_RADIUS > BOX_SIZE.x / 2.0 {
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

fn spawn_particles(
    mut commands: Commands,
    mut pool: ResMut<ParticlePool>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    visualization_mode: Res<VisualizationMode>,
) {
    let mut rng = rand::thread_rng();

    for _ in 0..PARTICLES_PER_UPDATE {
        let y = rng.gen_range(-BOX_SIZE.y / 2.0..BOX_SIZE.y / 2.0);
        let position = Vec3::new(-BOX_SIZE.x / 2.0 + 0.0, y, 0.0);
        let velocity = wind_tunnel_velocity(y);
        let particle = FluidParticle {
            velocity,
            density: REST_DENSITY,
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
            let mesh = meshes.add(Circle::new(PARTICLE_RADIUS));
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

fn get_neighboring_cells(cell: (i32, i32)) -> [(i32, i32); 9] {
    [
        (cell.0 - 1, cell.1 - 1),
        (cell.0, cell.1 - 1),
        (cell.0 + 1, cell.1 - 1),
        (cell.0 - 1, cell.1),
        (cell.0, cell.1),
        (cell.0 + 1, cell.1),
        (cell.0 - 1, cell.1 + 1),
        (cell.0, cell.1 + 1),
        (cell.0 + 1, cell.1 + 1),
    ]
}

fn w_poly6(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = 315.0 / (64.0 * PI * h.powi(9));
    coef * (h * h - r * r).powi(3)
}

fn w_spiky_gradient(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = -45.0 / (PI * h.powi(6));
    coef * (h - r).powi(2)
}

fn w_viscosity_laplacian(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = 45.0 / (PI * h.powi(6));
    coef * (h - r)
}

fn print_fps(time: Res<Time>, query: Query<&FluidParticle>) {
    // Print FPS every second
    if time.elapsed_secs() % 1.0 < time.delta_secs() {
        let fps = 1.0 / time.delta_secs();
        let particles = query.iter().count();
        info!("FPS: {:.0}, Particles: {:?}", fps, particles);
    }
}

fn pan_camera(
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

#[derive(Resource, Default)]
enum VisualizationMode {
    #[default]
    Velocity,
    Pressure,
    Density,
}

fn update_particle_colors(
    visualization_mode: Res<VisualizationMode>,
    mut query: Query<(&FluidParticle, &mut MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (particle, material) in query.iter_mut() {
        let color = match *visualization_mode {
            VisualizationMode::Velocity => {
                let velocity_magnitude = particle.velocity.length();
                let freestream = FREESTREAM_VELOCITY.length();
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
                let max_q = 0.5 * REST_DENSITY * FREESTREAM_VELOCITY.length_squared();
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
                let normalized = ((particle.density / REST_DENSITY) - 1.0).clamp(-0.1, 0.1) * 2.0; // Scale to -1.0 to 1.0
                let base_hue = 270.0;
                let hue = base_hue - (normalized * 210.0); // HSL: Purple (270°) to Yellow (60°), full saturation, 50% lightness
                Color::hsl(hue.clamp(60.0, 270.0), 1.0, 0.5)
            }
        };
        materials.get_mut(&material.0).unwrap().color = color;
    }
}

fn handle_visualization_toggle(
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
