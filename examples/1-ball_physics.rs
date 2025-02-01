use bevy::color::palettes::css::RED;
use bevy::prelude::*;
use rand::Rng;
use std::collections::HashMap;

const BOX_SIZE: Vec2 = Vec2::new(500.0, 500.0);
const SPAWN_HOLDOFF: f32 = 30.0;
const BALL_RADIUS: f32 = 4.0;
const BALL_COUNT: usize = 1000;
const RESTITUTION: f32 = 0.95;
const VEL_RANGE: f32 = 300.0;
const GRID_CELL_SIZE: f32 = BALL_RADIUS * 4.0; // Size of each grid cell

const RENDER_SCALE: f32 = 1.0;

#[derive(Component)]
struct Ball {
    velocity: Vec2,
}

#[derive(Resource)]
struct SpatialGrid {
    cells: HashMap<(i32, i32), Vec<(Entity, Vec2)>>,
}

impl Default for SpatialGrid {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }
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
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_spatial_grid,
                update_balls.after(update_spatial_grid),
                print_fps,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn the camera
    commands.spawn(Camera2d::default());

    let mut rng = rand::thread_rng();

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(BOX_SIZE.x, BOX_SIZE.y))),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(Color::WHITE))),
        Transform::from_xyz(0.0, 0.0, -1.0),
    ));

    for _ in 0..BALL_COUNT {
        let position = Vec3::new(
            rng.gen_range(-(BOX_SIZE.x - SPAWN_HOLDOFF) / 2.0..(BOX_SIZE.x - SPAWN_HOLDOFF) / 2.0),
            rng.gen_range(-(BOX_SIZE.y - SPAWN_HOLDOFF) / 2.0..(BOX_SIZE.y - SPAWN_HOLDOFF) / 2.0),
            0.0,
        );
        let velocity = Vec2::new(
            rng.gen_range(-VEL_RANGE..VEL_RANGE),
            rng.gen_range(-VEL_RANGE..VEL_RANGE),
        );

        commands.spawn((
            Ball { velocity },
            Transform::from_translation(position),
            Mesh2d(meshes.add(Circle::new(BALL_RADIUS))),
            MeshMaterial2d(materials.add(ColorMaterial::from_color(RED))),
        ));
    }
}

fn get_grid_cell(position: Vec2) -> (i32, i32) {
    (
        (position.x / GRID_CELL_SIZE).floor() as i32,
        (position.y / GRID_CELL_SIZE).floor() as i32,
    )
}

fn update_spatial_grid(mut grid: ResMut<SpatialGrid>, query: Query<(Entity, &Transform)>) {
    grid.cells.clear();

    // Collect positions in parallel
    let positions: Vec<_> = query.iter().collect();
    positions.iter().for_each(|&(entity, transform)| {
        let pos = transform.translation.truncate();
        let cell = get_grid_cell(pos);
        grid.cells.entry(cell).or_default().push((entity, pos));
    });
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

fn update_balls(
    time: Res<Time>,
    grid: Res<SpatialGrid>,
    mut query: Query<(Entity, &mut Ball, &mut Transform)>,
) {
    query
        .par_iter_mut()
        .for_each(|(entity, mut ball, mut transform)| {
            let mut velocity = ball.velocity;
            let mut position = transform.translation.truncate();
            let current_cell = get_grid_cell(position);

            // Process collisions for this ball
            for neighbor_cell in get_neighboring_cells(current_cell).iter() {
                if let Some(neighbors) = grid.cells.get(neighbor_cell) {
                    for &(other_entity, other_pos) in neighbors {
                        if other_entity != entity {
                            let distance = position.distance(other_pos);
                            if distance < BALL_RADIUS * 2.0 {
                                let normal = (other_pos - position).normalize();
                                velocity = -velocity * RESTITUTION;

                                // Separate balls
                                let overlap = BALL_RADIUS * 2.0 - distance;
                                let separation = normal * (overlap * 0.5);
                                position -= separation;
                            }
                        }
                    }
                }
            }

            // Update position
            position += velocity * time.delta_secs();

            // Wall collisions
            if position.x - BALL_RADIUS < -BOX_SIZE.x / 2.0 {
                velocity.x = velocity.x.abs() * RESTITUTION;
                position.x = -BOX_SIZE.x / 2.0 + BALL_RADIUS;
            } else if position.x + BALL_RADIUS > BOX_SIZE.x / 2.0 {
                velocity.x = -velocity.x.abs() * RESTITUTION;
                position.x = BOX_SIZE.x / 2.0 - BALL_RADIUS;
            }

            if position.y - BALL_RADIUS < -BOX_SIZE.y / 2.0 {
                velocity.y = velocity.y.abs() * RESTITUTION;
                position.y = -BOX_SIZE.y / 2.0 + BALL_RADIUS;
            } else if position.y + BALL_RADIUS > BOX_SIZE.y / 2.0 {
                velocity.y = -velocity.y.abs() * RESTITUTION;
                position.y = BOX_SIZE.y / 2.0 - BALL_RADIUS;
            }

            // Apply updates directly
            ball.velocity = velocity;
            transform.translation = position.extend(0.0);
        });
}

fn print_fps(time: Res<Time>) {
    // Print FPS every second
    if time.elapsed_secs() % 1.0 < time.delta_secs() {
        let fps = 1.0 / time.delta_secs();
        info!("FPS: {:.0}", fps);
    }
}
