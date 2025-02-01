use bevy::prelude::*;
use std::f32::consts::PI;

pub fn rotate_point(point: Vec2, angle_rad: f32) -> Vec2 {
    let cos = angle_rad.cos();
    let sin = angle_rad.sin();
    Vec2::new(point.x * cos - point.y * sin, point.x * sin + point.y * cos)
}

pub fn rotate_vector(vector: Vec2, angle_rad: f32) -> Vec2 {
    rotate_point(vector, angle_rad)
}

pub fn w_poly6(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = 315.0 / (64.0 * PI * h.powi(9));
    coef * (h * h - r * r).powi(3)
}

pub fn w_spiky_gradient(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = -45.0 / (PI * h.powi(6));
    coef * (h - r).powi(2)
}

pub fn w_viscosity_laplacian(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let coef = 45.0 / (PI * h.powi(6));
    coef * (h - r)
}

pub fn get_grid_cell(position: Vec2, grid_cell_size: f32) -> (i32, i32) {
    (
        (position.x / grid_cell_size).floor() as i32,
        (position.y / grid_cell_size).floor() as i32,
    )
}

pub fn get_neighboring_cells(cell: (i32, i32)) -> [(i32, i32); 9] {
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

pub fn wind_tunnel_velocity(y: f32, freestream_velocity: Vec2, box_size: Vec2) -> Vec2 {
    let max_speed = freestream_velocity.x;
    let normalized_y = (y + box_size.y / 2.0) / box_size.y;
    let velocity_x = max_speed * (1.0 - (2.0 * normalized_y - 1.0).powi(2));
    Vec2::new(velocity_x, 0.0)
}
