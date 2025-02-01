use bevy::prelude::*;

#[derive(Component)]
pub struct FluidParticle {
    pub velocity: Vec2,
    pub density: f32,
    pub pressure: f32,
}

impl Default for FluidParticle {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            density: 1.225,
            pressure: 0.0,
        }
    }
}

#[derive(Component)]
pub struct SolidBoundary {
    pub points: Vec<Vec2>,
    pub normals: Vec<Vec2>,
}

impl SolidBoundary {
    pub fn new_airfoil(chord: f32, position: Vec2, angle_of_attack: f32) -> Self {
        let num_points = 50;
        let mut points = Vec::with_capacity(num_points);
        let mut normals = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let x = (i as f32) / (num_points as f32 - 1.0);
            let t = 0.12; // maximum thickness
            let y = 5.0
                * t
                * chord
                * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x.powi(2) + 0.2843 * x.powi(3)
                    - 0.1015 * x.powi(4));

            let point = Vec2::new(x * chord - chord / 2.0, y);

            points.push(crate::math::rotate_point(point, angle_of_attack) + position);
            points.push(
                crate::math::rotate_point(Vec2::new(point.x, -point.y), angle_of_attack) + position,
            );

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
            normals.push(crate::math::rotate_vector(normal, angle_of_attack));
            normals.push(crate::math::rotate_vector(
                Vec2::new(-dy, -1.0).normalize(),
                angle_of_attack,
            ));
        }

        Self { points, normals }
    }

    pub fn new_wind_tunnel(box_size: Vec2) -> Self {
        let half_width = box_size.x / 2.0;
        let half_height = box_size.y / 2.0;

        let points = vec![
            Vec2::new(-half_width, half_height),
            Vec2::new(half_width, half_height),
            Vec2::new(-half_width, -half_height),
            Vec2::new(half_width, -half_height),
        ];

        let normals = vec![
            Vec2::new(0.0, -1.0),
            Vec2::new(0.0, -1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        Self { points, normals }
    }
}
