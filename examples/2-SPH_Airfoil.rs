use bevy::prelude::*;
use sph_plugin::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (200.0 * 4.0, 200.0 * 4.0).into(),
                title: "SPH Simulation".to_string(),
                resizable: false,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(SPHPlugin {
            config: SPHConfig::default(),
        })
        .run();
}
