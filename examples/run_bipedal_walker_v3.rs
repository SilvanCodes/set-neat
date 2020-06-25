use set_neat::Genome;
use favannat::network::{activations, Fabricator, Evaluator};
use favannat::matrix::fabricator::MatrixFabricator;
use gym::{State, SpaceData};
use ndarray::{Axis, stack};
use std::fs;

pub const ENV: &str = "BipedalWalker-v3";

fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let winner_json = fs::read_to_string(
        format!("examples/winner_{}.json", ENV)
    ).expect("cant read file");
    let winner: Genome = serde_json::from_str(&winner_json).unwrap();


    let evaluator = MatrixFabricator::fabricate(&winner).unwrap();

    let mut recent_observation = env.reset().expect("Unable to reset");
    let mut done = false;

    while !done {
        env.render();
        let mut observations = recent_observation.get_box().unwrap();
        // normalize inputs
        observations.mapv_inplace(activations::TANH);
        // add bias input
        let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

        let State { observation, is_done, .. } = env.step(&SpaceData::BOX(output)).unwrap();
        recent_observation = observation;
        done = is_done;
    }
    env.close();
}