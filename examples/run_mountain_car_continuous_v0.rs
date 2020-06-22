use set_neat::genome::Genome;
use favannat::network::{Fabricator, Evaluator, activations};
use favannat::matrix::fabricator::MatrixFabricator;
use gym::{State, SpaceData};
use ndarray::{Axis, stack};

use std::fs;

fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make("MountainCarContinuous-v0");

    let winner_json = fs::read_to_string("examples/winner_mountain_car_continuous_v0.json").expect("cant read file");
    let winner: Genome = serde_json::from_str(&winner_json).unwrap();

    // let net = genome_to_net(&winner);
    let evaluator = MatrixFabricator::fabricate(&winner).unwrap();

    let mut done = false;
    let mut recent_observation = env.reset().expect("Unable to reset");

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