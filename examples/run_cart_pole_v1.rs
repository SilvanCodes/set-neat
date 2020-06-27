use favannat::matrix::fabricator::MatrixFabricator;
use favannat::network::{activations, Evaluator, Fabricator};
use gym::{SpaceData, State};
use ndarray::{stack, Axis};
use set_neat::Genome;

use std::fs;

pub const ENV: &str = "CartPole-v1";

fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let winner_json =
        fs::read_to_string(format!("examples/winner_{}.json", ENV)).expect("cant read file");
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

        if output[0] > 0.0 {
            let State {
                observation,
                is_done,
                ..
            } = env.step(&SpaceData::DISCRETE(0)).unwrap();
            recent_observation = observation;
            done = is_done;
        } else {
            let State {
                observation,
                is_done,
                ..
            } = env.step(&SpaceData::DISCRETE(1)).unwrap();
            recent_observation = observation;
            done = is_done;
        }
    }
    env.close();
}
