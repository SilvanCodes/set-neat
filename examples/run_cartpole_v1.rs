use set_neat::genome::Genome;
use favannat::network::{Fabricator, Evaluator, activations};
use favannat::matrix::fabricator::MatrixFabricator;
use gym::{State, SpaceData};
use ndarray::{Axis, stack};

use std::fs;


fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make("CartPole-v1");

    let winner_json = fs::read_to_string("examples/winner_cartpole_v1.json").expect("cant read file");
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
            let State { observation, is_done, .. } = env.step(&SpaceData::DISCRETE(0)).unwrap();
            recent_observation = observation;
            done = is_done;
        } else {
            let State { observation, is_done, .. } = env.step(&SpaceData::DISCRETE(1)).unwrap();
            recent_observation = observation;
            done = is_done;
        }
    }
    env.close();
}