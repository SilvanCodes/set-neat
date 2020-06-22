use set_neat::Neat;
use set_neat::runtime::Evaluation::{Progress, Solution};
use set_neat::genome::Genome;
use favannat::network::{activations, Fabricator, Evaluator};
use favannat::matrix::fabricator::MatrixFabricator;
use gym::{State, SpaceData};
use ndarray::{Axis, stack};

use std::time::Instant;
use std::fs;


fn main() {
    fn fitness_function(genome: &Genome) -> f64 {
        let gym = gym::GymClient::default();
        let env = gym.make("CartPole-v1");
        let runs = 100;
        let expected_steps = 500;

        let evaluator = MatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        for _ in 0..runs {
            let mut recent_observation = env.reset().expect("Unable to reset");
            let mut total_reward = 0.0;
            let mut done = false;

            for _ in 0..expected_steps {
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                observations.mapv_inplace(activations::TANH);
                // add bias input
                let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                if output[0] > 0.0 {
                    let State { observation, is_done, reward } = env.step(&SpaceData::DISCRETE(0)).unwrap();
                    recent_observation = observation;
                    total_reward += reward;
                    done = is_done;
                } else {
                    let State { observation, is_done, reward } = env.step(&SpaceData::DISCRETE(1)).unwrap();
                    recent_observation = observation;
                    total_reward += reward;
                    done = is_done;
                }
                if done {
                    // println!("finished with reward {} after {} steps", reward, step);
                    break;
                }
                
            }
            fitness += total_reward;
        }
        env.close();
        fitness / runs as f64
    };

    let neat = Neat::new("examples/cartpole_v1.toml", fitness_function, 495.0);

    let now = Instant::now();

    if let Some(winner) = neat.run().filter_map(|evaluation| {
        match evaluation {
            Progress(report) => {println!("{:#?}", report); None},
            Solution(genome) => Some(genome)
        }
    }).next() {
        fs::write(
            "examples/winner_cartpole_v1.json",
            serde_json::to_string(&winner).unwrap()
        ).expect("Unable to write file");

        let secs = now.elapsed().as_millis();
        println!("winning genome ({},{}) after {} seconds: {:?}", winner.node_genes.len(), winner.connection_genes.len(), secs as f64 / 1000.0, winner);
        let evaluator = MatrixFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);

        let gym = gym::GymClient::default();
        let env = gym.make("CartPole-v1");

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
}