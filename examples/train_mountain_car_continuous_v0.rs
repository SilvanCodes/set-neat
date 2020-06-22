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
        let env = gym.make("MountainCarContinuous-v0");
        let runs = 100;
        let steps = 200;

        let evaluator = MatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        for _ in 0..runs {
            let mut total_reward = 0.0;
            let mut recent_observation = env.reset().expect("Unable to reset");

            // while !done {
            for _ in 0..steps {
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                observations.mapv_inplace(activations::TANH);
                // add bias input
                let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                let State { observation, is_done, reward } = env.step(&SpaceData::BOX(output)).unwrap();

                recent_observation = observation;
                if is_done {
                    total_reward = reward;
                    // println!("finished with reward {} after {} steps", total_reward, steps);
                    break;
                }
            }
            fitness += total_reward;
        }
        env.close();
        fitness / runs as f64
    };

    let neat = Neat::new("examples/mountain_car_continuous_v0.toml", fitness_function, 90.0);

    let now = Instant::now();

    println!("starting training...");

    if let Some(winner) = neat.run().filter_map(|evaluation| {
        match evaluation {
            Progress(report) => {println!("{:#?}", report); None},
            Solution(genome) => Some(genome)
        }
    }).next() {
        fs::write(
            "examples/winner_mountain_car_continuous_v0.json",
            serde_json::to_string(&winner).unwrap()
        ).expect("Unable to write file");
        
        let secs = now.elapsed().as_millis();
        println!("winning genome ({},{}) after {} seconds: {:?}", winner.node_genes.len(), winner.connection_genes.len(), secs as f64 / 1000.0, winner);
        let evaluator = MatrixFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);

        let gym = gym::GymClient::default();
        let env = gym.make("MountainCarContinuous-v0");

        let steps = 200;
        let mut recent_observation = env.reset().expect("Unable to reset");

        for step in 0..steps {
            env.render();
            let mut observations = recent_observation.get_box().unwrap();
            // normalize inputs
            observations.mapv_inplace(activations::TANH);
            // add bias input
            let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

            let State { observation, is_done, .. } = env.step(&SpaceData::BOX(output)).unwrap();

            recent_observation = observation;
            if is_done {
                println!("finished with after {} steps", step);
                break;
            }
        }
        env.close();
        println!("finished");
    }
}