/* use favannat::matrix::fabricator::MatrixFabricator;
use favannat::network::{Evaluator, Fabricator};
use gym::{SpaceData, State};
use ndarray::{stack, Axis};
use set_neat::{Genome, Neat, Progress, Solution, activations};

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::{SmallRng, ThreadRng};
use rand::SeedableRng;

use std::fs;
use std::time::Instant;

fn main() {
    fn fitness_function(genome: &Genome) -> f64 {
        let gym = gym::GymClient::default();
        let env = gym.make("MountainCar-v0");
        let runs = 100;

        let evaluator = MatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        let actions = [
            &SpaceData::DISCRETE(0),
            &SpaceData::DISCRETE(1),
            &SpaceData::DISCRETE(2),
        ];
        // let mut rng = SmallRng::from_rng(&mut ThreadRng::default()).unwrap();

        for _ in 0..runs {
            let mut total_reward = 0.0;
            let mut done = false;
            let mut recent_observation = env.reset().expect("Unable to reset");

            while !done {
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                observations.mapv_inplace(activations::TANH);
                // add bias input
                let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                /* let softmaxsum: f64 = output.iter().map(|x| x.exp()).sum();
                let softmax: Vec<f64> = output.iter().map(|x| x.exp() / softmaxsum).collect();
                let dist = WeightedIndex::new(&softmax).unwrap();
                let State { observation, reward, is_done } = env.step(actions[dist.sample(&mut rng)]).unwrap(); */

                let max = output.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let pos = output.iter().position(|&value| value == max).unwrap();
                let State {
                    observation,
                    reward,
                    is_done,
                } = env.step(actions[pos]).unwrap();

                recent_observation = observation;
                total_reward += reward;
                done = is_done;
            }
            fitness += total_reward;
        }
        env.close();
        fitness / runs as f64
    };

    let neat = Neat::new("examples/mountaincar_v0.toml", fitness_function, -110.0);

    let now = Instant::now();

    println!("starting training...");

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Progress(report) => {
                println!("{:#?}", report);
                None
            }
            Solution(genome) => Some(genome),
        })
        .next()
    {
        fs::write(
            "examples/winner_mountain_car_v0.json",
            serde_json::to_string(&winner).unwrap(),
        )
        .expect("Unable to write file");

        let secs = now.elapsed().as_millis();
        println!(
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.node_genes.len(),
            winner.connection_genes.len(),
            secs as f64 / 1000.0,
            winner
        );
        let evaluator = MatrixFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);

        let gym = gym::GymClient::default();
        let env = gym.make("MountainCar-v0");
        // let steps = 200;

        let actions = [
            &SpaceData::DISCRETE(0),
            &SpaceData::DISCRETE(1),
            &SpaceData::DISCRETE(2),
        ];
        let mut rng = SmallRng::from_rng(&mut ThreadRng::default()).unwrap();

        let mut recent_observation = env.reset().expect("Unable to reset");
        let mut done = false;

        while !done {
            env.render();
            let mut observations = recent_observation.get_box().unwrap();
            // normalize inputs
            observations.mapv_inplace(activations::TANH);
            // add bias input
            let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

            let softmaxsum: f64 = output.iter().map(|x| x.exp()).sum();
            let softmax: Vec<f64> = output.iter().map(|x| x.exp() / softmaxsum).collect();
            let dist = WeightedIndex::new(&softmax).unwrap();

            let State {
                observation,
                is_done,
                ..
            } = env.step(actions[dist.sample(&mut rng)]).unwrap();
            recent_observation = observation;
            done = is_done;
        }
        env.close();
        println!("finished");
    }
}
 */
