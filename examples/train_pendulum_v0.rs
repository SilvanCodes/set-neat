/* use favannat::matrix::fabricator::StatefulMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{stack, Axis};
use set_neat::{Genome, Neat, Progress, Solution, activations};

use std::fs;
use std::time::Instant;

pub const RUNS: usize = 100;
pub const STEPS: usize = 100;
pub const ENV: &str = "Pendulum-v0";

fn main() {
    fn fitness_function(genome: &Genome) -> f64 {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);

        let mut evaluator = StatefulMatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        for _ in 0..RUNS {
            let mut total_reward = 0.0;
            let mut recent_observation = env.reset().expect("Unable to reset");

            for _ in 0..STEPS {
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                observations.mapv_inplace(activations::TANH);
                // add bias input
                let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                let State {
                    observation,
                    is_done,
                    reward,
                } = env.step(&SpaceData::BOX(output * 2.0)).unwrap();

                recent_observation = observation;
                total_reward += reward;

                if is_done {
                    // println!("finished with reward {} after {} steps", total_reward, steps);
                    break;
                }
            }
            fitness += total_reward;
        }
        env.close();
        fitness / RUNS as f64
    };

    let neat = Neat::new(&format!("examples/{}.toml", ENV), fitness_function, -400.0);

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
            format! {"examples/winner_{}.json", ENV},
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
        let mut evaluator = StatefulMatrixFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);

        let gym = gym::GymClient::default();
        let env = gym.make(ENV);

        let mut recent_observation = env.reset().expect("Unable to reset");

        for _ in 0..STEPS {
            env.render();
            let mut observations = recent_observation.get_box().unwrap();
            // normalize inputs
            observations.mapv_inplace(activations::TANH);
            // add bias input
            let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

            let State {
                observation,
                is_done,
                ..
            } = env.step(&SpaceData::BOX(output * 2.0)).unwrap();

            recent_observation = observation;
            if is_done {
                break;
            }
        }
        env.close();
        println!("finished");
    }
}
 */
