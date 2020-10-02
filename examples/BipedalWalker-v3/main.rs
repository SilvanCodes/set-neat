use favannat::matrix::fabricator::StatefulMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{stack, Axis};
use set_neat::{activations, Evaluation, Genome, Neat, Progress};

use log::{error, info, warn};
use std::fs;
use std::time::Instant;
use std::time::SystemTime;

pub const RUNS: usize = 1;
pub const STEPS: usize = 1600;
pub const ENV: &str = "BipedalWalker-v3";

fn main() {
    log4rs::init_file(format!("examples/{}/config.yaml", ENV), Default::default()).unwrap();

    fn fitness_function(genome: &Genome) -> Progress {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);

        let mut evaluator = StatefulMatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        for _ in 0..RUNS {
            let mut recent_observation = env.reset().expect("Unable to reset");
            let mut total_reward = 0.0;

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
                } = env.step(&SpaceData::BOX(output)).unwrap();
                recent_observation = observation;
                total_reward += reward;

                if is_done {
                    // println!("finished with reward {} after {} steps", reward, step);
                    break;
                }
            }
            fitness += total_reward;
        }
        env.close();

        // normailze with run count
        fitness /= RUNS as f64;

        if fitness >= 300.0 {
            Progress::Solution(genome.clone())
        } else {
            Progress::Fitness(fitness)
        }
    };

    let neat = Neat::new(&format!("examples/{}/config.toml", ENV), fitness_function);

    let now = Instant::now();

    let time_stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    info!("starting training: {:#?}", neat.parameters);

    /* let winner_json = fs::read_to_string(
        format!("examples/winner_{}.json", ENV)
    ).expect("cant read file");
    let winner: Genome = serde_json::from_str(&winner_json).unwrap(); */

    fs::write(
        format!("examples/{}/{}_parameters.json", ENV, time_stamp),
        serde_json::to_string(&neat.parameters).unwrap(),
    )
    .expect("Unable to write file");

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                // info!(serde_json::to_string(&report).unwrap());

                info!("{}", serde_json::to_string(&report).unwrap());
                None
            }
            Evaluation::Solution(genome) => Some(genome),
        })
        .next()
    {
        fs::write(
            format!("examples/{}/{}_winner.json", ENV, time_stamp),
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
        let mut done = false;

        while !done {
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
            } = env.step(&SpaceData::BOX(output)).unwrap();
            recent_observation = observation;
            done = is_done;
        }
        env.close();
    }
}
