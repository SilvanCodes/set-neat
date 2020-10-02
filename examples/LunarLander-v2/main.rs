use favannat::matrix::fabricator::StatefulMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{array, stack, Axis};
use rand::{distributions::WeightedIndex, prelude::SmallRng, prelude::ThreadRng, SeedableRng};
use rand_distr::Distribution;
use set_neat::{activations, Evaluation, Genome, Neat, Progress};

use log::info;
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 1;
pub const VALIDATION_RUNS: usize = 3;
pub const ENV: &str = "LunarLander-v2";

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.get(1).is_some() {
        let winner_json = fs::read_to_string(format!("examples/{}/winner_1601592694.json", ENV))
            .expect("cant read file");
        let winner: Genome = serde_json::from_str(&winner_json).unwrap();
        showcase(winner);
    } else {
        train();
    }
}

fn train() {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    fn fitness_function(genome: &Genome) -> Progress {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);
        let mut rng = SmallRng::from_rng(&mut ThreadRng::default()).unwrap();

        let mut evaluator = StatefulMatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;
        let mut done = false;

        let actions = [
            &SpaceData::DISCRETE(0),
            &SpaceData::DISCRETE(1),
            &SpaceData::DISCRETE(2),
            &SpaceData::DISCRETE(3),
        ];

        let mut final_observation = SpaceData::BOX(array![]);

        for _ in 0..RUNS {
            let mut recent_observation = env.reset().expect("Unable to reset");
            let mut total_reward = 0.0;

            loop {
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
                    reward,
                    is_done,
                } = env.step(actions[dist.sample(&mut rng)]).unwrap();

                recent_observation = observation;
                total_reward += reward;

                if is_done {
                    // println!("finished with reward {} after {} steps", reward, step);
                    final_observation = recent_observation;
                    break;
                }
            }
            fitness += total_reward;
        }
        env.close();

        fitness /= RUNS as f64;

        if fitness > 0.0 {
            dbg!(fitness);
        }

        if fitness >= 200.0 {
            info!("hit task theshold, starting validation runs...");
            fitness = 0.0;
            for _ in 0..VALIDATION_RUNS {
                let mut recent_observation = env.reset().expect("Unable to reset");
                let mut total_reward = 0.0;

                while !done {
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
                        reward,
                        is_done,
                    } = env.step(actions[dist.sample(&mut rng)]).unwrap();

                    recent_observation = observation;
                    total_reward += reward;
                    done = is_done;
                }
                fitness += total_reward;
            }
            if fitness >= 200.0 {
                return Progress::Solution(genome.clone());
            }
        }
        // Progress::Fitness(fitness)
        let state = final_observation.get_box().unwrap().to_vec();
        // state.truncate(4);
        Progress::Novelty(state)
    };

    let neat = Neat::new(&format!("examples/{}/config.toml", ENV), fitness_function);

    let now = Instant::now();

    info!(target: "app::parameters", "starting training: {:#?}", neat.parameters);

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                info!(target: "app::progress", "{}", serde_json::to_string(&report).unwrap());
                // TODO: save report
                // TODO: render top_performer
                if report.fitness_peak > report.archive_threshold {
                    showcase(report.top_performer);
                }
                None
            }
            Evaluation::Solution(genome) => Some(genome),
        })
        .next()
    {
        let time_stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        fs::write(
            format!("examples/{}/{}_winner.json", ENV, time_stamp),
            serde_json::to_string(&winner).unwrap(),
        )
        .expect("Unable to write file");
        fs::write(
            format!("examples/{}/{}_winner_parameters.json", ENV, time_stamp),
            serde_json::to_string(&neat.parameters).unwrap(),
        )
        .expect("Unable to write file");

        let secs = now.elapsed().as_millis();
        info!(
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.node_genes.len(),
            winner.connection_genes.len(),
            secs as f64 / 1000.0,
            winner
        );
    }
}

fn showcase(genome: Genome) {
    let gym = gym::GymClient::default();
    let env = gym.make(ENV);
    let mut rng = SmallRng::from_rng(&mut ThreadRng::default()).unwrap();

    let mut evaluator = StatefulMatrixFabricator::fabricate(&genome).unwrap();

    let actions = [
        &SpaceData::DISCRETE(0),
        &SpaceData::DISCRETE(1),
        &SpaceData::DISCRETE(2),
        &SpaceData::DISCRETE(3),
    ];

    let mut recent_observation = env.reset().expect("Unable to reset");
    let mut total_reward = 0.0;
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
            reward,
            is_done,
        } = env.step(actions[dist.sample(&mut rng)]).unwrap();

        recent_observation = observation;
        total_reward += reward;
        done = is_done;
    }
    println!("finished with reward: {}", total_reward);
}
