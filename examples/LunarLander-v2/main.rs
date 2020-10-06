use favannat::matrix::fabricator::StatefulMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{array, stack, Axis};
use rand::{distributions::WeightedIndex, prelude::SmallRng, SeedableRng};
use rand_distr::Distribution;
use set_neat::{activations, Evaluation, Genome, Neat, Progress};

use log::info;
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 1;
pub const VALIDATION_RUNS: usize = 100;
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

    let means;
    let std_dev;
    {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);

        // collect samples for standard scaler
        let samples = env.reset().unwrap().get_box().unwrap();

        let mut samples = samples.insert_axis(Axis(0));

        println!("sampling for scaler");
        for i in 0..1000 {
            println!("sampling {}", i);
            let State { observation, .. } = env.step(&env.action_space().sample()).unwrap();

            samples = stack![
                Axis(0),
                samples,
                observation.get_box().unwrap().insert_axis(Axis(0))
            ];
        }
        env.close();
        println!("done sampling");

        dbg!(&samples);

        means = samples.mean_axis(Axis(0)).unwrap();
        std_dev = samples.std_axis(Axis(0), 0.0);
    }

    dbg!(&means);

    dbg!(&std_dev);

    let fitness_function = move |genome: &Genome| -> Progress {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);
        let mut rng = SmallRng::seed_from_u64(42);

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
            // println!("setting up run..");
            let mut recent_observation = env.reset().expect("Unable to reset");
            let mut total_reward = 0.0;

            loop {
                // println!("looping");
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                // observations.mapv_inplace(activations::TANH);
                observations -= &means;
                observations /= &std_dev;
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
            // log possible solutions to file
            info!(target: "app::solution", "{}", serde_json::to_string(genome).unwrap());
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
            info!("finished validation runs with {} average fitness", fitness);
            if fitness >= 200.0 {
                return Progress::Solution(genome.clone());
            }
        }
        // Progress::Fitness(fitness)
        let state = final_observation.get_box().unwrap().to_vec();
        // state.truncate(4);
        Progress::Novelty(state)
    };

    let neat = Neat::new(
        &format!("examples/{}/config.toml", ENV),
        Box::new(fitness_function),
    );

    let now = Instant::now();

    info!(target: "app::parameters", "starting training: {:#?}", neat.parameters);

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                info!(target: "app::progress", "{}", serde_json::to_string(&report).unwrap());
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
    let mut rng = SmallRng::seed_from_u64(42);

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
    env.close();
    println!("finished with reward: {}", total_reward);
}
