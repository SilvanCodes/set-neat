use favannat::matrix::fabricator::StatefulMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{array, stack, Array1, Axis};
use set_neat::{scores::Raw, Behavior, Evaluation, Genome, Neat, Progress};

use log::info;
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 3;
pub const VALIDATION_RUNS: usize = 100;
pub const ENV: &str = "LunarLanderContinuous-v2";
pub const REQUIRED_FITNESS: f64 = 200.0;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.get(1).is_some() {
        let winner_json = fs::read_to_string(format!("examples/{}/1602346227_winner.json", ENV))
            .expect("cant read file");
        let winner: Genome = serde_json::from_str(&winner_json).unwrap();
        let scaler_json = fs::read_to_string(format!(
            "examples/{}/1602346227_winner_standard_scaler.json",
            ENV
        ))
        .expect("cant read file");
        let standard_scaler: (Array1<f64>, Array1<f64>) =
            serde_json::from_str(&scaler_json).unwrap();
        showcase(standard_scaler, winner);
    } else {
        train(standard_scaler());
    }
}

fn standard_scaler() -> (Array1<f64>, Array1<f64>) {
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
    println!("done sampling");

    (
        samples.mean_axis(Axis(0)).unwrap(),
        samples.std_axis(Axis(0), 0.0),
    )
}

fn train(standard_scaler: (Array1<f64>, Array1<f64>)) {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    let (means, std_dev) = standard_scaler.clone();

    // dbg!(&means);

    // dbg!(&std_dev);

    let fitness_function = move |genome: &Genome| -> Progress {
        let gym = gym::GymClient::default();
        let env = gym.make(ENV);

        let mut evaluator = StatefulMatrixFabricator::fabricate(genome).unwrap();
        let mut fitness = 0.0;

        let mut final_observation = SpaceData::BOX(array![]);

        for _ in 0..RUNS {
            // println!("setting up run..");
            let mut recent_observation = env.reset().expect("Unable to reset");
            let mut total_reward = 0.0;

            loop {
                // println!("looping");
                let mut observations = recent_observation.get_box().unwrap();
                // normalize inputs
                observations -= &means;
                observations /= &std_dev;
                // add bias input
                let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                let State {
                    observation,
                    reward,
                    is_done,
                } = env.step(&SpaceData::BOX(output)).unwrap();

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

        fitness /= RUNS as f64;

        if fitness > 0.0 {
            dbg!(fitness);
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");
            let mut validation_fitness = 0.0;
            for _ in 0..VALIDATION_RUNS {
                let mut recent_observation = env.reset().expect("Unable to reset");
                let mut total_reward = 0.0;

                loop {
                    let mut observations = recent_observation.get_box().unwrap();
                    // normalize inputs
                    observations -= &means;
                    observations /= &std_dev;
                    // add bias input
                    let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

                    let State {
                        observation,
                        reward,
                        is_done,
                    } = env.step(&SpaceData::BOX(output)).unwrap();

                    recent_observation = observation;
                    total_reward += reward;

                    if is_done {
                        // println!("finished with reward {} after {} steps", reward, step);
                        final_observation = recent_observation;
                        break;
                    }
                }
                dbg!(total_reward);
                validation_fitness += total_reward;
            }

            validation_fitness /= VALIDATION_RUNS as f64;
            // log possible solutions to file
            let mut genome = genome.clone();
            genome.fitness.raw = Raw::fitness(validation_fitness);
            info!(target: "app::solutions", "{}", serde_json::to_string(&genome).unwrap());
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                let state = final_observation.get_box().unwrap().to_vec();
                // state.truncate(6);
                return Progress::Solution(
                    Raw::fitness(validation_fitness),
                    Behavior(state),
                    genome,
                );
            }
        }

        // env.close();

        // Progress::Fitness(fitness)
        let state = final_observation.get_box().unwrap().to_vec();
        // state.truncate(6);
        Progress::new(Raw::fitness(fitness), Behavior(state))
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
                if report.novelty.raw_maximum > report.archive_threshold {
                    showcase(standard_scaler.clone(), report.top_performer);
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
        fs::write(
            format!(
                "examples/{}/{}_winner_standard_scaler.json",
                ENV, time_stamp
            ),
            serde_json::to_string(&standard_scaler).unwrap(),
        )
        .expect("Unable to write file");

        let secs = now.elapsed().as_millis();
        info!(
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            secs as f64 / 1000.0,
            winner
        );
    }
}

fn showcase(standard_scaler: (Array1<f64>, Array1<f64>), genome: Genome) {
    let (means, std_dev) = standard_scaler;

    // dbg!(&means);
    // dbg!(&std_dev);

    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let mut evaluator = StatefulMatrixFabricator::fabricate(&genome).unwrap();

    let mut recent_observation = env.reset().expect("Unable to reset");
    let mut total_reward = 0.0;
    let mut done = false;

    while !done {
        env.render();
        let mut observations = recent_observation.get_box().unwrap();
        // normalize inputs
        observations -= &means;
        observations /= &std_dev;
        // add bias input
        let output = evaluator.evaluate(stack![Axis(0), observations, [1.0]]);

        let State {
            observation,
            reward,
            is_done,
        } = env.step(&SpaceData::BOX(output)).unwrap();

        recent_observation = observation;
        total_reward += reward;
        done = is_done;
    }
    println!("finished with reward: {}", total_reward);
}
