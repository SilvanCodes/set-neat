use favannat::matrix::fabricator::RecurrentMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{SpaceData, State};
use ndarray::{stack, Array1, Array2, Axis};
use set_neat::{scores::Raw, Behavior, Evaluation, Genome, Neat, Progress};

use log::{error, info};
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 1;
pub const VALIDATION_RUNS: usize = 100;
pub const STEPS: usize = 1600;
pub const ENV: &str = "BipedalWalker-v3";
pub const REQUIRED_FITNESS: f64 = 300.0;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.get(1).is_some() {
        let winner_json = fs::read_to_string(format!("examples/{}/winner_1601592694.json", ENV))
            .expect("cant read file");
        let winner: Genome = serde_json::from_str(&winner_json).unwrap();
        let standard_scaler: (Array1<f64>, Array1<f64>) =
            serde_json::from_str(&winner_json).unwrap();
        run(&winner, &standard_scaler, 1, STEPS, true);
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

    let mut std_dev = samples.std_axis(Axis(0), 0.0);

    std_dev.map_inplace(|v| {
        if *v == 0.0 {
            *v = f64::EPSILON
        }
    });

    dbg!(samples.mean_axis(Axis(0)).unwrap(), std_dev)
}

fn train(standard_scaler: (Array1<f64>, Array1<f64>)) {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    let standard_scaler_1 = standard_scaler.clone();

    let fitness_function = move |genome: &Genome| -> Progress {
        let (fitness, all_observations) = run(genome, &standard_scaler, RUNS, STEPS, false);

        if fitness > 0.0 {
            dbg!(fitness);
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");
            let (validation_fitness, all_observations) =
                run(genome, &standard_scaler, VALIDATION_RUNS, STEPS, false);
            // log possible solutions to file
            let mut genome = genome.clone();
            genome.fitness.raw = Raw::fitness(validation_fitness);
            info!(target: "app::solutions", "{}", serde_json::to_string(&genome).unwrap());
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
                let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);

                return Progress::Solution(
                    Raw::fitness(validation_fitness),
                    Behavior(
                        observation_means
                            .iter()
                            .take(14)
                            .cloned()
                            .chain(observation_std_dev.iter().take(14).cloned())
                            .collect(),
                    ),
                    genome,
                );
            }
        }

        let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
        let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);

        Progress::new(
            Raw::fitness(fitness),
            Behavior(
                observation_means
                    .iter()
                    .take(14)
                    .cloned()
                    .chain(observation_std_dev.iter().take(14).cloned())
                    .collect(),
            ),
        )
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
                /* if report.fitness_peak > report.archive_threshold {
                    showcase(report.top_performer);
                } */
                run(&report.top_performer, &standard_scaler_1, 1, STEPS, true);
                // showcase(standard_scaler_1.clone(), report.top_performer);
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
            serde_json::to_string(&standard_scaler_1).unwrap(),
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

fn run(
    net: &Genome,
    standard_scaler: &(Array1<f64>, Array1<f64>),
    runs: usize,
    steps: usize,
    render: bool,
) -> (f64, Array2<f64>) {
    let (means, std_dev) = standard_scaler;

    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let mut evaluator = RecurrentMatrixFabricator::fabricate(net).unwrap();
    let mut fitness = 0.0;
    let mut all_observations = Array2::zeros((1, 24));

    for run in 0..runs {
        evaluator.reset_internal_state();
        let mut recent_observation = env.reset().expect("Unable to reset");
        let mut total_reward = 0.0;

        for step in 0..steps {
            if render {
                env.render();
            }

            let mut observations = recent_observation.get_box().unwrap();

            all_observations = stack![
                Axis(0),
                all_observations,
                observations.clone().insert_axis(Axis(0))
            ];

            // normalize inputs
            observations -= means;
            observations /= std_dev;

            // add bias input
            let input = stack![Axis(0), observations, [1.0]];
            let output = evaluator.evaluate(input.clone());

            let (observation, reward, is_done) = match env.step(&SpaceData::BOX(output.clone())) {
                Ok(State {
                    observation,
                    reward,
                    is_done,
                }) => (observation, reward, is_done),
                Err(err) => {
                    error!("evaluation error: {}", err);
                    dbg!(means);
                    dbg!(std_dev);
                    dbg!(input);
                    dbg!(output);
                    dbg!(evaluator);
                    dbg!(net);
                    dbg!(all_observations);
                    panic!("evaluation error");
                    /* (
                        SpaceData::BOX(array![
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        ]),
                        0.0,
                        true,
                    ) */
                }
            };

            recent_observation = observation;
            total_reward += reward;

            if is_done {
                if render {
                    println!("finished with reward {} after {} steps", total_reward, step);
                }
                break;
            }
        }
        fitness += total_reward;
    }

    (fitness / runs as f64, all_observations)
}
