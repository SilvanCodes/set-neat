use favannat::matrix::fabricator::RecurrentMatrixFabricator;
use favannat::{
    looping::fabricator::LoopingFabricator,
    network::{StatefulEvaluator, StatefulFabricator},
};
use gym::{utility::StandardScaler, SpaceData, State};
use ndarray::{stack, Array2, Axis};
use set_neat::{Evaluation, Individual, Neat, Progress};

use log::{error, info};
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 1;
pub const PRE_VALIDATION_RUNS: usize = 10;
pub const VALIDATION_RUNS: usize = 100;
pub const STEPS: usize = usize::MAX;
pub const ENV: &str = "LunarLanderContinuous-v2";
pub const REQUIRED_FITNESS: f64 = 200.0;
pub const GENERATIONS: usize = 1000;

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(timestamp) = args.get(1) {
        let winner_json = fs::read_to_string(format!("examples/{}/{}_winner.json", ENV, timestamp))
            .expect("cant read file");
        let winner: Individual = serde_json::from_str(&winner_json).unwrap();
        let scaler_json = fs::read_to_string(format!(
            "examples/{}/{}_winner_standard_scaler.json",
            ENV, timestamp
        ))
        .expect("cant read file");
        let standard_scaler: StandardScaler = serde_json::from_str(&scaler_json).unwrap();
        // showcase(standard_scaler, winner);
        run(&standard_scaler, &winner, 1, STEPS, true, false);
    } else {
        train(StandardScaler::for_environment(ENV));
    }
}

fn train(standard_scaler: StandardScaler) {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    info!(target: "app::parameters", "standard scaler: {:?}", &standard_scaler);

    let other_standard_scaler = standard_scaler.clone();

    let fitness_function = move |individual: &Individual| -> Progress {
        let standard_scaler = &standard_scaler;

        let (fitness, all_observations) =
            run(standard_scaler, individual, RUNS, STEPS, false, false);

        if fitness > 0.0 {
            dbg!(fitness);
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");

            let (validation_fitness, all_observations) = run(
                &standard_scaler,
                individual,
                VALIDATION_RUNS,
                STEPS,
                false,
                false,
            );

            // log possible solutions to file
            let mut individual = individual.clone();
            individual.fitness.raw = validation_fitness;
            info!(target: "app::solutions", "{}", serde_json::to_string(&individual).unwrap());
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                // let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
                // let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);
                return Progress::fitness(
                    validation_fitness,
                    /* all_observations
                    .row(all_observations.shape()[0] - 1)
                    .to_vec(), */
                )
                .solved(individual);
            }
        }
        /* if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting pre-validation runs...");

            let (pre_validation_fitness, _) = run(
                standard_scaler,
                individual,
                PRE_VALIDATION_RUNS,
                STEPS,
                false,
                false,
            );

            info!(
                "finished pre-validation runs with {} average fitness",
                pre_validation_fitness
            );

            if pre_validation_fitness >= REQUIRED_FITNESS {
                info!("hit task theshold, starting validation runs...");

                let (validation_fitness, all_observations) = run(
                    &standard_scaler,
                    individual,
                    VALIDATION_RUNS,
                    STEPS,
                    false,
                    false,
                );

                // log possible solutions to file
                let individual = individual.clone();
                info!(target: "app::solutions", "{}", serde_json::to_string(&individual).unwrap());
                info!(
                    "finished validation runs with {} average fitness",
                    validation_fitness
                );
                if validation_fitness > REQUIRED_FITNESS {
                    // let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
                    // let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);
                    return Progress::fitness(
                        validation_fitness,
                        /* all_observations
                        .row(all_observations.shape()[0] - 1)
                        .to_vec()[0..2]
                        .to_vec(), */
                    )
                    .solved(individual);
                }
            }
        } */
        // let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
        // let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);

        Progress::fitness(
            fitness,
            /* all_observations
            .row(all_observations.shape()[0] - 1)
            .to_vec(), */
        )
    };

    let neat = Neat::new(
        &format!("examples/{}/config.toml", ENV),
        Box::new(fitness_function),
    );

    let now = Instant::now();

    info!(target: "app::parameters", "starting training...\nRUNS:{:#?}\nVALIDATION_RUNS:{:#?}\nSTEPS: {:#?}\nREQUIRED_FITNESS:{:#?}\nPARAMETERS: {:#?}", RUNS, VALIDATION_RUNS, STEPS, REQUIRED_FITNESS, neat.parameters);

    if let Some(winner) = neat
        .run()
        .take(GENERATIONS)
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                info!(target: "app::progress", "{}", serde_json::to_string(&report).unwrap());
                /* if report.num_generation % 5 == 0 {
                    run(
                        &other_standard_scaler,
                        &report.top_performer,
                        1,
                        STEPS,
                        true,
                        true,
                    );
                } */
                None
            }
            Evaluation::Solution(individual) => Some(individual),
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
            serde_json::to_string(&other_standard_scaler).unwrap(),
        )
        .expect("Unable to write file");

        let secs = now.elapsed().as_millis();
        info!(
            "winning individual ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            secs as f64 / 1000.0,
            winner
        );
    } else {
        println!("##### OUT OF TIME #####");
    }
}

fn run(
    standard_scaler: &StandardScaler,
    net: &Individual,
    runs: usize,
    steps: usize,
    render: bool,
    debug: bool,
) -> (f64, Array2<f64>) {
    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let mut evaluator = RecurrentMatrixFabricator::fabricate(net).unwrap();
    // let mut evaluator = LoopingFabricator::fabricate(net).unwrap();
    let mut fitness = 0.0;
    let mut all_observations = Array2::zeros((1, 8));

    if debug {
        dbg!(net);
        dbg!(&evaluator);
    }

    for run in 0..runs {
        evaluator.reset_internal_state();
        let mut recent_observation = env.reset().expect("Unable to reset");
        let mut total_reward = 0.0;

        if debug {
            dbg!(run);
        }

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

            standard_scaler.scale_inplace(observations.view_mut());

            // add bias input
            let input = stack![Axis(0), observations, [1.0]];
            let output = evaluator.evaluate(input.clone());

            if debug {
                dbg!(&input);
                dbg!(&output);
            }

            let (observation, reward, is_done) = match env.step(&SpaceData::BOX(output.clone())) {
                Ok(State {
                    observation,
                    reward,
                    is_done,
                }) => (observation, reward, is_done),
                Err(err) => {
                    error!("evaluation error: {}", err);
                    dbg!(run);
                    dbg!(input);
                    dbg!(output);
                    dbg!(evaluator);
                    dbg!(net);
                    dbg!(all_observations);
                    panic!("evaluation error");
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

    if debug {
        dbg!(&all_observations);
    }

    (fitness / runs as f64, all_observations)
}
