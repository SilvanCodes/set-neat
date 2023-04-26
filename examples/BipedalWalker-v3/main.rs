use favannat::{SparseMatrixRecurrentFabricator, StatefulEvaluator, StatefulFabricator};
use gym::client::MakeOptions;
use gym::space_data::SpaceData;
use gym::{space_template::SpaceTemplate, utility::StandardScaler, Action, State};
use ndarray::{concatenate, stack, Array2, Axis};
use set_neat::{Individual, Neat, Progress};

use log::{error, info};
use std::{env, fs, sync::RwLock};
use std::{
    ops::Deref,
    time::{Instant, SystemTime},
};

pub const RUNS: usize = 3;
pub const VALIDATION_RUNS: usize = 100;
pub const STEPS: usize = 1600;
pub const ENV: &str = "BipedalWalker-v3";
pub const REQUIRED_FITNESS: f64 = 300.0;

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(timestamp) = args.get(1) {
        let winner_json =
            fs::read_to_string(format!("examples/{}/{}_candidate.json", ENV, timestamp))
                .expect("cant read file");
        let winner: Individual = serde_json::from_str(&winner_json).unwrap();
        // let standard_scaler: StandardScaler = serde_json::from_str(&winner_json).unwrap();
        run(&winner, 1, STEPS, true);
    } else {
        train(StandardScaler::for_environment(ENV));
    }
}

fn train(standard_scaler: StandardScaler) {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    let max_score: RwLock<f64> = RwLock::new(0.0);

    let standard_scaler_1 = standard_scaler.clone();

    let fitness_function = move |individual: &Individual| -> Progress {
        let (fitness, all_observations) = run(individual, RUNS, STEPS, false);

        if fitness > 0.0 {
            dbg!(fitness);
            let mut save = false;

            if let Ok(max_score) = max_score.read() {
                if fitness > *max_score {
                    save = true;
                }
            }

            if save {
                if let Ok(mut max_score) = max_score.write() {
                    *max_score = fitness;
                    let time_stamp = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let mut individual = individual.clone();
                    individual.fitness.raw = fitness;
                    fs::write(
                        format!("examples/{}/{}_candidate.json", ENV, time_stamp),
                        serde_json::to_string(&individual).unwrap(),
                    )
                    .expect("Unable to write file");
                }
            }
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");
            let (validation_fitness, all_observations) =
                run(individual, VALIDATION_RUNS, STEPS, false);
            // log possible solutions to file
            let mut individual = individual.clone();
            individual.fitness.raw = validation_fitness;
            info!(target: "app::solutions", "{}", serde_json::to_string(&individual).unwrap());
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
                let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);

                return Progress::fitness(
                    validation_fitness,
                    /* observation_means
                    .iter()
                    .take(14)
                    .cloned()
                    .chain(observation_std_dev.iter().take(14).cloned())
                    .collect(), */
                )
                .solved(individual);
            }
        }

        let observation_means = all_observations.mean_axis(Axis(0)).unwrap();
        let observation_std_dev = all_observations.std_axis(Axis(0), 0.0);

        Progress::fitness(
            fitness,
            /* observation_means
            .iter()
            .take(14)
            .cloned()
            .chain(observation_std_dev.iter().take(14).cloned())
            .collect(), */
        )
    };

    let neat = Neat::new(
        &format!("examples/{}/config.toml", ENV),
        Box::new(fitness_function),
    );

    let now = Instant::now();

    info!(target: "app::parameters", "starting training: {:#?}", neat.parameters);

    if let Some(winner) = neat.run().find_map(|(statistics, solution)| {
        info!(target: "app::progress", "{}", serde_json::to_string(&statistics).unwrap());
        if statistics.population.num_generation % 5 == 0 {
            run(
                &statistics.population.top_performer,
                // &standard_scaler_1,
                1,
                STEPS,
                true,
            );
        }
        solution
    }) {
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

        info!(
            "winning individual ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            now.elapsed().as_secs(),
            winner
        );
    }
}

fn run(
    net: &Individual,
    // standard_scaler: &StandardScaler,
    runs: usize,
    steps: usize,
    render: bool,
) -> (f64, Array2<f64>) {
    let gym = gym::client::GymClient::default();
    let env = if render {
        gym.make(
            ENV,
            Some(MakeOptions {
                render_mode: Some(gym::client::RenderMode::Human),
                ..Default::default()
            }),
        )
        .unwrap()
    } else {
        gym.make(ENV, None).unwrap()
    };

    let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(net.deref()).unwrap();
    let mut fitness = 0.0;

    let mut all_observations;

    if let SpaceTemplate::Box { shape, .. } = env.observation_space() {
        all_observations = Array2::zeros((1, shape[0]));
    } else {
        panic!("is no box observation space")
    }

    for run in 0..runs {
        evaluator.reset_internal_state();
        let (mut recent_observation, _info) = env.reset(None).expect("Unable to reset");
        let mut total_reward = 0.0;

        for step in 0..steps {
            if render {
                env.render();
            }

            let mut observations = recent_observation.get_box().unwrap();

            all_observations = concatenate![
                Axis(0),
                all_observations,
                observations.clone().insert_axis(Axis(0))
            ];

            // normalize inputs
            // standard_scaler.scale_inplace(observations.view_mut());

            // add bias input
            let input = concatenate![Axis(0), observations, [1.0]];
            let output = evaluator.evaluate(input.clone());

            let (observation, reward, is_done) = match env.step(&SpaceData::Box(output.clone())) {
                Ok(State {
                    observation,
                    reward,
                    is_done,
                    ..
                }) => (observation, reward, is_done),
                Err(err) => {
                    error!("evaluation error: {}", err);
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
