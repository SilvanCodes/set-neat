use favannat::{
    matrix::recurrent::fabricator::MatrixRecurrentFabricator,
    network::{StatefulEvaluator, StatefulFabricator},
};
use gym::{utility::StandardScaler, Action, State};
use ndarray::{concatenate, Array2, Axis};
use rand::{distributions::WeightedIndex, prelude::SmallRng, SeedableRng};
use rand_distr::Distribution;
use set_neat::{Individual, Neat, Progress};

use log::{error, info};
use std::time::SystemTime;
use std::{env, fs};
use std::{ops::Deref, time::Instant};

pub const RUNS: usize = 3;
pub const STEPS: usize = 3000;
pub const VALIDATION_RUNS: usize = 100;
pub const ENV: &str = "LunarLander-v2";
pub const REQUIRED_FITNESS: f64 = 200.0;

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

    if let Some(winner) = neat.run().take(100).find_map(|(statistics, solution)| {
        info!(target: "app::progress", "{}", serde_json::to_string(&statistics).unwrap());
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
    let gym = gym::client::GymClient::default();
    let env = gym.make(ENV, None).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);

    let actions = [
        &Action::Discrete(0),
        &Action::Discrete(1),
        &Action::Discrete(2),
        &Action::Discrete(3),
    ];

    let mut evaluator = MatrixRecurrentFabricator::fabricate(net.deref()).unwrap();
    let mut fitness = 0.0;
    let mut all_observations = Array2::zeros((1, 8));

    if debug {
        dbg!(net);
        dbg!(&evaluator);
    }

    for run in 0..runs {
        evaluator.reset_internal_state();
        let (mut recent_observation, _info) = env.reset(None).expect("Unable to reset");
        let mut total_reward = 0.0;

        if debug {
            dbg!(run);
        }

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

            standard_scaler.scale_inplace(observations.view_mut());

            // add bias input
            let input = concatenate![Axis(0), observations, [1.0]];
            let output = evaluator.evaluate(input.clone());

            let softmaxsum: f64 = output.iter().map(|x| x.exp()).sum();
            let softmax: Vec<f64> = output.iter().map(|x| x.exp() / softmaxsum).collect();
            let dist = WeightedIndex::new(&softmax).unwrap();

            if debug {
                dbg!(&input);
                dbg!(&output);
            }

            let (observation, reward, is_done) = match env.step(actions[dist.sample(&mut rng)]) {
                Ok(State {
                    observation,
                    reward,
                    is_done,
                    ..
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
