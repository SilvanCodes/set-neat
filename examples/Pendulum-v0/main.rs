use favannat::{MatrixRecurrentFabricator, StatefulEvaluator, StatefulFabricator};
use gym::client::MakeOptions;
use gym::space_data::SpaceData;
use gym::space_template::SpaceTemplate;
use gym::{utility::StandardScaler, State};
use ndarray::{concatenate, stack, Array2, Axis};
use set_genome::Connection;
use set_neat::{Individual, Neat, Progress};

use log::{error, info};
use std::{cell::RefCell, time::Instant};
use std::{env, fs};
use std::{ops::Deref, time::SystemTime};

pub const RUNS: usize = 1;
pub const STEPS: usize = usize::MAX;
pub const VALIDATION_RUNS: usize = 100;
pub const ENV: &str = "Pendulum-v0";
pub const REQUIRED_FITNESS: f64 = -130.0;
pub const GENERATIONS: usize = 1000;

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(timestamp) = args.get(1) {
        let winner_json =
            fs::read_to_string(format!("examples/{}/winner.json", ENV)).expect("cant read file");
        let winner: Individual = serde_json::from_str(&winner_json).unwrap();
        let scaler_json =
            fs::read_to_string(format!("examples/{}/winner_standard_scaler.json", ENV))
                .expect("cant read file");
        let standard_scaler: StandardScaler = serde_json::from_str(&scaler_json).unwrap();
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

        if fitness.is_nan() {
            println!("environment gave nan fitness");
            dbg!(individual);
            dbg!(fitness);
            dbg!(all_observations);
            return Progress::empty();
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");

            let (validation_fitness, _) = run(
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
                return Progress::fitness(validation_fitness).solved(individual);
            }
        }

        Progress::fitness(fitness)
    };

    let neat = Neat::new(
        &format!("examples/{}/config.toml", ENV),
        Box::new(fitness_function),
    );

    let now = Instant::now();

    info!(target: "app::parameters", "starting training...\nRUNS:{:#?}\nVALIDATION_RUNS:{:#?}\nSTEPS: {:#?}\nREQUIRED_FITNESS:{:#?}\nPARAMETERS: {:#?}", RUNS, VALIDATION_RUNS, STEPS, REQUIRED_FITNESS, neat.parameters);

    let mut ff_connections_in_winner_in_run = Vec::new();
    let mut rc_connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();
    let mut generations_till_winner_in_run = Vec::new();
    let mut score_of_winner_in_run = Vec::new();

    let mut worst_possible = Individual::default();
    worst_possible.fitness.raw = f64::NEG_INFINITY;
    let all_time_best: RefCell<Individual> = RefCell::new(worst_possible);

    let mut generations = 1;

    for _ in 0..1 {
        generations = 1;
        if let Some(winner) = neat
            .run()
            .take(GENERATIONS)
            .find_map(|(statistics, solution)| {
                all_time_best.replace_with(|prev| {
                    if statistics.population.top_performer.fitness.raw > prev.fitness.raw {
                        statistics.population.top_performer.clone()
                    } else {
                        prev.clone()
                    }
                });
                generations += 1;

                info!(target: "app::progress", "{}", serde_json::to_string(&statistics).unwrap());

                solution
            })
        {
            /* let time_stamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            fs::write(
                format!("examples/{}/winner.json", ENV),
                serde_json::to_string(&winner).unwrap(),
            )
            .expect("Unable to write file");
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
            fs::write(
                format!("examples/{}/winner_standard_scaler.json", ENV),
                serde_json::to_string(&other_standard_scaler).unwrap(),
            )
            .expect("Unable to write file");

            let secs = now.elapsed().as_secs();
            info!(
                "winning individual ({},{}) after {} seconds: {:?}",
                winner.nodes().count(),
                winner.feed_forward.len(),
                secs,
                winner
            ); */

            ff_connections_in_winner_in_run.push(winner.feed_forward.len());
            rc_connections_in_winner_in_run.push(winner.recurrent.len());
            nodes_in_winner_in_run.push(winner.hidden.len());
            generations_till_winner_in_run.push(generations);
            score_of_winner_in_run.push(winner.fitness.raw);
        }
    }

    let avg_F = ff_connections_in_winner_in_run.iter().sum::<usize>() as f64
        / ff_connections_in_winner_in_run.len() as f64;
    let avg_R = rc_connections_in_winner_in_run.iter().sum::<usize>() as f64
        / rc_connections_in_winner_in_run.len() as f64;
    let avg_H =
        nodes_in_winner_in_run.iter().sum::<usize>() as f64 / nodes_in_winner_in_run.len() as f64;
    let avg_generations = generations_till_winner_in_run.iter().sum::<usize>() as f64
        / nodes_in_winner_in_run.len() as f64;
    let avg_score =
        score_of_winner_in_run.iter().sum::<f64>() as f64 / score_of_winner_in_run.len() as f64;

    info!(
        target: "app::solutions",
        "|H| {}, |F| {}, |R| {}, #gens {}, avg_score {}",
        avg_H, avg_F, avg_R, avg_generations, avg_score
    );

    let all_time_best = all_time_best.into_inner();

    info!(
        target: "app::solutions",
        "all_time_best: |H| {}, |F| {}, |R| {}, #gens {}, avg_score {}",
        all_time_best.hidden.len(),
        all_time_best.feed_forward.len(),
        all_time_best.recurrent.len(),
        generations,
        all_time_best.fitness.raw
    );
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

    let mut evaluator = MatrixRecurrentFabricator::fabricate(net.deref()).unwrap();
    let mut fitness = 0.0;

    let mut all_observations;

    if let SpaceTemplate::Box { shape, .. } = env.observation_space() {
        all_observations = Array2::zeros((1, shape[0]));
    } else {
        panic!("is no box observation space")
    }

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

            if debug {
                dbg!(&input);
                dbg!(&output);
            }

            let (observation, reward, is_done) =
                match env.step(&SpaceData::Box(output.clone() * 2.0)) {
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
