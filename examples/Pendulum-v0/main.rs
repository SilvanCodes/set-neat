use favannat::{
    looping::fabricator::LoopingFabricator,
    matrix::fabricator::RecurrentMatrixFabricator,
    network::{StatefulEvaluator, StatefulFabricator},
};
use gym::{SpaceData, SpaceTemplate, State};
use ndarray::{stack, Array2, Axis};
use set_neat::{utility::gym::StandardScaler, Evaluation, Genome, Neat, Progress};

use log::{error, info};
use std::time::Instant;
use std::time::SystemTime;
use std::{env, fs};

pub const RUNS: usize = 10;
pub const STEPS: usize = usize::MAX;
pub const VALIDATION_RUNS: usize = 100;
pub const ENV: &str = "Pendulum-v0";
pub const REQUIRED_FITNESS: f64 = -100.0;

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(timestamp) = args.get(1) {
        let winner_json =
            fs::read_to_string(format!("examples/{}/winner.json", ENV)).expect("cant read file");
        let winner: Genome = serde_json::from_str(&winner_json).unwrap();
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

    let fitness_function = move |genome: &Genome| -> Progress {
        let standard_scaler = &standard_scaler;

        let (fitness, all_observations) = run(standard_scaler, genome, RUNS, STEPS, false, false);

        if fitness.is_nan() {
            println!("environment gave nan fitness");
            dbg!(genome);
            dbg!(fitness);
            dbg!(all_observations);
            return Progress::empty();
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");

            let (validation_fitness, _) = run(
                &standard_scaler,
                genome,
                VALIDATION_RUNS,
                STEPS,
                false,
                false,
            );

            // log possible solutions to file
            let mut genome = genome.clone();
            genome.set_fitness(validation_fitness);
            info!(target: "app::solutions", "{}", serde_json::to_string(&genome).unwrap());
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                return Progress::fitness(validation_fitness).solved(genome);
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

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                info!(target: "app::progress", "{}", serde_json::to_string(&report).unwrap());
                // render top performer every x runs
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
            Evaluation::Solution(genome) => Some(genome),
        })
        .next()
    {
        let time_stamp = SystemTime::now()
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
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            secs,
            winner
        );
    }
}

fn run(
    standard_scaler: &StandardScaler,
    net: &Genome,
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

    let mut all_observations;

    if let SpaceTemplate::BOX { shape, .. } = env.observation_space() {
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

            let (observation, reward, is_done) =
                match env.step(&SpaceData::BOX(output.clone() * 2.0)) {
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