use std::{cell::RefCell, ops::Deref};

use favannat::{
    matrix::fabricator::FeedForwardMatrixFabricator,
    neat_original::fabricator::NeatOriginalFabricator,
    network::{Evaluator, Fabricator, StatefulEvaluator, StatefulFabricator},
};
use log::info;
use ndarray::{array, Array1, Axis};
use set_neat::{Individual, Neat, Progress};

pub const ENV: &str = "xor";

fn main() {
    log4rs::init_file(format!("examples/{}/log.yaml", ENV), Default::default()).unwrap();

    let fitness_function = |individual: &Individual| -> Progress {
        let result_0;
        let result_1;
        let result_2;
        let result_3;

        /* match NeatOriginalFabricator::fabricate(individual.deref()) {
        Ok(mut evaluator) => { */
        match FeedForwardMatrixFabricator::fabricate(individual.deref()) {
            Ok(evaluator) => {
                result_0 = evaluator.evaluate(array![1.0, 1.0, 0.0]);
                result_1 = evaluator.evaluate(array![1.0, 1.0, 1.0]);
                result_2 = evaluator.evaluate(array![1.0, 0.0, 1.0]);
                result_3 = evaluator.evaluate(array![1.0, 0.0, 0.0]);
            }
            Err(e) => {
                println!("error fabricating individual: {:?} {:?}", individual, e);
                panic!("")
            }
        }

        // calculate fitness

        let fitness = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        if fitness > 15.9 {
            let mut winner = individual.clone();
            winner.fitness.raw = fitness;
            Progress::fitness(fitness).solved(winner)
        } else {
            Progress::fitness(fitness)
        }
    };

    let neat = Neat::new("examples/xor/config.toml", Box::new(fitness_function));

    let mut ff_connections_in_winner_in_run = Vec::new();
    let mut rc_connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();
    let mut generations_till_winner_in_run = Vec::new();
    let mut score_of_winner_in_run = Vec::new();

    let mut worst_possible = Individual::default();
    worst_possible.fitness.raw = f64::NEG_INFINITY;
    let all_time_best: RefCell<Individual> = RefCell::new(worst_possible);

    let mut generations;

    for run in 0..100 {
        generations = 1;

        if let Some(winner) = neat.run().find_map(|(statistics, solution)| {
            all_time_best.replace_with(|prev| {
                if statistics.population.top_performer.fitness.raw > prev.fitness.raw {
                    statistics.population.top_performer.clone()
                } else {
                    prev.clone()
                }
            });
            dbg!(
                "######################################## {}",
                generations,
                run,
            );

            // info!(target: "app::progress", "{}", serde_json::to_string(&statistics).unwrap());

            generations += 1;

            solution
        }) {
            ff_connections_in_winner_in_run.push(winner.feed_forward.len() as f64);
            rc_connections_in_winner_in_run.push(winner.recurrent.len() as f64);
            nodes_in_winner_in_run.push(winner.hidden.len() as f64);
            generations_till_winner_in_run.push(generations as f64);
            score_of_winner_in_run.push(winner.fitness.raw);
        }
    }

    let ff_connections_in_winner_in_run = Array1::from(ff_connections_in_winner_in_run);
    let rc_connections_in_winner_in_run = Array1::from(rc_connections_in_winner_in_run);
    let nodes_in_winner_in_run = Array1::from(nodes_in_winner_in_run);
    let generations_till_winner_in_run = Array1::from(generations_till_winner_in_run);
    let score_of_winner_in_run = Array1::from(score_of_winner_in_run);

    let f_std_dev = ff_connections_in_winner_in_run
        .var_axis(Axis(0), 0.0)
        .mapv_into(|x| (x + f64::EPSILON).sqrt());

    let f_avg = ff_connections_in_winner_in_run.mean_axis(Axis(0)).unwrap();

    let r_std_dev = rc_connections_in_winner_in_run
        .var_axis(Axis(0), 0.0)
        .mapv_into(|x| (x + f64::EPSILON).sqrt());

    let r_avg = rc_connections_in_winner_in_run.mean_axis(Axis(0)).unwrap();

    let h_std_dev = nodes_in_winner_in_run
        .var_axis(Axis(0), 0.0)
        .mapv_into(|x| (x + f64::EPSILON).sqrt());

    let h_avg = nodes_in_winner_in_run.mean_axis(Axis(0)).unwrap();

    let gens_std_dev = generations_till_winner_in_run
        .var_axis(Axis(0), 0.0)
        .mapv_into(|x| (x + f64::EPSILON).sqrt());

    let gens_avg = generations_till_winner_in_run.mean_axis(Axis(0)).unwrap();

    let score_std_dev = score_of_winner_in_run
        .var_axis(Axis(0), 0.0)
        .mapv_into(|x| (x + f64::EPSILON).sqrt());

    let score_avg = score_of_winner_in_run.mean_axis(Axis(0)).unwrap();

    info!(
        target: "app::solutions",
        "|H| {} (+/- {}), |F| {} (+/- {}), |R| {} (+/- {}), #gens {} (+/- {}), avg_score {} (+/- {})",
        h_avg, h_std_dev, f_avg, f_std_dev, r_avg, r_std_dev, gens_avg, gens_std_dev, score_avg, score_std_dev
    );

    let all_time_best = all_time_best.into_inner();

    info!(
        target: "app::solutions",
        "all_time_best: |H| {}, |F| {}, |R| {}, #gens {}, avg_score {}",
        all_time_best.hidden.len(),
        all_time_best.feed_forward.len(),
        all_time_best.recurrent.len(),
        f64::NAN,
        all_time_best.fitness.raw
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn fitness_function_good_result() {
        let result_0: Vec<f64> = vec![1.0];
        let result_1: Vec<f64> = vec![0.0];
        let result_2: Vec<f64> = vec![1.0];
        let result_3: Vec<f64> = vec![0.0];

        let result = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        println!("result {:?}", res/*  */ult);

        assert_eq!(result, 16.0);
    }

    #[test]
    fn fitness_function_bad_result() {
        let result_0: Vec<f64> = vec![0.0];
        let result_1: Vec<f64> = vec![1.0];
        let result_2: Vec<f64> = vec![0.0];
        let result_3: Vec<f64> = vec![1.0];

        let result = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        println!("result {:?}", result);

        assert_eq!(result, 0.0);
    }
}
