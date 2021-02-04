use std::cell::RefCell;

use favannat::{
    matrix::fabricator::FeedForwardMatrixFabricator,
    neat_original::fabricator::NeatOriginalFabricator,
    network::{Evaluator, Fabricator, StatefulEvaluator, StatefulFabricator},
};
use ndarray::array;
use set_neat::{Individual, Neat, Progress};

fn main() {
    let fitness_function = |individual: &Individual| -> Progress {
        let result_0;
        let result_1;
        let result_2;
        let result_3;

        match NeatOriginalFabricator::fabricate(individual) {
            Ok(mut evaluator) => {
                /* match FeedForwardMatrixFabricator::fabricate(individual) {
                Ok(evaluator) => { */
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

    for _ in 0..100 {
        generations = 1;

        if let Some(winner) = neat.run().find_map(|(statistics, solution)| {
            all_time_best.replace_with(|prev| {
                if statistics.population.top_performer.fitness.raw > prev.fitness.raw {
                    statistics.population.top_performer.clone()
                } else {
                    prev.clone()
                }
            });
            generations += 1;

            solution
        }) {
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

    println!(
        "|H| {}, |F| {}, |R| {}, #gens {}, avg_score {}",
        avg_H, avg_F, avg_R, avg_generations, avg_score
    );

    let all_time_best = all_time_best.into_inner();

    println!(
        "all_time_best: |H| {}, |F| {}, |R| {}, #gens {}, avg_score {}",
        all_time_best.hidden.len(),
        all_time_best.feed_forward.len(),
        all_time_best.recurrent.len(),
        avg_generations,
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
