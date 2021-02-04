use favannat::{
    matrix::fabricator::FeedForwardMatrixFabricator,
    neat_original::fabricator::NeatOriginalFabricator,
    network::{Evaluator, Fabricator, StatefulEvaluator, StatefulFabricator},
};
use ndarray::array;
use set_neat::{Evaluation, Individual, Neat, Progress};
use std::time::Instant;

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
            Progress::fitness(fitness).solved(individual.clone())
        } else {
            Progress::fitness(fitness)
        }
    };

    let neat = Neat::new("examples/xor/config.toml", Box::new(fitness_function));

    let mut millis_elapsed_in_run = Vec::new();
    let mut connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();
    let mut generations_till_winner_in_run = Vec::new();

    for i in 0..100 {
        let now = Instant::now();
        let mut generations = 0;

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Evaluation::Progress(report) => {
                    generations = report.population.num_generation;
                    None
                }
                Evaluation::Solution(individual) => Some(individual),
            })
            .next()
        {
            millis_elapsed_in_run.push(now.elapsed().as_millis() as f64);
            connections_in_winner_in_run.push(winner.feed_forward.len());
            nodes_in_winner_in_run.push(winner.nodes().count());
            generations_till_winner_in_run.push(generations);
            println!(
                "finished run {} in {} seconds ({}, {}) {}",
                i,
                millis_elapsed_in_run.last().unwrap() / 1000.0,
                winner.nodes().count(),
                winner.feed_forward.len(),
                generations
            );
        }
    }

    let num_runs = millis_elapsed_in_run.len() as f64;

    let total_millis: f64 = millis_elapsed_in_run.iter().sum();
    let total_connections: usize = connections_in_winner_in_run.iter().sum();
    let total_nodes: usize = nodes_in_winner_in_run.iter().sum();
    let total_generations: usize = generations_till_winner_in_run.iter().sum();

    println!(
        "did {} runs in {} seconds / {} nodes average / {} connections / {} generations per run",
        num_runs,
        total_millis / num_runs / 1000.0,
        total_nodes as f64 / num_runs,
        total_connections as f64 / num_runs,
        total_generations as f64 / num_runs
    );

    /* let now = Instant::now();

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                println!("{:#?}", report);
                None
            }
            Evaluation::Solution(individual) => Some(individual),
        })
        .next()
    {
        let secs = now.elapsed().as_millis();
        println!(
            "winning individual ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            secs as f64 / 1000.0,
            winner
        );
        let evaluator = NeatOriginalFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);
    } */
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
