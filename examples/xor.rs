/* use favannat::matrix::fabricator::MatrixFabricator;
use favannat::network::{Evaluator, Fabricator};
use ndarray::array;
use set_neat::{Genome, Neat, Progress, Solution};
use std::time::Instant;

fn main() {
    fn fitness_function(genome: &Genome) -> f64 {
        /* let result_rr0;
        let result_rr1;
        let result_rr2; */
        let result_0;
        let result_1;
        let result_2;
        let result_3;

        match MatrixFabricator::fabricate(genome) {
            Ok(evaluator) => {
                result_0 = evaluator.evaluate(array![1.0, 1.0, 0.0]);
                result_1 = evaluator.evaluate(array![1.0, 1.0, 1.0]);
                result_2 = evaluator.evaluate(array![1.0, 0.0, 1.0]);
                result_3 = evaluator.evaluate(array![1.0, 0.0, 0.0]);

                /* result_rr0 = [
                    evaluator.evaluate(array![1.0, 1.0, 0.0]),
                    evaluator.evaluate(array![1.0, 1.0, 1.0]),
                    evaluator.evaluate(array![1.0, 0.0, 1.0]),
                    evaluator.evaluate(array![1.0, 0.0, 0.0]),
                ];

                result_rr1 = [
                    evaluator.evaluate(array![0.0, 1.0, 1.0]),
                    evaluator.evaluate(array![1.0, 1.0, 1.0]),
                    evaluator.evaluate(array![1.0, 1.0, 0.0]),
                    evaluator.evaluate(array![0.0, 1.0, 0.0]),
                ];

                result_rr2 = [
                    evaluator.evaluate(array![1.0, 0.0, 1.0]),
                    evaluator.evaluate(array![1.0, 1.0, 1.0]),
                    evaluator.evaluate(array![0.0, 1.0, 1.0]),
                    evaluator.evaluate(array![0.0, 0.0, 1.0]),
                ]; */
            }
            Err(e) => {
                println!("error fabricating genome: {:?} {:?}", genome, e);
                panic!("")
            }
        }

        // calculate fitness
        (4.0 - ((1.0 - result_0[0])
            + (0.0 - result_1[0]).abs()
            + (1.0 - result_2[0])
            + (0.0 - result_3[0]).abs()))
        .powi(2)

        /* let rr0 = (4.0 - ((1.0 - result_rr0[0][0])
                + (0.0 - result_rr0[1][0]).abs()
                + (1.0 - result_rr0[2][0])
                + (0.0 - result_rr0[3][0]).abs()))
        .powi(2);
        let rr1 = (4.0 - ((1.0 - result_rr1[0][0])
                + (0.0 - result_rr1[1][0]).abs()
                + (1.0 - result_rr1[2][0])
                + (0.0 - result_rr1[3][0]).abs()))
        .powi(2);
        let rr2 = (4.0 - ((1.0 - result_rr2[0][0])
                + (0.0 - result_rr2[1][0]).abs()
                + (1.0 - result_rr2[2][0])
                + (0.0 - result_rr2[3][0]).abs()))
        .powi(2);

        (rr0 + rr1 + rr2) / 3.0 */
    }

    let neat = Neat::new("examples/XOR.toml", fitness_function, 15.9);

    /* let mut millis_elapsed_in_run = Vec::new();
    let mut connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();
    let mut generations_till_winner_in_run = Vec::new();

    for i in 0..100 {
        let now = Instant::now();
        let mut generations = 0;

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Progress(report) => {
                    generations = report.num_generation;
                    None
                }
                Solution(genome) => Some(genome),
            })
            .next()
        {
            millis_elapsed_in_run.push(now.elapsed().as_millis() as f64);
            connections_in_winner_in_run.push(winner.connection_genes.len());
            nodes_in_winner_in_run.push(winner.node_genes.len());
            generations_till_winner_in_run.push(generations);
            println!(
                "finished run {} in {} seconds ({}, {})",
                i,
                millis_elapsed_in_run.last().unwrap() / 1000.0,
                winner.node_genes.len(),
                winner.connection_genes.len()
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
    ); */

    let now = Instant::now();

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Progress(report) => {
                println!("{:#?}", report);
                None
            }
            Solution(genome) => Some(genome),
        })
        .next()
    {
        let secs = now.elapsed().as_millis();
        println!(
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.node_genes.len(),
            winner.connection_genes.len(),
            secs as f64 / 1000.0,
            winner
        );
        let evaluator = MatrixFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);
    }
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

        println!("result {:?}", result);

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
 */
