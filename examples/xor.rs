use favannat::network::{Fabricator, Evaluator};
use favannat::matrix::fabricator::MatrixFabricator;
use set_neat::{Neat, Genome, Progress, Solution};
use ndarray::array;
use std::time::Instant;

fn main() {
    fn fitness_function(genome: &Genome) -> f64 {
        let result_0;
        let result_1;
        let result_2;
        let result_3;

        match MatrixFabricator::fabricate(genome) {
            Ok(evaluator) => {
                result_0 = evaluator.evaluate(array![1.0, 1.0, 0.0]);// 2.0 + 0.5;
                result_1 = evaluator.evaluate(array![1.0, 1.0, 1.0]);// 2.0 + 0.5;
                result_2 = evaluator.evaluate(array![1.0, 0.0, 1.0]);// 2.0 + 0.5;
                result_3 = evaluator.evaluate(array![1.0, 0.0, 0.0]);// 2.0 + 0.5;
            },
            Err(e) => {
                println!("error fabricating genome: {:?} {:?}", genome, e);
                panic!("")
            }
        }

        // calculate fitness
        (4.0 - ((1.0 - result_0[0]) + (0.0 - result_1[0]).abs() + (1.0 - result_2[0]) + (0.0 - result_3[0]).abs())).powi(2)
    }

    let neat = Neat::new("examples/XOR.toml", fitness_function, 15.0);

    let mut millis_elapsed_in_run = Vec::new();
    let mut connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();

    for i in 0..100 {
        let now = Instant::now();

        if let Some(winner) = neat.run().filter_map(|evaluation| {
            match evaluation {
                Progress(_) => None,
                Solution(genome) => Some(genome)
            }
        }).next() {
            millis_elapsed_in_run.push(now.elapsed().as_millis() as f64);
            connections_in_winner_in_run.push(winner.connection_genes.len());
            nodes_in_winner_in_run.push(winner.node_genes.len());
            println!("finished run {} in {} seconds ({}, {})", i, millis_elapsed_in_run.last().unwrap() / 1000.0, winner.node_genes.len(), winner.connection_genes.len());
        }
    }

    let num_runs = millis_elapsed_in_run.len() as f64;

    let total_millis: f64 = millis_elapsed_in_run.iter().sum();
    let total_connections: usize = connections_in_winner_in_run.iter().sum();
    let total_nodes: usize = nodes_in_winner_in_run.iter().sum();

    println!(
        "did {} runs in {} seconds / {} nodes average / {} connections  per run",
        num_runs, total_millis / num_runs / 1000.0,
        total_nodes as f64 / num_runs,
        total_connections as f64 / num_runs
    );

    /* let now = Instant::now();

    if let Some(winner) = neat.run().filter_map(|evaluation| {
        match evaluation {
            Progress(report) => {println!("{:#?}", report); None},
            Solution(genome) => Some(genome)
        }
    }).next() {
        let secs = now.elapsed().as_millis();
        println!("winning genome ({},{}) after {} seconds: {:?}",winner.node_genes.len(), winner.connection_genes.len(), secs as f64 / 1000.0, winner);
        let evaluator = MatrixFabricator::fabricate(&winner).unwrap();
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

        let result = (4.0 - ((1.0 - result_0[0]) + (0.0 - result_1[0]).abs() + (1.0 - result_2[0]) + (0.0 - result_3[0]).abs())).powi(2);

        println!("result {:?}", result);

        assert_eq!(result, 16.0);
    }

    #[test]
    fn fitness_function_bad_result() {
        let result_0: Vec<f64> = vec![0.0];
        let result_1: Vec<f64> = vec![1.0];
        let result_2: Vec<f64> = vec![0.0];
        let result_3: Vec<f64> = vec![1.0];

        let result = (4.0 - ((1.0 - result_0[0]) + (0.0 - result_1[0]).abs() + (1.0 - result_2[0]) + (0.0 - result_3[0]).abs())).powi(2);

        println!("result {:?}", result);

        assert_eq!(result, 0.0);
    }
}