use criterion::{criterion_group, criterion_main, Criterion};
use set_neat::{IdGenerator, Individual, NeatRng, Parameters};

pub fn crossover_same_genome_benchmark(c: &mut Criterion) {
    let mut id_gen = IdGenerator::default();
    let parameters: Parameters = Default::default();

    // create randomn source
    let mut rng = NeatRng::new(
        parameters.setup.seed,
        parameters.mutation.weight_perturbation_std_dev,
    );

    let individual_0 = Individual::initial(&mut id_gen, &parameters);

    let individual_1 = individual_0.clone();

    c.bench_function("crossover same genome", |b| {
        b.iter(|| individual_0.crossover(&individual_1, &mut rng.small))
    });
}

pub fn crossover_higly_mutated_genomes_benchmark(c: &mut Criterion) {
    let mut id_gen = IdGenerator::default();
    let mut parameters: Parameters = Default::default();

    parameters.mutation.new_node_chance = 1.0;
    parameters.mutation.new_connection_chance = 1.0;

    // create randomn source
    let mut rng = NeatRng::new(
        parameters.setup.seed,
        parameters.mutation.weight_perturbation_std_dev,
    );

    let mut individual_0 = Individual::initial(&mut id_gen, &parameters);

    let mut individual_1 = individual_0.clone();

    for _ in 0..100 {
        individual_0.mutate(&mut rng, &mut id_gen, &parameters);
        individual_1.mutate(&mut rng, &mut id_gen, &parameters);
    }

    c.bench_function("crossover higly mutated genomes", |b| {
        b.iter(|| individual_0.crossover(&individual_1, &mut rng.small))
    });
}

pub fn mutate_genome_benchmark(c: &mut Criterion) {
    let mut id_gen = IdGenerator::default();
    let mut parameters: Parameters = Default::default();

    parameters.mutation.new_node_chance = 1.0;
    parameters.mutation.new_connection_chance = 1.0;

    // create randomn source
    let mut rng = NeatRng::new(
        parameters.setup.seed,
        parameters.mutation.weight_perturbation_std_dev,
    );

    let mut individual_0 = Individual::initial(&mut id_gen, &parameters);

    c.bench_function("mutate genome", |b| {
        b.iter(|| individual_0.mutate(&mut rng, &mut id_gen, &parameters))
    });
}

// Makes no sense, as connections can only be added when sufficient nodes are free
pub fn add_connection_to_genome_benchmark(c: &mut Criterion) {
    let mut id_gen = IdGenerator::default();
    let parameters: Parameters = Default::default();

    // create randomn source
    let mut rng = NeatRng::new(
        parameters.setup.seed,
        parameters.mutation.weight_perturbation_std_dev,
    );

    let mut individual_0 = Individual::initial(&mut id_gen, &parameters);

    c.bench_function("add connection to genome", |b| {
        b.iter(|| individual_0.add_connection(&mut rng, &parameters))
    });
}

pub fn add_node_to_genome_benchmark(c: &mut Criterion) {
    let mut id_gen = IdGenerator::default();
    let parameters: Parameters = Default::default();

    // create randomn source
    let mut rng = NeatRng::new(
        parameters.setup.seed,
        parameters.mutation.weight_perturbation_std_dev,
    );

    let mut individual_0 = Individual::initial(&mut id_gen, &parameters);

    c.bench_function("add connection to genome", |b| {
        b.iter(|| individual_0.add_node(&mut rng, &mut id_gen, &parameters))
    });
}

criterion_group!(
    benches,
    // mutate_genome_benchmark,
    crossover_same_genome_benchmark,
    crossover_higly_mutated_genomes_benchmark,
    // add_connection_to_genome_benchmark,
    add_node_to_genome_benchmark
);
criterion_main!(benches);
