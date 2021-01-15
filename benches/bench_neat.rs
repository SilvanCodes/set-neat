use criterion::{criterion_group, criterion_main, Criterion};
use set_neat::{activations::Activation, Context, Genome, Parameters};

pub fn crossover_same_genome_benchmark(c: &mut Criterion) {
    let mut parameters: Parameters = Default::default();
    let mut context = Context::new(&parameters);

    parameters.setup.dimension.input = 1;
    parameters.setup.dimension.output = 1;

    let mut genome_0 = Genome::new(&mut context, &parameters);

    genome_0.init(&mut context, &parameters);

    let genome_1 = genome_0.clone();

    c.bench_function("crossover same genome", |b| {
        b.iter(|| genome_0.crossover(&genome_1, &mut context))
    });
}

pub fn crossover_higly_mutated_genomes_benchmark(c: &mut Criterion) {
    let mut parameters: Parameters = Default::default();
    parameters.initialization.activations = vec![Activation::Absolute, Activation::Cosine];
    let mut context = Context::new(&parameters);

    parameters.setup.dimension.input = 1;
    parameters.setup.dimension.output = 1;

    parameters.mutation.gene_node = 1.0;
    parameters.mutation.gene_connection = 1.0;

    let mut genome_0 = Genome::new(&mut context, &parameters);

    genome_0.init(&mut context, &parameters);

    let mut genome_1 = genome_0.clone();

    for _ in 0..100 {
        genome_0.mutate(&mut context, &parameters);
        genome_1.mutate(&mut context, &parameters);
    }

    c.bench_function("crossover higly mutated genomes", |b| {
        b.iter(|| genome_0.crossover(&genome_1, &mut context))
    });
}

pub fn mutate_genome_benchmark(c: &mut Criterion) {
    let mut parameters: Parameters = Default::default();
    let mut context = Context::new(&parameters);

    parameters.setup.dimension.input = 1;
    parameters.setup.dimension.output = 1;

    parameters.mutation.gene_node = 1.0;
    parameters.mutation.gene_connection = 1.0;

    let mut genome_0 = Genome::new(&mut context, &parameters);

    genome_0.init(&mut context, &parameters);

    c.bench_function("mutate genome", |b| {
        b.iter(|| genome_0.mutate(&mut context, &parameters))
    });
}

pub fn add_connection_to_genome_benchmark(c: &mut Criterion) {
    let mut parameters: Parameters = Default::default();
    parameters.initialization.activations = vec![Activation::Absolute, Activation::Cosine];
    let mut context = Context::new(&parameters);

    parameters.setup.dimension.input = 1;
    parameters.setup.dimension.output = 1;

    let mut genome_0 = Genome::new(&mut context, &parameters);

    c.bench_function("add connection to genome", |b| {
        b.iter(|| genome_0.add_connection(&mut context, &parameters))
    });
}

pub fn add_node_to_genome_benchmark(c: &mut Criterion) {
    let mut parameters: Parameters = Default::default();
    parameters.initialization.activations = vec![Activation::Absolute, Activation::Cosine];
    let mut context = Context::new(&parameters);

    parameters.setup.dimension.input = 1;
    parameters.setup.dimension.output = 1;

    let mut genome_0 = Genome::new(&mut context, &parameters);

    genome_0.init(&mut context, &parameters);

    c.bench_function("add node to genome", |b| {
        b.iter(|| genome_0.add_node(&mut context, &parameters))
    });
}

criterion_group!(
    benches,
    // mutate_genome_benchmark,
    crossover_same_genome_benchmark,
    crossover_higly_mutated_genomes_benchmark,
    add_connection_to_genome_benchmark,
    add_node_to_genome_benchmark
);
criterion_main!(benches);
