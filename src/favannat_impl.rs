use std::collections::HashMap;

use favannat::network::{EdgeLike, NetLike, NodeLike, Recurrent};

use crate::{
    activations::{self, Activation},
    genes::{ConnectionGene, Id, NodeGene, Weight},
    Genome,
};

impl NodeLike for NodeGene {
    fn id(&self) -> usize {
        self.id.0
    }
    fn activation(&self) -> fn(f64) -> f64 {
        match self.activation {
            Activation::Linear => activations::LINEAR,
            Activation::Sigmoid => activations::SIGMOID,
            Activation::Gaussian => activations::GAUSSIAN,
            Activation::Tanh => activations::TANH,
            Activation::Step => activations::STEP,
            Activation::Sine => activations::SINE,
            Activation::Cosine => activations::COSINE,
            Activation::Inverse => activations::INVERSE,
            Activation::Absolute => activations::ABSOLUTE,
            Activation::Relu => activations::RELU,
            Activation::Squared => activations::SQUARED,
        }
    }
}

impl EdgeLike for ConnectionGene {
    fn start(&self) -> usize {
        self.input.0
    }
    fn end(&self) -> usize {
        self.output.0
    }
    fn weight(&self) -> f64 {
        self.weight.0
    }
}

impl NetLike<NodeGene, ConnectionGene> for Genome {
    fn nodes(&self) -> Vec<&NodeGene> {
        let mut nodes: Vec<&NodeGene> = self.node_genes.iter().collect();

        nodes.sort_unstable();
        nodes
    }
    fn edges(&self) -> Vec<&ConnectionGene> {
        let mut edges: Vec<&ConnectionGene> = self.connection_genes.iter().collect();

        edges.sort_unstable();
        edges
    }
    fn inputs(&self) -> Vec<&NodeGene> {
        let mut inputs: Vec<&NodeGene> = self
            .node_genes
            .iter()
            .filter(|node_gene| node_gene.is_input())
            .collect();

        inputs.sort_unstable();
        inputs
    }
    fn outputs(&self) -> Vec<&NodeGene> {
        let mut outputs: Vec<&NodeGene> = self
            .node_genes
            .iter()
            .filter(|node_gene| node_gene.is_output())
            .collect();

        outputs.sort_unstable();
        outputs
    }
}

impl Recurrent<NodeGene, ConnectionGene> for Genome {
    type Net = Genome;

    fn unroll(&self) -> Self::Net {
        let mut unrolled_genome = Genome::from(self);

        // maps recurrent connection input to wrapped actual input
        let mut unroll_map: HashMap<Id, Id> = HashMap::new();
        let mut tmp_ids = (0..usize::MAX).rev();

        for recurrent_connection in &self.recurrent_connection_genes {
            let recurrent_input =
                unroll_map
                    .entry(recurrent_connection.input)
                    .or_insert_with(|| {
                        let recurrent_input_id = Id(tmp_ids.next().unwrap());

                        let recurrent_input = NodeGene::input(recurrent_input_id);
                        let recurrent_output =
                            NodeGene::output(Id(tmp_ids.next().unwrap()), Some(Activation::Linear));

                        // used to carry value into next evaluation
                        let outward_wrapping_connection = ConnectionGene::new(
                            recurrent_connection.input,
                            recurrent_output.id,
                            Some(Weight(1.0)),
                        );

                        // add nodes for wrapping
                        unrolled_genome.node_genes.insert(recurrent_input);
                        unrolled_genome.node_genes.insert(recurrent_output);

                        // add outward wrapping connection
                        unrolled_genome
                            .connection_genes
                            .insert(outward_wrapping_connection);

                        recurrent_input_id
                    });

            let inward_wrapping_connection = ConnectionGene::new(
                *recurrent_input,
                recurrent_connection.output,
                Some(recurrent_connection.weight),
            );

            unrolled_genome
                .connection_genes
                .insert(inward_wrapping_connection);
        }
        unrolled_genome
    }

    fn memory(&self) -> usize {
        let mut sources: Vec<Id> = self
            .recurrent_connection_genes
            .iter()
            .map(|connection| connection.input)
            .collect();
        sources.sort_unstable();
        sources.dedup();
        sources.len()
    }
}

#[cfg(test)]
mod tests {
    use favannat::network::Recurrent;

    use crate::{Context, Genome, Parameters};

    #[test]
    fn unroll_genome() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.mutation.recurrent = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        // should add recurrent connection from input to output
        assert!(genome_0.add_connection(&mut context, &parameters).is_ok());
        // dont add same connection twice
        assert!(genome_0.add_connection(&mut context, &parameters).is_err());

        assert_eq!(genome_0.recurrent_connection_genes.len(), 1);

        let genome_1 = genome_0.unroll();

        assert_eq!(genome_1.node_genes.len(), 4);
        assert_eq!(genome_1.connection_genes.len(), 3);
    }
}
