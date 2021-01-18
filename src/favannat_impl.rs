use std::collections::HashMap;

use favannat::network::{EdgeLike, NetLike, NodeLike, Recurrent};

use crate::{
    genes::{activations, connections::Connection, nodes::Node, Activation, Id},
    individual::Individual,
};

impl NodeLike for Node {
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

impl EdgeLike for Connection {
    fn start(&self) -> usize {
        self.input.0
    }
    fn end(&self) -> usize {
        self.output.0
    }
    fn weight(&self) -> f64 {
        self.weight
    }
}

impl NetLike<Node, Connection> for Individual {
    fn nodes(&self) -> Vec<&Node> {
        /* let mut nodes: Vec<&Node> = self.nodes().collect();

        nodes.sort_unstable();
        nodes */
        self.genome.nodes().collect()
    }
    fn edges(&self) -> Vec<&Connection> {
        self.genome.feed_forward.as_sorted_vec()
    }
    fn inputs(&self) -> Vec<&Node> {
        self.genome.inputs.as_sorted_vec()
    }
    fn outputs(&self) -> Vec<&Node> {
        self.genome.outputs.as_sorted_vec()
    }
}

impl Recurrent<Node, Connection> for Individual {
    type Net = Individual;

    fn unroll(&self) -> Self::Net {
        let mut unrolled_genome = self.clone();

        // maps recurrent connection input to wrapped actual input
        let mut unroll_map: HashMap<Id, Id> = HashMap::new();
        let mut tmp_ids = (0..usize::MAX).rev();

        for recurrent_connection in self.genome.recurrent.as_sorted_vec() {
            let recurrent_input =
                unroll_map
                    .entry(recurrent_connection.input)
                    .or_insert_with(|| {
                        let wrapper_input_id = Id(tmp_ids.next().unwrap());

                        let wrapper_input_node = Node::new(wrapper_input_id, Activation::Linear);
                        let wrapper_output_node =
                            Node::new(Id(tmp_ids.next().unwrap()), Activation::Linear);

                        // used to carry value into next evaluation
                        let outward_wrapping_connection = Connection::new(
                            recurrent_connection.input,
                            1.0,
                            wrapper_output_node.id,
                        );

                        // add nodes for wrapping
                        unrolled_genome.inputs.insert(wrapper_input_node);
                        unrolled_genome.outputs.insert(wrapper_output_node);

                        // add outward wrapping connection
                        unrolled_genome
                            .feed_forward
                            .insert(outward_wrapping_connection);

                        wrapper_input_id
                    });

            let inward_wrapping_connection = Connection::new(
                recurrent_input.clone(),
                recurrent_connection.weight,
                recurrent_connection.output,
            );

            unrolled_genome
                .feed_forward
                .insert(inward_wrapping_connection);
        }
        unrolled_genome
    }

    fn recurrent_edges(&self) -> Vec<&Connection> {
        self.recurrent.as_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use favannat::network::Recurrent;

    // use crate::{Context, Genome, Parameters};

    /* #[test]
    fn unroll_genome() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation_std_dev = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.input_dimension = 1;
        parameters.setup.output_dimension = 1;
        parameters.mutation.connection_is_recurrent_chance = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        // should add recurrent connection from input to output
        assert!(genome_0.add_connection(&mut context, &parameters).is_ok());
        // dont add same connection twice
        assert!(genome_0.add_connection(&mut context, &parameters).is_err());

        assert_eq!(genome_0.recurrent.len(), 1);

        let genome_1 = genome_0.unroll();

        assert_eq!(genome_1.hidden.len(), 2);
        assert_eq!(genome_1.feed_forward.len(), 3);
    } */
}
