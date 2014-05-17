package main;

import weka.classifiers.Classifier;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RecomenderMain {

	public static void predict(String name, Integer total_nominations,
			String category_next, String awarded_next) throws Exception {
		// load training data
		Instances data = DataSource.read("oscars.training.arff");
		data.setClassIndex(data.numAttributes() - 1);

		// train classifier
		Classifier cl = new VotedPerceptron();
		cl.buildClassifier(data);

		// create new set of data
		Instances newData = new Instances(data, 1);
		Instance inst = new DenseInstance(5);
		inst.setDataset(newData);

		// output on stdout
		System.out.println(data);
		System.out.println(newData);

		System.out.println("Training finished!\n");

		// add attribute values
		inst.setValue(0, name);
		inst.setValue(1, total_nominations);
		inst.setValue(2, category_next);
		inst.setValue(3, awarded_next);

		// predict class
		Double pred = cl.classifyInstance(inst);
		Double classValue = inst.classValue();
		System.out.println(inst);
		System.out.println(classValue + " -> " + pred);

		System.out.println("Predicting finished!");
	}

	public static void main(String[] args) throws Exception {
		/**
		 * ATRIBUTES: name, total_nominations, category_next, available_next
		 * CLASS: class_repeat_nomination
		 */
		String name, category_next, awarded_next;
		Integer total_nominations;

		name = "Jeff-Bridges";
		total_nominations = 6;
		category_next = "bestActor";
		awarded_next = "no";

		predict(name, total_nominations, category_next, awarded_next);

	}

}
