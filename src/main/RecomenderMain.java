package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.Classifier;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RecomenderMain {

	public static void predict(String name, Integer total_nominations,
			String category_next, String awarded_next) throws Exception {
		System.out.println("\n\n");
		System.out.println("*****************");
		System.out.println("* TRAINING DATA *");
		System.out.println("*****************");

		// load training data
		Instances data = DataSource.read("oscars.training.arff");
		data.setClassIndex(data.numAttributes() - 1);

		// train classifier
		Classifier cl = new VotedPerceptron();
		cl.buildClassifier(data);
		// output on stdout
		System.out.println(data);

		// create new set of data
		Instances newData = new Instances(data, 1);
		Instance inst = new DenseInstance(5);
		inst.setDataset(newData);
		Thread.sleep(1000);
		System.out.println("\n\n");
		System.out.println("*****************");
		System.out.println("*   TEST DATA   *");
		System.out.println("*****************");
		System.out.println(newData);

		// add attribute values
		inst.setValue(0, name);
		inst.setValue(1, total_nominations);
		inst.setValue(2, category_next);
		inst.setValue(3, awarded_next);
		System.out.println(inst);

		// predict class
		double pred = cl.classifyInstance(inst);
		Thread.sleep(1000);
		System.out.println("\n\n");
		System.out.println("*****************");
		System.out.println("*    RESULTS    *");
		System.out.println("*****************");
		// get prediction
		System.out.println(inst.stringValue(0) + " will repeat nomination? "
				+ data.classAttribute().value((int) pred).toUpperCase());
	}

	public static void main(String[] args) throws Exception {
		/**
		 * ATRIBUTES: name, total_nominations, category_next, available_next
		 * CLASS: class_repeat_nomination
		 */
		String name, category_next, awarded_next;
		Integer total_nominations;

		// name = "Jeff-Bridges";
		// total_nominations = 6;
		// category_next = "bestActor";
		// awarded_next = "no";
		try {
			BufferedReader bufferRead = new BufferedReader(
					new InputStreamReader(System.in));
			System.out.println("*****************");
			System.out.println("*    WELCOME    *");
			System.out.println("*****************");
			System.out
					.println("Please, introduces the following parameters...");
			System.out.println("Actor");
			name = bufferRead.readLine();
			System.out.println("Total nominations");
			total_nominations = Integer.parseInt(bufferRead.readLine());
			System.out.println("Category next");
			category_next = bufferRead.readLine();
			System.out.println("Awarded next");
			awarded_next = bufferRead.readLine();

			predict(name, total_nominations, category_next, awarded_next);

		} catch (IOException e) {
			e.getMessage();
		}

	}

}
