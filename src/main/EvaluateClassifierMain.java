package main;

import utils.EvaluateClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EvaluateClassifierMain {

	public static void main(String[] args) throws Exception {

		// variables
		int numFolds = 10;
		int numSeeds = 1;

		// load training data
		Instances data = DataSource
				.read("/Users/nadim/Workspace/weka-3-7-10/data/weather.nominal.arff");
		data.setClassIndex(data.numAttributes() - 1);

		// select a classifier
		Classifier cl = new ZeroR();
		cl.buildClassifier(data);

		// performs 10-fold Cross-Validation to baseline classifier
		EvaluateClassifier ec = new EvaluateClassifier();
		Evaluation eval = ec.crossValidation(cl, data, numFolds, numSeeds);
		ec.printInstancesData(data);
		ec.printClassifierOutput(data, cl, eval);

		// getCurveROC
		ec.plotCurveROC(eval);

		// getConfusionMatrix
		// ec.getConfusionMatrix(eval);

		// getKappa
		// ec.getKappa(eval);

	}
}
