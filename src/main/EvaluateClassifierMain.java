package main;

import utils.EvaluateClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EvaluateClassifierMain {

	public static void main(String[] args) throws Exception {
		// load training data
		Instances data = DataSource
				.read("/Users/nadim/Workspace/weka-3-7-10/data/weather.nominal.arff");
		data.setClassIndex(data.numAttributes() - 1);

		// performs 10-fold Cross-Validation to baseline classifier
		Classifier cl = new ZeroR();
		int numFolds = 10;
		int numSeeds = 1;
		EvaluateClassifier ec = new EvaluateClassifier();
		System.out.println("Evaluating ZeroR Classifier (Baseline)");
		Evaluation eval = ec.crossValidation(cl, data, numFolds, numSeeds);

		// getConfusionMatrix
		System.out.println("The confusion Matrix");
		ec.getConfusionMatrix(eval);

		// getKappa
		System.out.println("The Kappa Index");
		ec.getKappa(eval);

		// getCurveROC
		System.out.println("Plotting Curve ROC...");
		ec.plotCurveROC(eval);

		System.out.println("Finish!");

	}
}
