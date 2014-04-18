package utils;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.classifiers.evaluation.Evaluation;

public class EvaluateClassifier implements EvaluateClassifierInterface {

	@Override
	public Evaluation crossValidation(Classifier classifier, Instances data,
			int numFolds, int numSeeds) throws Exception {

		Evaluation eval = new Evaluation(data);
		Random rand = new Random(numSeeds);

		eval.crossValidateModel(classifier, data, numFolds, rand);

		System.out.println(eval.toClassDetailsString());

		return eval;

	}

	@Override
	public double[][] getConfusionMatrix(Evaluation eval) {
		double[][] cnMatrix = eval.confusionMatrix();
		for (int row_i = 0; row_i < cnMatrix.length; row_i++) {
			for (int col_i = 0; col_i < cnMatrix.length; col_i++) {
				System.out.print(cnMatrix[row_i][col_i]);
				System.out.print("|");
			}
			System.out.println();
		}
		return cnMatrix;
	}

	@Override
	public double getKappa(Evaluation eval) {
		double kappaIndex = eval.kappa();
		return kappaIndex;
	}

}
