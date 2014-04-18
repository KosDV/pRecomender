package utils;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public interface EvaluateClassifierInterface {
	/**
	 * Performs a n-Fold-crossValidation of a Classifier using the given
	 * Instances
	 * 
	 * @param classifier
	 * @param data
	 * @param numFolds
	 *            , recommended numFolds = 10;
	 * @param numSeeds
	 *            , i.e numSeeds = 1;
	 * @return evaluated Evaluation
	 * @throws Exception
	 */
	public Evaluation crossValidation(Classifier classifier, Instances data,
			int numFolds, int numSeeds) throws Exception;

	/**
	 * 
	 * @param evaluated
	 *            Evaluation
	 * @return a copy of the confusion matrix as a two-dimensional array
	 */
	public double[][] getConfusionMatrix(Evaluation eval);

	/**
	 * 
	 * @param evaluated
	 *            Evaluation
	 * @return value of Kappa statistic if class is nominal
	 */
	public double getKappa(Evaluation eval);

	/**
	 * based pretty much in FracPete (fracpete at waikato dot ac dot nz) code
	 * 
	 * @param evaluated
	 *            Evaluation
	 */
	public void plotCurveROC(Evaluation eval);

	public void printClassifierOutput(Instances data, Classifier cl,
			Evaluation eval) throws Exception;

	public void printInstancesData(Instances data);
}
