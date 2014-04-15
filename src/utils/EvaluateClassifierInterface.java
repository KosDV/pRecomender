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
     * @param numFolds, recommended numFolds = 10;
     * @param numSeeds, i.e numSeeds = 1;
     * @return the evaluated Evaluation
     * @throws Exception
     */
    public Evaluation crossValidation(Classifier classifier, Instances data,
            int numFolds, int numSeeds) throws Exception;

    /**
     * 
     * @param eval
     * @return a copy of the confusion matrix as a two-dimensional array
     */
    public double[][] getConfusionMatrix(Evaluation eval);

    /**
     * 
     * @param eval
     * @return the value of kappa statistic if class is nominal
     */
    public double getKappa(Evaluation eval);
}
