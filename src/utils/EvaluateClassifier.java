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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public double getKappa(Evaluation eval) {
        // TODO Auto-generated method stub
        return 0;
    }

}
