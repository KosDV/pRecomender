package utils;

import java.awt.BorderLayout;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

/**
 * Business Logic implementation of Classifier Evaluation
 *
 */
public class EvaluateClassifier implements EvaluateClassifierInterface {

	@Override
	public Evaluation crossValidation(Classifier classifier, Instances data,
			int numFolds, int numSeeds) throws Exception {

		Evaluation eval = new Evaluation(data);
		Random rand = new Random(numSeeds);

		eval.crossValidateModel(classifier, data, numFolds, rand);

		return eval;
	}

	@Override
	public double[][] getConfusionMatrix(Evaluation eval) {
		double[][] cnMatrix = eval.confusionMatrix();
		System.out.println("\nThe confusion Matrix");
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
		System.out.println("\nThe Kappa Index");
		System.out.println(kappaIndex);

		return kappaIndex;
	}

	@Override
	public void plotCurveROC(Evaluation eval) {
		System.out.println("\nPlotting Curve ROC...");
		// Generate the curve
		ThresholdCurve tc = new ThresholdCurve();
		Instances curve = tc.getCurve(eval.predictions());

		// Plot the curve
		ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
		tvp.setROCString("(Area under ROC = "
				+ Utils.doubleToString(ThresholdCurve.getROCArea(curve), 4)
				+ ")");
		tvp.setName(curve.relationName());
		PlotData2D plotData = new PlotData2D(curve);
		plotData.setPlotName(curve.relationName());
		plotData.addInstanceNumberAttribute();
		// Specifying the number of connected points, all
		boolean[] cp = new boolean[curve.numInstances()];
		for (int i = 0; i < cp.length; i++) {
			cp[i] = true;
		}
		try {
			plotData.setConnectPoints(cp);
			tvp.addPlot(plotData);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Display curve
		final JFrame jf = new JFrame("WEKA ROC: " + tvp.getName());
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(tvp, BorderLayout.CENTER);
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setVisible(true);

		System.out.println("...Finish!");
	}

	@Override
	public void printClassifierOutput(Instances data, Classifier cl,
			Evaluation eval) throws Exception {
		System.out.println("=== Instances summary (full training set) ===\n");
		System.out.println(data.toSummaryString());
		System.out.println("=== Classifier model (full training set) ===\n");
		System.out.println(cl + "\n");
		System.out.println("=== Stratified cross-validation ===");
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
	}

	@Override
	public void printInstancesData(Instances data) {
		System.out.println("=== Instances ===");
		System.out.println(data.toString() + "\n");
	}

}
