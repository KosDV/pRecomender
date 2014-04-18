package examples;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NewInstancesRoc {
	public static void main(String[] args) throws Exception {
		// load dataset
		Instances data = DataSource.read("/Users/nadim/Workspace/weka-3-7-10/data/weather.nominal.arff");
		// Constructor creating an empty set of instances. Copies references to
		// the header information from the given set of instances.
		Instances newData = null;
		newData = new Instances(data, 1);
		// Create empty instance with 5 attribute values
		Instance inst = new DenseInstance(5);

		// Set instance's dataset to be the dataset "race"
		inst.setDataset(newData);

		// Set instance's values for the attributes "length", "weight", and
		// "position"
		// @attribute outlook {sunny, overcast, rainy}
		// @attribute temperature {hot, mild, cool}
		// @attribute humidity {high, normal}
		// @attribute windy {TRUE, FALSE}
		// @attribute play {yes, no}

		inst.setValue(0, "sunny");
		inst.setValue(1, "hot");
		inst.setValue(2, "high");
		inst.setValue(3, "TRUE");

		// Print the instance
		System.out.println("The instance: " + inst);
		// output on stdout
		System.out.println(data);
		System.out.println(newData);
	}
}