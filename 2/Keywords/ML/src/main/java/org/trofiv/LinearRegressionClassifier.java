package org.trofiv;

import org.apache.commons.lang.ArrayUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

@SuppressWarnings("CloneableClassWithoutClone")
public class LinearRegressionClassifier extends Classifier implements
        WeightedInstancesHandler, ImprovedClassifier {
    /**
     * Array for storing coefficients of linear regression.
     */
    private double[] m_Coefficients;
    /**
     * Which attributes are relevant?
     */
    private boolean[] m_SelectedAttributes;
    /**
     * Variable for storing transformed training data.
     */
    private Instances m_TransformedData;
    /**
     * The filter for removing missing values.
     */
    private ReplaceMissingValues m_MissingFilter;
    /**
     * The filter storing the transformation from nominal to
     * binary attributes.
     */
    private NominalToBinary m_TransformFilter;
    /**
     * The mean of the class attribute
     */
    private double m_ClassMean;
    /**
     * The index of the class attribute
     */
    private int m_ClassIndex;
    /**
     * The attributes means
     */
    private double[] m_Means;
    /**
     * The attribute standard deviations
     */
    private double[] m_StdDevs;
    /**
     * Test data instances
     */
    private Instances m_TestData;

    /**
     * Builds a regression model for the given data.
     *
     * @param data the training data to be used for generating the
     *             linear regression function
     * @throws Exception if the classifier could not be built successfully
     */
    @Override
    @SuppressWarnings({"ProhibitedExceptionDeclared", "OverlyLongMethod"})
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        // Preprocessed instances
        m_TransformFilter = new NominalToBinary();
        m_TransformFilter.setInputFormat(data);
        data = Filter.useFilter(data, m_TransformFilter);
        m_MissingFilter = new ReplaceMissingValues();
        m_MissingFilter.setInputFormat(data);
        data = Filter.useFilter(data, m_MissingFilter);
        data.deleteWithMissingClass();

        m_ClassIndex = data.classIndex();
        m_TransformedData = data;

        // Turn all attributes on for a start
        m_SelectedAttributes = new boolean[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != m_ClassIndex) {
                m_SelectedAttributes[i] = true;
            }
        }

        //noinspection AssignmentToNull
        m_Coefficients = null;

        // Compute means and standard deviations
        m_Means = new double[data.numAttributes()];
        m_StdDevs = new double[data.numAttributes()];
        for (int j = 0; j < data.numAttributes(); j++) {
            if (j != data.classIndex()) {
                m_Means[j] = data.meanOrMode(j);
                m_StdDevs[j] = Math.sqrt(data.variance(j));
                if (m_StdDevs[j] == 0) {
                    m_SelectedAttributes[j] = false;
                }
            }
        }

        m_ClassMean = data.meanOrMode(m_TransformedData.classIndex());

        // Perform the regression
        findBestModel();

        // Save memory
        m_TransformedData = new Instances(data);
    }

    /**
     * Classifies the given instance using the linear regression function.
     *
     * @param instance the test instance
     * @return the classification
     * @throws Exception if classification can't be done successfully
     */
    @SuppressWarnings("ProhibitedExceptionDeclared")
    @Override
    public double classifyInstance(final Instance instance) throws Exception {
        // Transform the input instance
        m_TransformFilter.input(instance);
        m_TransformFilter.batchFinished();
        final Instance output = m_TransformFilter.output();

        m_MissingFilter.input(output);
        m_MissingFilter.batchFinished();
        final Instance newOutput = m_MissingFilter.output();

        // Calculate the dependent variable from the regression model
        return regressionPrediction(newOutput, m_SelectedAttributes, m_Coefficients);
    }

    /**
     * Performs a greedy search for the best regression model using
     * Akaike's criterion.
     */
    private void findBestModel() {
        m_Coefficients = doRegression(m_SelectedAttributes);
    }

    /**
     * Calculate the squared error of a regression model on the
     * training data
     *
     * @param selectedAttributes an array of flags indicating which
     *                           attributes are included in the regression model
     * @param coefficients       an array of coefficients for the regression
     *                           model
     * @return the mean squared error on the training data
     */
    private double calculateSE(
            final boolean[] selectedAttributes,
            final double[] coefficients) {
        double mse = 0;

        for (int i = 0; i < m_TransformedData.numInstances(); i++) {
            final double prediction = regressionPrediction(m_TransformedData.instance(i), selectedAttributes, coefficients);
            final double error = prediction - m_TransformedData.instance(i).classValue();
            mse += error * error;
        }

        return mse;
    }

    /**
     * Calculate the dependent value for a given instance for a
     * given regression model.
     *
     * @param transformedInstance the input instance
     * @param selectedAttributes  an array of flags indicating which
     *                            attributes are included in the regression model
     * @param coefficients        an array of coefficients for the regression
     *                            model
     * @return the regression value for the instance.
     */
    private double regressionPrediction(
            final Instance transformedInstance,
            final boolean[] selectedAttributes,
            final double[] coefficients) {
        double result = 0;
        int column = 0;

        for (int j = 0; j < transformedInstance.numAttributes(); j++) {
            if (m_ClassIndex != j && selectedAttributes[j]) {
                result += coefficients[column] * transformedInstance.value(j);
                column++;
            }
        }

        result += coefficients[column];

        return result;
    }

    /**
     * Calculate a linear regression using the selected attributes
     *
     * @param selectedAttributes an array of booleans where each element
     *                           is true if the corresponding attribute should be included in the
     *                           regression.
     * @return an array of coefficients for the linear regression model.
     */
    @SuppressWarnings({"OverlyComplexMethod", "OverlyLongMethod"})
    private double[] doRegression(final boolean[] selectedAttributes) {
        int numAttributes = 0;

        for (boolean selectedAttribute : selectedAttributes) {
            if (selectedAttribute) {
                numAttributes++;
            }
        }

        // Check whether there are still attributes left
        Matrix independent = null, dependent = null;

        if (numAttributes > 0) {
            independent = new Matrix(m_TransformedData.numInstances(), numAttributes);
            dependent = new Matrix(m_TransformedData.numInstances(), 1);

            for (int i = 0; i < m_TransformedData.numInstances(); i++) {
                final Instance inst = m_TransformedData.instance(i);
                final double sqrt_weight = Math.sqrt(inst.weight());
                int column = 0;

                for (int j = 0; j < m_TransformedData.numAttributes(); j++) {
                    if (j == m_ClassIndex) {
                        dependent.set(i, 0, inst.classValue() * sqrt_weight);
                    } else {
                        if (selectedAttributes[j]) {
                            double value = inst.value(j) - m_Means[j];

                            // We only need to do this if we want to
                            // scale the input
                            value /= m_StdDevs[j];

                            independent.set(i, column, value * sqrt_weight);
                            column++;
                        }
                    }
                }
            }
        }

        // Compute coefficients (note that we have to treat the
        // intercept separately so that it doesn't get affected
        // by the ridge constant.)
        final double[] coefficients = new double[numAttributes + 1];

        if (numAttributes > 0) {
            final double m_Ridge = 1.0e-8;
            //noinspection ConstantConditions
            final double[] coeffsWithoutIntercept = independent.regression(dependent, m_Ridge).getCoefficients();
            System.arraycopy(coeffsWithoutIntercept, 0, coefficients, 0, numAttributes);
        }

        coefficients[numAttributes] = m_ClassMean;

        // Convert coefficients into original scale
        int column = 0;
        for (int i = 0; i < m_TransformedData.numAttributes(); i++) {
            if (i != m_TransformedData.classIndex() && selectedAttributes[i]) {
                coefficients[column] /= m_StdDevs[i];
                coefficients[coefficients.length - 1] -= coefficients[column] * m_Means[i];
                column++;
            }
        }

        return coefficients;
    }

    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public void evaluate(final Instances train, final Instances test) throws Exception {
        final Evaluation evaluation = new Evaluation(train);
        this.buildClassifier(train);
        evaluation.evaluateModel(this, test);
        m_TestData = new Instances(test);
    }

    @Override
    public double[] attributeQuality() {
        final double[] result = new double[m_SelectedAttributes.length];

        for (int i = 0; i < m_SelectedAttributes.length; i++) {
            if (i != m_ClassIndex) {
                Arrays.fill(m_SelectedAttributes, true);
                m_SelectedAttributes[i] = false;

                result[i] = calculateSE(m_SelectedAttributes, m_Coefficients);
            }
        }

        return result;
    }

    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public double[] ncg() throws Exception {
        final List<Double> originalUtility = new LinkedList<>();
        final List<Double> predictedUtility = new LinkedList<>();

        for (int i = 0; i < m_TestData.numInstances(); i++) {
            originalUtility.add(m_TestData.instance(i).classValue());
            predictedUtility.add(this.classifyInstance(m_TestData.instance(i)));
        }

        return Utils.ncg(predictedUtility, originalUtility).stream().mapToDouble(r -> r).toArray();
    }
}
