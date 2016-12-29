package org.trofiv;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

@SuppressWarnings({"CloneableClassWithoutClone", "WeakerAccess"})
public class BoostingAdditiveRegressionClassifier extends IteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, ImprovedClassifier {
    /**
     * Shrinkage (Learning rate). Default = no shrinkage.
     */
    protected static final double M_SHRINKAGE = 1.0;
    /**
     * The number of successfully generated base classifiers.
     */
    protected int m_NumIterationsPerformed;
    /**
     * The model for the mean
     */
    protected ZeroR m_zeroR;
    /**
     * whether we have suitable data or nor (if not, ZeroR model is used)
     */
    protected boolean m_SuitableData = true;
    /**
     * Test data instances
     */
    private Instances m_TestData;

    /**
     * Constructor which takes base classifier as argument.
     *
     * @param classifier the base classifier to use
     */
    public BoostingAdditiveRegressionClassifier(final Classifier classifier) {
        m_Classifier = classifier;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        final Capabilities result = super.getCapabilities();

        // class
        result.disableAllClasses();
        result.disableAllClassDependencies();
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);

        return result;
    }

    /**
     * Build the classifier on the supplied data
     *
     * @param data the training data
     * @throws Exception if the classifier could not be built successfully
     */
    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public void buildClassifier(final Instances data) throws Exception {
        super.buildClassifier(data);

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        Instances newData = new Instances(data);
        newData.deleteWithMissingClass();

        // Add the model for the mean first
        m_zeroR = new ZeroR();
        m_zeroR.buildClassifier(newData);

        // only class? -> use only ZeroR model
        if (newData.numAttributes() == 1) {
            System.err.println("Cannot build model (only class attribute present in data!), "
                    + "using ZeroR model instead!");
            m_SuitableData = false;
            return;
        }

        m_SuitableData = true;

        newData = residualReplace(newData, m_zeroR, false);
        double sum = 0;

        for (int i = 0; i < newData.numInstances(); i++) {
            sum += newData.instance(i).weight() *
                    newData.instance(i).classValue() * newData.instance(i).classValue();
        }

        m_NumIterationsPerformed = 0;
        double temp_sum;

        do {
            temp_sum = sum;

            // Build the classifier
            m_Classifiers[m_NumIterationsPerformed].buildClassifier(newData);

            newData = residualReplace(newData, m_Classifiers[m_NumIterationsPerformed], true);
            sum = 0;

            for (int i = 0; i < newData.numInstances(); i++) {
                sum += newData.instance(i).weight() *
                        newData.instance(i).classValue() * newData.instance(i).classValue();
            }

            m_NumIterationsPerformed++;
        } while (temp_sum - sum > weka.core.Utils.SMALL && m_NumIterationsPerformed < m_Classifiers.length);
    }

    /**
     * Classify an instance.
     *
     * @param inst the instance to predict
     * @return a prediction for the instance
     * @throws Exception if an error occurs
     */
    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public double classifyInstance(final Instance inst) throws Exception {
        double prediction = m_zeroR.classifyInstance(inst);

        if (!m_SuitableData) {
            return prediction;
        }

        for (int i = 0; i < m_NumIterationsPerformed; i++) {
            double toAdd = m_Classifiers[i].classifyInstance(inst);
            toAdd *= M_SHRINKAGE;
            prediction += toAdd;
        }

        return prediction;
    }

    /**
     * Replace the class values of the instances from the current iteration
     * with residuals after predicting with the supplied classifier.
     *
     * @param data         the instances to predict
     * @param c            the classifier to use
     * @param useShrinkage whether shrinkage is to be applied to the model's output
     * @return a new set of instances with class values replaced by residuals
     * @throws Exception if something goes wrong
     */
    @SuppressWarnings("ProhibitedExceptionDeclared")
    private static Instances residualReplace(
            final Instances data,
            final Classifier c,
            final boolean useShrinkage) throws Exception {
        final Instances newInst = new Instances(data);

        for (int i = 0; i < newInst.numInstances(); i++) {
            double pred = c.classifyInstance(newInst.instance(i));
            if (useShrinkage) {
                pred *= M_SHRINKAGE;
            }
            final double residual = newInst.instance(i).classValue() - pred;
            newInst.instance(i).setClassValue(residual);
        }

        return newInst;
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
        final List<List<Double>> merged = new LinkedList<>();

        for (int i = 0; i < m_TestData.numAttributes(); i++) {
            merged.add(new LinkedList<>());
        }

        for (int i = 0; i < m_Classifiers.length && i < m_NumIterationsPerformed; i++) {
            final RegressionTree classifier = (RegressionTree) m_Classifiers[i];
            final Map<Integer, List<Double>> sse = classifier.sse();

            sse.entrySet().stream().forEach(entry -> {
                final List<Double> arr = entry.getValue();
                merged.get(entry.getKey()).addAll(arr);
            });
        }

        return merged.stream()
                .map(i -> i.stream().mapToDouble(d -> d).sum())
                .mapToDouble(i -> i)
                .toArray();
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
