package org.trofiv;

import org.apache.commons.lang.ArrayUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.rules.ZeroR;
import weka.core.*;
import weka.core.Utils;

import java.io.Serializable;
import java.util.*;

@SuppressWarnings({"CloneableClassWithoutClone", "WeakerAccess"})
public class RegressionDecisionTree extends Classifier
        implements WeightedInstancesHandler, Sourcable, RegressionTree {
    /**
     * Number of folds for reduced error pruning.
     */
    protected static final int M_NUM_FOLDS = 3;
    /**
     * Seed for random data shuffling.
     */
    protected static final int M_SEED = 1;
    /**
     * The minimum number of instances per leaf.
     */
    protected static final double M_MIN_NUM = 2;
    /**
     * The minimum proportion of the total variance (over all the data) required for split.
     */
    protected static final double M_MIN_VARIANCE_PROP = 1.0e-3;
    /**
     * Upper bound on the tree depth
     */
    protected static final int M_MAX_DEPTH = -1;
    private static final String NO_MODEL_BUILT_YET = "REPTree: No model built yet.";
    /**
     * ZeroR model that is used if no attributes are present.
     */
    protected ZeroR m_zeroR;
    /**
     * The Tree object
     */
    protected Tree m_Tree;
    /**
     * Don't prune
     */
    @SuppressWarnings("unused")
    protected boolean m_NoPruning;

    private static void traverseTree(final Tree tree, final Map<Integer, List<Double>> map) {
        for (Object subTreeBase : ArrayUtils.nullToEmpty(tree.m_Successors)) {
            final Tree subTree = (Tree) subTreeBase;
            traverseTree(subTree, map);
        }

        if (tree.m_Attribute != -1) {
            if (map.containsKey(tree.m_Attribute)) {
                final List<Double> list = map.get(tree.m_Attribute);
                list.add(tree.m_HoldOutError);
            } else {
                final List<Double> list = new LinkedList<>();
                list.add(tree.m_HoldOutError);
                map.put(tree.m_Attribute, list);
            }
        }
    }

    /**
     * Builds classifier.
     *
     * @param data the data to train with
     * @throws Exception if building fails
     */
    @Override
    @SuppressWarnings({"ProhibitedExceptionDeclared", "OverlyComplexMethod", "OverlyLongMethod"})
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        //noinspection UnsecureRandomNumberGeneration
        final Random random = new Random(M_SEED);

        //noinspection AssignmentToNull
        m_zeroR = null;
        if (data.numAttributes() == 1) {
            m_zeroR = new ZeroR();
            m_zeroR.buildClassifier(data);
            return;
        }

        // Randomize and stratify
        data.randomize(random);
        if (data.classAttribute().isNominal()) {
            data.stratify(M_NUM_FOLDS);
        }

        // Split data into training and pruning set
        final Instances train;
        Instances prune = null;

        if (m_NoPruning) {
            train = data;
        } else {
            train = data.trainCV(M_NUM_FOLDS, 0, random);
            prune = data.testCV(M_NUM_FOLDS, 0);
        }

        // Create array of sorted indices and weights
        final int[][][] sortedIndices = new int[1][train.numAttributes()][0];
        final double[][][] weights = new double[1][train.numAttributes()][0];
        final double[] vals = new double[train.numInstances()];

        for (int j = 0; j < train.numAttributes(); j++) {
            if (j != train.classIndex()) {
                weights[0][j] = new double[train.numInstances()];
                if (train.attribute(j).isNominal()) {
                    // Handling nominal attributes. Putting indices of
                    // instances with missing values at the end.
                    sortedIndices[0][j] = new int[train.numInstances()];
                    int count = 0;
                    for (int i = 0; i < train.numInstances(); i++) {
                        final Instance inst = train.instance(i);
                        if (!inst.isMissing(j)) {
                            sortedIndices[0][j][count] = i;
                            weights[0][j][count] = inst.weight();
                            count++;
                        }
                    }
                    for (int i = 0; i < train.numInstances(); i++) {
                        final Instance inst = train.instance(i);
                        if (inst.isMissing(j)) {
                            sortedIndices[0][j][count] = i;
                            weights[0][j][count] = inst.weight();
                            count++;
                        }
                    }
                } else {
                    // Sorted indices are computed for numeric attributes
                    for (int i = 0; i < train.numInstances(); i++) {
                        final Instance inst = train.instance(i);
                        vals[i] = inst.value(j);
                    }
                    sortedIndices[0][j] = Utils.sort(vals);
                    for (int i = 0; i < train.numInstances(); i++) {
                        weights[0][j][i] = train.instance(sortedIndices[0][j][i]).weight();
                    }
                }
            }
        }

        // Compute initial class counts
        final double[] classProbs = new double[train.numClasses()];
        double totalWeight = 0, totalSumSquared = 0;

        for (int i = 0; i < train.numInstances(); i++) {
            final Instance inst = train.instance(i);
            if (data.classAttribute().isNominal()) {
                //noinspection NumericCastThatLosesPrecision
                classProbs[(int) inst.classValue()] += inst.weight();
                totalWeight += inst.weight();
            } else {
                classProbs[0] += inst.classValue() * inst.weight();
                totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
                totalWeight += inst.weight();
            }
        }

        m_Tree = new Tree();
        double trainVariance = 0;

        if (data.classAttribute().isNumeric()) {
            trainVariance = Tree.
                    singleVariance(classProbs[0], totalSumSquared, totalWeight) / totalWeight;
            classProbs[0] /= totalWeight;
        }

        // Build tree
        m_Tree.buildTree(sortedIndices, weights, train, totalWeight, classProbs,
                new Instances(train, 0), M_MIN_NUM, M_MIN_VARIANCE_PROP *
                        trainVariance, 0, M_MAX_DEPTH);

        // Insert pruning data and perform reduced error pruning
        if (!m_NoPruning) {
            m_Tree.insertHoldOutSet(prune);
            m_Tree.reducedErrorPrune();
            m_Tree.backfitHoldOutSet();
        }
    }

    /**
     * Computes class distribution of an instance using the tree.
     *
     * @param instance the instance to compute the distribution for
     * @return the computed class probabilities
     * @throws Exception if computation fails
     */
    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public double[] distributionForInstance(final Instance instance) throws Exception {
        return m_zeroR != null ? m_zeroR.distributionForInstance(instance) : m_Tree.distributionForInstance(instance);
    }

    /**
     * Returns the tree as if-then statements.
     *
     * @param className the name for the generated class
     * @return the tree as a Java if-then type statement
     * @throws Exception if something goes wrong
     */
    @Override
    @SuppressWarnings("ProhibitedExceptionDeclared")
    public String toSource(final String className) throws Exception {
        if (m_Tree == null) {
            throw new IllegalStateException(NO_MODEL_BUILT_YET);
        }

        final StringBuilder[] source = m_Tree.toSource(className, m_Tree);
        return "class " + className + " {\n\n"
                + "  public static double classify(Object [] i)\n"
                + "    throws Exception {\n\n"
                + "    double p = Double.NaN;\n"
                + source[0]  // Assignment code
                + "    return p;\n"
                + "  }\n"
                + source[1]  // Support code
                + "}\n";
    }

    /**
     * Return a map of (lists of SSE values for each node) for each attribute
     *
     * @return map
     */
    @Override
    public Map<Integer, List<Double>> sse() {
        final Map<Integer, List<Double>> map = new HashMap<>(14, 1.0f);
        traverseTree(this.m_Tree, map);
        return map;
    }

    /**
     * An inner class for building and storing the tree structure
     */
    @SuppressWarnings({"OverlyComplexClass", "JavaDoc",
            "NumericCastThatLosesPrecision", "AssignmentToNull",
            "FieldNotUsedInToString"})
    protected static class Tree implements Serializable {
        /**
         * The header information (for printing the tree).
         */
        protected Instances m_Info;

        /**
         * The subtrees of this tree.
         */
        protected Tree[] m_Successors;

        /**
         * The attribute to split on.
         */
        protected int m_Attribute = -1;

        /**
         * The split point.
         */
        protected double m_SplitPoint = Double.NaN;

        /**
         * The proportions of training instances going down each branch.
         */
        protected double[] m_Prop;

        /**
         * Class probabilities from the training data in the nominal case. Holds the mean in the numeric case.
         */
        protected double[] m_ClassProbs;

        /**
         * The (unnormalized) class distribution in the nominal case. Holds the sum of squared errors and the weight in
         * the numeric case.
         */
        protected double[] m_Distribution;

        /**
         * Class distribution of hold-out set at node in the nominal case.  Straight sum of weights plus sum of weighted
         * targets in the numeric case (i.e. array has only two elements).
         */
        protected double[] m_HoldOutDist;

        /**
         * The hold-out error of the node. The number of miss-classified instances in the nominal case, the sum of
         * squared errors in the numeric case.
         */
        protected double m_HoldOutError;

        /**
         * Computes class distribution for an attribute.
         *
         * @param props
         * @param dists
         * @param att           the attribute index
         * @param sortedIndices the sorted indices of the instances
         * @param weights       the weights of the instances
         * @param subsetWeights the weights of the subset
         * @param data          the data to work with
         * @return the split point
         * @throws Exception if computation fails
         */
        @SuppressWarnings({"NumericCastThatLosesPrecision", "OverlyComplexMethod", "OverlyLongMethod"})
        protected static double distribution(
                final double[][] props,
                final double[][][] dists,
                final int att,
                final int[] sortedIndices,
                final double[] weights,
                final double[][] subsetWeights,
                final Instances data) {

            double splitPoint = Double.NaN;
            final Attribute attribute = data.attribute(att);
            final double[][] dist;
            int i;

            if (attribute.isNominal()) {
                // For nominal attributes
                dist = new double[attribute.numValues()][data.numClasses()];
                for (i = 0; i < sortedIndices.length; i++) {
                    final Instance inst = data.instance(sortedIndices[i]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    dist[(int) inst.value(att)][(int) inst.classValue()] += weights[i];
                }
            } else {

                // For numeric attributes
                final double[][] currDist = new double[2][data.numClasses()];
                dist = new double[2][data.numClasses()];

                // Move all instances into second subset
                for (int j = 0; j < sortedIndices.length; j++) {
                    final Instance inst = data.instance(sortedIndices[j]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    currDist[1][(int) inst.classValue()] += weights[j];
                }
                final double priorVal = priorVal(currDist);
                System.arraycopy(currDist[1], 0, dist[1], 0, dist[1].length);

                // Try all possible split points
                double currSplit = data.instance(sortedIndices[0]).value(att);
                double bestVal = -Double.MAX_VALUE;
                for (i = 0; i < sortedIndices.length; i++) {
                    final Instance inst = data.instance(sortedIndices[i]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    if (inst.value(att) > currSplit) {
                        final double currVal = gain(currDist, priorVal);
                        if (currVal > bestVal) {
                            bestVal = currVal;
                            //noinspection MagicNumber
                            splitPoint = (inst.value(att) + currSplit) / 2.0;

                            // Check for numeric precision problems
                            if (splitPoint <= currSplit) {
                                splitPoint = inst.value(att);
                            }

                            for (int j = 0; j < currDist.length; j++) {
                                System.arraycopy(currDist[j], 0, dist[j], 0,
                                        dist[j].length);
                            }
                        }
                    }
                    currSplit = inst.value(att);
                    currDist[0][(int) inst.classValue()] += weights[i];
                    currDist[1][(int) inst.classValue()] -= weights[i];
                }
            }

            // Compute weights
            props[att] = new double[dist.length];
            for (int k = 0; k < props[att].length; k++) {
                props[att][k] = Utils.sum(dist[k]);
            }
            if (Utils.sum(props[att]) > 0) {
                Utils.normalize(props[att]);
            } else {
                for (int k = 0; k < props[att].length; k++) {
                    props[att][k] = 1.0 / props[att].length;
                }
            }

            // Distribute counts
            while (i < sortedIndices.length) {
                final Instance inst = data.instance(sortedIndices[i]);
                for (int j = 0; j < dist.length; j++) {
                    dist[j][(int) inst.classValue()] += props[att][j] * weights[i];
                }
                i++;
            }

            // Compute subset weights
            subsetWeights[att] = new double[dist.length];
            for (int j = 0; j < dist.length; j++) {
                subsetWeights[att][j] += Utils.sum(dist[j]);
            }

            // Return distribution and split point
            dists[att] = dist;
            return splitPoint;
        }

        /**
         * Computes class distribution for an attribute.
         *
         * @param props
         * @param dists
         * @param att           the attribute index
         * @param sortedIndices the sorted indices of the instances
         * @param weights       the weights of the instances
         * @param subsetWeights the weights of the subset
         * @param data          the data to work with
         * @param vals
         * @return the split point
         * @throws Exception if computation fails
         */
        @SuppressWarnings({"OverlyComplexMethod", "OverlyLongMethod"})
        protected static double numericDistribution(
                final double[][] props,
                final double[][][] dists,
                final int att,
                final int[] sortedIndices,
                final double[] weights,
                final double[][] subsetWeights,
                final Instances data,
                final double[] vals) {

            double splitPoint = Double.NaN;
            final Attribute attribute = data.attribute(att);
            final double[] sums;
            final double[] sumSquared;
            final double[] sumOfWeights;
            double totalSum, totalSumSquared, totalSumOfWeights;

            int i;

            if (attribute.isNominal()) {

                // For nominal attributes
                sums = new double[attribute.numValues()];
                sumSquared = new double[attribute.numValues()];
                sumOfWeights = new double[attribute.numValues()];
                for (i = 0; i < sortedIndices.length; i++) {
                    final Instance inst = data.instance(sortedIndices[i]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    //noinspection NumericCastThatLosesPrecision
                    final int attVal = (int) inst.value(att);
                    sums[attVal] += inst.classValue() * weights[i];
                    sumSquared[attVal] +=
                            inst.classValue() * inst.classValue() * weights[i];
                    sumOfWeights[attVal] += weights[i];
                }
                totalSum = Utils.sum(sums);
                totalSumSquared = Utils.sum(sumSquared);
                totalSumOfWeights = Utils.sum(sumOfWeights);
            } else {

                // For numeric attributes
                sums = new double[2];
                sumSquared = new double[2];
                sumOfWeights = new double[2];
                final double[] currSums = new double[2];
                final double[] currSumSquared = new double[2];
                final double[] currSumOfWeights = new double[2];

                // Move all instances into second subset
                for (int j = 0; j < sortedIndices.length; j++) {
                    final Instance inst = data.instance(sortedIndices[j]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    currSums[1] += inst.classValue() * weights[j];
                    currSumSquared[1] +=
                            inst.classValue() * inst.classValue() * weights[j];
                    currSumOfWeights[1] += weights[j];

                }
                totalSum = currSums[1];
                totalSumSquared = currSumSquared[1];
                totalSumOfWeights = currSumOfWeights[1];

                sums[1] = currSums[1];
                sumSquared[1] = currSumSquared[1];
                sumOfWeights[1] = currSumOfWeights[1];

                // Try all possible split points
                double currSplit = data.instance(sortedIndices[0]).value(att);
                double bestVal = Double.MAX_VALUE;
                for (i = 0; i < sortedIndices.length; i++) {
                    final Instance inst = data.instance(sortedIndices[i]);
                    if (inst.isMissing(att)) {
                        break;
                    }
                    if (inst.value(att) > currSplit) {
                        final double currVal = variance(currSums, currSumSquared, currSumOfWeights);
                        if (currVal < bestVal) {
                            bestVal = currVal;
                            //noinspection MagicNumber
                            splitPoint = (inst.value(att) + currSplit) / 2.0;

                            // Check for numeric precision problems
                            if (splitPoint <= currSplit) {
                                splitPoint = inst.value(att);
                            }

                            for (int j = 0; j < 2; j++) {
                                sums[j] = currSums[j];
                                sumSquared[j] = currSumSquared[j];
                                sumOfWeights[j] = currSumOfWeights[j];
                            }
                        }
                    }

                    currSplit = inst.value(att);

                    final double classVal = inst.classValue() * weights[i];
                    final double classValSquared = inst.classValue() * classVal;

                    currSums[0] += classVal;
                    currSumSquared[0] += classValSquared;
                    currSumOfWeights[0] += weights[i];

                    currSums[1] -= classVal;
                    currSumSquared[1] -= classValSquared;
                    currSumOfWeights[1] -= weights[i];
                }
            }

            // Compute weights
            props[att] = new double[sums.length];
            System.arraycopy(sumOfWeights, 0, props[att], 0, props[att].length);
            if (Utils.sum(props[att]) > 0) {
                Utils.normalize(props[att]);
            } else {
                for (int k = 0; k < props[att].length; k++) {
                    props[att][k] = 1.0 / props[att].length;
                }
            }


            // Distribute counts for missing values
            while (i < sortedIndices.length) {
                final Instance inst = data.instance(sortedIndices[i]);
                for (int j = 0; j < sums.length; j++) {
                    sums[j] += props[att][j] * inst.classValue() * weights[i];
                    sumSquared[j] += props[att][j] * inst.classValue() *
                            inst.classValue() * weights[i];
                    sumOfWeights[j] += props[att][j] * weights[i];
                }
                totalSum += inst.classValue() * weights[i];
                totalSumSquared +=
                        inst.classValue() * inst.classValue() * weights[i];
                totalSumOfWeights += weights[i];
                i++;
            }

            // Compute final distribution
            final double[][] dist = new double[sums.length][data.numClasses()];
            for (int j = 0; j < sums.length; j++) {
                dist[j][0] = sumOfWeights[j] > 0 ? sums[j] / sumOfWeights[j] : totalSum / totalSumOfWeights;
            }

            // Compute variance gain
            final double priorVar = singleVariance(totalSum, totalSumSquared, totalSumOfWeights);
            final double variable = variance(sums, sumSquared, sumOfWeights);
            final double gain = priorVar - variable;

            // Return distribution and split point
            subsetWeights[att] = sumOfWeights;
            dists[att] = dist;
            vals[att] = gain;
            return splitPoint;
        }

        /**
         * Computes variance for subsets.
         *
         * @param s
         * @param sS
         * @param sumOfWeights
         * @return the variance
         */
        protected static double variance(
                final double[] s,
                final double[] sS,
                final double[] sumOfWeights) {
            double variable = 0;

            for (int i = 0; i < s.length; i++) {
                if (sumOfWeights[i] > 0) {
                    variable += singleVariance(s[i], sS[i], sumOfWeights[i]);
                }
            }

            return variable;
        }

        /**
         * Computes the variance for a single set
         *
         * @param s
         * @param sS
         * @param weight the weight
         * @return the variance
         */
        protected static double singleVariance(final double s, final double sS, final double weight) {
            return sS - s * s / weight;
        }

        /**
         * Computes value of splitting criterion before split.
         *
         * @param dist
         * @return the splitting criterion
         */
        protected static double priorVal(final double[][] dist) {
            return ContingencyTables.entropyOverColumns(dist);
        }

        /**
         * Computes value of splitting criterion after split.
         *
         * @param dist
         * @param priorVal the splitting criterion
         * @return the gain after splitting
         */
        protected static double gain(final double[][] dist, final double priorVal) {
            return priorVal - ContingencyTables.entropyConditionedOnRows(dist);
        }

        /**
         * Recursively outputs the tree.
         *
         * @return the generated subtree
         */
        public String toString() {
            return toString(0, null);
        }

        /**
         * Recursively outputs the tree.
         *
         * @param level  the current level
         * @param parent the current parent
         * @return the generated subtree
         */
        @SuppressWarnings("OverlyLongMethod")
        protected String toString(final int level, final Tree parent) {
            try {
                final StringBuilder text = new StringBuilder(500);

                if (m_Attribute == -1) {
                    // Output leaf info
                    return leafString(parent);
                }
                if (m_Info.attribute(m_Attribute).isNominal()) {

                    // For nominal attributes
                    for (int i = 0; i < m_Successors.length; i++) {
                        text.append('\n');
                        for (int j = 0; j < level; j++) {
                            text.append("|   ");
                        }
                        text.append(m_Info.attribute(m_Attribute).name()).append(" = ").append(m_Info.attribute(m_Attribute).value(i));
                        text.append(m_Successors[i].toString(level + 1, this));
                    }
                } else {

                    // For numeric attributes
                    text.append('\n');
                    for (int j = 0; j < level; j++) {
                        text.append("|   ");
                    }
                    text.append(m_Info.attribute(m_Attribute).name()).append(" < ").append(Utils.doubleToString(m_SplitPoint, 2));
                    text.append(m_Successors[0].toString(level + 1, this));
                    text.append('\n');
                    for (int j = 0; j < level; j++) {
                        text.append("|   ");
                    }
                    text.append(m_Info.attribute(m_Attribute).name()).append(" >= ").append(Utils.doubleToString(m_SplitPoint, 2));
                    text.append(m_Successors[1].toString(level + 1, this));
                }

                return text.toString();
            } catch (Exception e) {
                e.printStackTrace();
                return "Decision tree: tree can't be printed";
            }
        }

        /**
         * Computes class distribution of an instance using the tree.
         *
         * @param instance the instance to compute the distribution for
         * @return the distribution
         * @throws Exception if computation fails
         */
        @SuppressWarnings("OverlyComplexMethod")
        protected double[] distributionForInstance(final Instance instance) {
            double[] returnedDist = null;

            if (m_Attribute > -1) {

                // Node is not a leaf
                if (instance.isMissing(m_Attribute)) {

                    // Value is missing
                    returnedDist = new double[m_Info.numClasses()];

                    // Split instance up
                    for (int i = 0; i < m_Successors.length; i++) {
                        final double[] help = m_Successors[i].distributionForInstance(instance);
                        if (help != null) {
                            for (int j = 0; j < help.length; j++) {
                                returnedDist[j] += m_Prop[i] * help[j];
                            }
                        }
                    }
                } else if (m_Info.attribute(m_Attribute).isNominal()) {
                    // For nominal attributes
                    //noinspection NumericCastThatLosesPrecision
                    returnedDist = m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
                } else {
                    // For numeric attributes
                    returnedDist = instance.value(m_Attribute) < m_SplitPoint ?
                            m_Successors[0].distributionForInstance(instance) :
                            m_Successors[1].distributionForInstance(instance);
                }
            }
            if (m_Attribute == -1 || returnedDist == null) {

                // Node is a leaf or successor is empty
                if (m_ClassProbs == null) {
                    return m_ClassProbs;
                }
                return m_ClassProbs.clone();
            } else {
                return returnedDist;
            }
        }

        /**
         * Returns a string containing java source code equivalent to the test made at this node. The instance being
         * tested is called "i". This routine assumes to be called in the order of branching, enabling us to set the >=
         * condition test (the last one) of a numeric splitpoint to just "true" (because being there in the flow implies
         * that the previous less-than test failed).
         *
         * @param index index of the value tested
         * @return a value of type 'String'
         */
        public final String sourceExpression(final int index) {
            if (index < 0) {
                return "i[" + m_Attribute + "] == null";
            }
            final StringBuilder expr;
            if (m_Info.attribute(m_Attribute).isNominal()) {
                expr = new StringBuilder("i[");
                expr.append(m_Attribute).append(']');
                expr.append(".equals(\"").append(m_Info.attribute(m_Attribute)
                        .value(index)).append("\")");
            } else {
                expr = new StringBuilder("");
                if (index == 0) {
                    expr.append("((Double)i[")
                            .append(m_Attribute).append("]).doubleValue() < ")
                            .append(m_SplitPoint);
                } else {
                    expr.append("true");
                }
            }
            return expr.toString();
        }

        /**
         * Returns source code for the tree as if-then statements. The class is assigned to variable "p", and assumes
         * the tested instance is named "i". The results are returned as two StringBuilders: a section of code for
         * assignment of the class, and a section of code containing support code (eg: other support methods).
         * <p>
         * TODO: If the outputted source code encounters a missing value for the evaluated attribute, it stops branching
         * and uses the class distribution of the current node to decide the return value. This is unlike the behaviour
         * of distributionForInstance().
         *
         * @param className the classname that this static classifier has
         * @param parent    parent node of the current node
         * @return an array containing two StringBuilders, the first string containing assignment code, and the second
         * containing source for support code.
         * @throws Exception if something goes wrong
         */
        @SuppressWarnings("OverlyLongMethod")
        public StringBuilder[] toSource(final String className, final Tree parent) {

            final double[] currentProbs = m_ClassProbs == null ? parent.m_ClassProbs : m_ClassProbs;

            final long printID = UUID.randomUUID().getLeastSignificantBits();

            // Is this a leaf?
            final StringBuilder[] result = new StringBuilder[2];
            if (m_Attribute == -1) {
                result[0] = new StringBuilder("	p = ");
                if (m_Info.classAttribute().isNumeric()) {
                    result[0].append(currentProbs[0]);
                } else {
                    result[0].append(Utils.maxIndex(currentProbs));
                }
                result[0].append(";\n");
                result[1] = new StringBuilder("");
            } else {
                final StringBuilder text = new StringBuilder("");
                final StringBuilder atEnd = new StringBuilder("");

                text.append("  static double N").append(Integer.toHexString(this.hashCode()))
                        .append(printID)
                        .append("(Object []i) {\n")
                        .append("    double p = Double.NaN;\n");

                text.append("    /* ").append(m_Info.attribute(m_Attribute).name()).append(" */\n");
                // Missing attribute?
                text.append("    if (").append(this.sourceExpression(-1)).append(") {\n")
                        .append("      p = ");
                if (m_Info.classAttribute().isNumeric()) {
                    text.append(currentProbs[0]).append(";\n");
                } else {
                    text.append(Utils.maxIndex(currentProbs)).append(";\n");
                }
                text.append("    } ");

                // Branching of the tree
                for (int i = 0; i < m_Successors.length; i++) {
                    text.append("else if (").append(this.sourceExpression(i)).append(") {\n");
                    // Is the successor a leaf?
                    if (m_Successors[i].m_Attribute == -1) {
                        double[] successorProbs = m_Successors[i].m_ClassProbs;
                        if (successorProbs == null) {
                            successorProbs = m_ClassProbs;
                        }
                        text.append("      p = ");
                        if (m_Info.classAttribute().isNumeric()) {
                            text.append(successorProbs[0]).append(";\n");
                        } else {
                            text.append(Utils.maxIndex(successorProbs)).append(";\n");
                        }
                    } else {
                        final StringBuilder[] sub = m_Successors[i].toSource(className, this);
                        text.append("").append(sub[0]);
                        atEnd.append("").append(sub[1]);
                    }
                    text.append("    } ");
                    if (i == m_Successors.length - 1) {
                        text.append('\n');
                    }
                }

                text.append("    return p;\n  }\n");

                result[0] = new StringBuilder("    p = " + className + ".N");
                result[0].append(Integer.toHexString(this.hashCode())).append(printID).append("(i);\n");
                result[1] = text.append(atEnd);
            }
            return result;
        }

        /**
         * Outputs description of a leaf node.
         *
         * @param parent the parent of the node
         * @return the description of the node
         * @throws Exception if generation fails
         */
        @SuppressWarnings("ReuseOfLocalVariable")
        protected String leafString(final Tree parent) {
            if (m_Info.classAttribute().isNumeric()) {
                final double classMean = m_ClassProbs == null ? parent.m_ClassProbs[0] : m_ClassProbs[0];
                final StringBuilder buffer = new StringBuilder(100);
                buffer.append(" : ").append(Utils.doubleToString(classMean, 2));
                double avgError = 0;
                if (m_Distribution[1] > 0) {
                    avgError = m_Distribution[0] / m_Distribution[1];
                }
                buffer.append(" (")
                        .append(Utils.doubleToString(m_Distribution[1], 2))
                        .append('/')
                        .append(Utils.doubleToString(avgError, 2))
                        .append(')');
                avgError = 0;
                if (m_HoldOutDist[0] > 0) {
                    avgError = m_HoldOutError / m_HoldOutDist[0];
                }
                buffer.append(" [")
                        .append(Utils.doubleToString(m_HoldOutDist[0], 2))
                        .append('/').append(Utils.doubleToString(avgError, 2))
                        .append(']');
                return buffer.toString();
            } else {
                final int maxIndex = m_ClassProbs == null ? Utils.maxIndex(parent.m_ClassProbs) : Utils.maxIndex(m_ClassProbs);
                return " : " + m_Info.classAttribute().value(maxIndex) +
                        " (" + Utils.doubleToString(Utils.sum(m_Distribution), 2) +
                        '/' +
                        Utils.doubleToString(Utils.sum(m_Distribution) -
                                m_Distribution[maxIndex], 2) + ')' +
                        " [" + Utils.doubleToString(Utils.sum(m_HoldOutDist), 2) + '/' +
                        Utils.doubleToString(Utils.sum(m_HoldOutDist) -
                                m_HoldOutDist[maxIndex], 2) + ']';
            }
        }

        /**
         * Recursively generates a tree.
         *
         * @param sortedIndices the sorted indices of the instances
         * @param weights       the weights of the instances
         * @param data          the data to work with
         * @param totalWeight
         * @param classProbs    the class probabilities
         * @param header        the header of the data
         * @param minNum        the minimum number of instances in a leaf
         * @param minVariance
         * @param depth         the current depth of the tree
         * @param maxDepth      the maximum allowed depth of the tree
         * @throws Exception if generation fails
         */
        @SuppressWarnings({"OverlyComplexMethod", "OverlyLongMethod", "ConstantConditions", "PointlessBooleanExpression"})
        protected void buildTree(
                final int[][][] sortedIndices,
                final double[][][] weights,
                final Instances data,
                final double totalWeight,
                final double[] classProbs,
                final Instances header,
                final double minNum,
                final double minVariance,
                final int depth,
                final int maxDepth) {

            // Store structure of dataset, set minimum number of instances
            // and make space for potential info from pruning data
            m_Info = header;
            m_HoldOutDist = data.classAttribute().isNumeric() ? new double[2] : new double[data.numClasses()];

            // Make leaf if there are no training instances
            int helpIndex = 0;
            if (data.classIndex() == 0) {
                helpIndex = 1;
            }
            if (sortedIndices[0][helpIndex].length == 0) {
                m_Distribution = data.classAttribute().isNumeric() ? new double[2] : new double[data.numClasses()];
                m_ClassProbs = null;
                sortedIndices[0] = null;
                weights[0] = null;
                return;
            }

            double priorVar = 0;
            if (data.classAttribute().isNumeric()) {

                // Compute prior variance
                double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
                for (int i = 0; i < sortedIndices[0][helpIndex].length; i++) {
                    final Instance inst = data.instance(sortedIndices[0][helpIndex][i]);
                    totalSum += inst.classValue() * weights[0][helpIndex][i];
                    totalSumSquared +=
                            inst.classValue() * inst.classValue() * weights[0][helpIndex][i];
                    totalSumOfWeights += weights[0][helpIndex][i];
                }
                priorVar = singleVariance(totalSum, totalSumSquared,
                        totalSumOfWeights);
            }

            // Check if node doesn't contain enough instances, is pure
            // or the maximum tree depth is reached
            m_ClassProbs = new double[classProbs.length];
            System.arraycopy(classProbs, 0, m_ClassProbs, 0, classProbs.length);
            //noinspection OverlyComplexBooleanExpression
            if (totalWeight < 2 * minNum ||
                    // Nominal case
                    data.classAttribute().isNominal() &&
                            Utils.eq(m_ClassProbs[Utils.maxIndex(m_ClassProbs)],
                                    Utils.sum(m_ClassProbs)) ||
                    // Numeric case
                    data.classAttribute().isNumeric() &&
                            priorVar / totalWeight < minVariance ||
                    // Check tree depth
                    M_MAX_DEPTH >= 0 && depth >= maxDepth) {

                // Make leaf
                m_Attribute = -1;
                distribution(data, totalWeight, priorVar);
                sortedIndices[0] = null;
                weights[0] = null;
                return;
            }

            // Compute class distributions and value of splitting
            // criterion for each attribute
            final double[] vals = new double[data.numAttributes()];
            final double[][][] dists = new double[data.numAttributes()][0][0];
            final double[][] props = new double[data.numAttributes()][0];
            final double[][] totalSubsetWeights = new double[data.numAttributes()][0];
            final double[] splits = new double[data.numAttributes()];
            if (data.classAttribute().isNominal()) {

                // Nominal case
                for (int i = 0; i < data.numAttributes(); i++) {
                    if (i != data.classIndex()) {
                        splits[i] = distribution(props, dists, i, sortedIndices[0][i],
                                weights[0][i], totalSubsetWeights, data);
                        vals[i] = gain(dists[i], priorVal(dists[i]));
                    }
                }
            } else {

                // Numeric case
                for (int i = 0; i < data.numAttributes(); i++) {
                    if (i != data.classIndex()) {
                        splits[i] =
                                numericDistribution(props, dists, i, sortedIndices[0][i],
                                        weights[0][i], totalSubsetWeights, data,
                                        vals);
                    }
                }
            }

            // Find best attribute
            m_Attribute = Utils.maxIndex(vals);
            final int numAttVals = dists[m_Attribute].length;

            // Check if there are at least two subsets with
            // required minimum number of instances
            int count = 0;
            for (int i = 0; i < numAttVals; i++) {
                if (totalSubsetWeights[m_Attribute][i] >= minNum) {
                    count++;
                }
                if (count > 1) {
                    break;
                }
            }

            // Any useful split found?
            if (Utils.gr(vals[m_Attribute], 0) && count > 1) {
                // Set split point, proportions, and temp arrays
                m_SplitPoint = splits[m_Attribute];
                m_Prop = props[m_Attribute];
                final double[][] attSubsetDists = dists[m_Attribute];
                final double[] attTotalSubsetWeights = totalSubsetWeights[m_Attribute];

                // Split data
                final int[][][][] subsetIndices = new int[numAttVals][1][data.numAttributes()][0];
                final double[][][][] subsetWeights = new double[numAttVals][1][data.numAttributes()][0];
                splitData(subsetIndices, subsetWeights, m_Attribute, m_SplitPoint,
                        sortedIndices[0], weights[0], data);

                // Release memory
                sortedIndices[0] = null;
                weights[0] = null;

                // Build successors
                m_Successors = new Tree[numAttVals];
                for (int i = 0; i < numAttVals; i++) {
                    m_Successors[i] = new Tree();
                    m_Successors[i].
                            buildTree(subsetIndices[i], subsetWeights[i],
                                    data, attTotalSubsetWeights[i],
                                    attSubsetDists[i], header, minNum,
                                    minVariance, depth + 1, maxDepth);

                    // Release as much memory as we can
                    attSubsetDists[i] = null;
                }
            } else {

                // Make leaf
                m_Attribute = -1;
                sortedIndices[0] = null;
                weights[0] = null;
            }

            // Normalize class counts
            distribution(data, totalWeight, priorVar);
        }

        private void distribution(
                final Instances data,
                final double totalWeight,
                final double priorVar) {
            if (data.classAttribute().isNominal()) {
                m_Distribution = new double[m_ClassProbs.length];
                System.arraycopy(m_ClassProbs, 0, m_Distribution, 0, m_ClassProbs.length);
                Utils.normalize(m_ClassProbs);
            } else {
                m_Distribution = new double[2];
                m_Distribution[0] = priorVar;
                m_Distribution[1] = totalWeight;
            }
        }

        /**
         * Splits instances into subsets.
         *
         * @param subsetIndices the sorted indices in the subset
         * @param subsetWeights the weights of the subset
         * @param att           the attribute index
         * @param splitPoint    the split point for numeric attributes
         * @param sortedIndices the sorted indices of the whole set
         * @param weights       the weights of the whole set
         * @param data          the data to work with
         * @throws Exception if something goes wrong
         */
        @SuppressWarnings({"OverlyComplexMethod", "OverlyLongMethod"})
        protected void splitData(final int[][][][] subsetIndices,
                                 final double[][][][] subsetWeights,
                                 final int att,
                                 final double splitPoint,
                                 final int[][] sortedIndices,
                                 final double[][] weights,
                                 final Instances data) {

            // For each attribute
            for (int i = 0; i < data.numAttributes(); i++) {
                if (i != data.classIndex()) {
                    final int[] num;
                    int j;
                    if (data.attribute(att).isNominal()) {

                        // For nominal attributes
                        num = new int[data.attribute(att).numValues()];
                        for (int k = 0; k < num.length; k++) {
                            subsetIndices[k][0][i] = new int[sortedIndices[i].length];
                            subsetWeights[k][0][i] = new double[sortedIndices[i].length];
                        }
                        for (j = 0; j < sortedIndices[i].length; j++) {
                            final Instance inst = data.instance(sortedIndices[i][j]);
                            if (inst.isMissing(att)) {

                                // Split instance up
                                splitInstanceUp(subsetIndices, subsetWeights, sortedIndices, weights, j, num, i);
                            } else {
                                final int subset = (int) inst.value(att);
                                subsetIndices[subset][0][i][num[subset]] =
                                        sortedIndices[i][j];
                                subsetWeights[subset][0][i][num[subset]] = weights[i][j];
                                num[subset]++;
                            }
                        }
                    } else {

                        // For numeric attributes
                        num = new int[2];
                        for (int k = 0; k < 2; k++) {
                            subsetIndices[k][0][i] = new int[sortedIndices[i].length];
                            subsetWeights[k][0][i] = new double[weights[i].length];
                        }
                        for (j = 0; j < sortedIndices[i].length; j++) {
                            final Instance inst = data.instance(sortedIndices[i][j]);
                            if (inst.isMissing(att)) {
                                // Split instance up
                                splitInstanceUp(subsetIndices, subsetWeights, sortedIndices, weights, j, num, i);
                            } else {
                                final int subset = inst.value(att) < splitPoint ? 0 : 1;
                                subsetIndices[subset][0][i][num[subset]] = sortedIndices[i][j];
                                subsetWeights[subset][0][i][num[subset]] = weights[i][j];
                                num[subset]++;
                            }
                        }
                    }

                    // Trim arrays
                    for (int k = 0; k < num.length; k++) {
                        final int[] copy = new int[num[k]];
                        System.arraycopy(subsetIndices[k][0][i], 0, copy, 0, num[k]);
                        subsetIndices[k][0][i] = copy;
                        final double[] copyWeights = new double[num[k]];
                        System.arraycopy(subsetWeights[k][0][i], 0,
                                copyWeights, 0, num[k]);
                        subsetWeights[k][0][i] = copyWeights;
                    }
                }
            }
        }

        private void splitInstanceUp(
                final int[][][][] subsetIndices,
                final double[][][][] subsetWeights,
                final int[][] sortedIndices,
                final double[][] weights,
                final int j,
                final int[] num,
                final int i) {
            for (int k = 0; k < num.length; k++) {
                if (m_Prop[k] > 0) {
                    subsetIndices[k][0][i][num[k]] = sortedIndices[i][j];
                    subsetWeights[k][0][i][num[k]] =
                            m_Prop[k] * weights[i][j];
                    num[k]++;
                }
            }
        }

        /**
         * Prunes the tree using the hold-out data (bottom-up).
         *
         * @return the error
         * @throws Exception if pruning fails for some reason
         */
        protected double reducedErrorPrune() {
            // Is node leaf ?
            if (m_Attribute == -1) {
                return m_HoldOutError;
            }

            // Prune all sub trees
            double errorTree = 0;
            for (Tree m_Successor : m_Successors) {
                errorTree += m_Successor.reducedErrorPrune();
            }

            // Replace sub tree with leaf if error doesn't get worse
            if (errorTree >= m_HoldOutError) {
                m_Attribute = -1;
                //noinspection AssignmentToNull
                m_Successors = null;
                return m_HoldOutError;
            } else {
                return errorTree;
            }
        }

        /**
         * Inserts hold-out set into tree.
         *
         * @param data the data to insert
         * @throws Exception if something goes wrong
         */
        protected void insertHoldOutSet(final Instances data) {
            for (int i = 0; i < data.numInstances(); i++) {
                insertHoldOutInstance(data.instance(i), data.instance(i).weight(), this);
            }
        }

        /**
         * Inserts an instance from the hold-out set into the tree.
         *
         * @param inst   the instance to insert
         * @param weight the weight of the instance
         * @param parent the parent of the node
         * @throws Exception if insertion fails
         */
        @SuppressWarnings("OverlyComplexMethod")
        protected void insertHoldOutInstance(
                final Instance inst,
                final double weight,
                final Tree parent) {

            // Insert instance into hold-out class distribution
            if (inst.classAttribute().isNominal()) {

                // Nominal case
                //noinspection NumericCastThatLosesPrecision
                m_HoldOutDist[(int) inst.classValue()] += weight;
                final int predictedClass = m_ClassProbs == null ?
                        Utils.maxIndex(parent.m_ClassProbs) :
                        Utils.maxIndex(m_ClassProbs);
                //noinspection NumericCastThatLosesPrecision
                if (predictedClass != (int) inst.classValue()) {
                    m_HoldOutError += weight;
                }
            } else {

                // Numeric case
                m_HoldOutDist[0] += weight;
                m_HoldOutDist[1] += weight * inst.classValue();
                final double diff = m_ClassProbs == null ?
                        parent.m_ClassProbs[0] - inst.classValue() :
                        m_ClassProbs[0] - inst.classValue();
                m_HoldOutError += diff * diff * weight;
            }

            // The process is recursive
            if (m_Attribute != -1) {

                // If node is not a leaf
                if (inst.isMissing(m_Attribute)) {

                    // Distribute instance
                    for (int i = 0; i < m_Successors.length; i++) {
                        if (m_Prop[i] > 0) {
                            m_Successors[i].insertHoldOutInstance(inst, weight *
                                    m_Prop[i], this);
                        }
                    }
                } else {

                    if (m_Info.attribute(m_Attribute).isNominal()) {

                        // Treat nominal attributes
                        //noinspection NumericCastThatLosesPrecision
                        m_Successors[(int) inst.value(m_Attribute)].
                                insertHoldOutInstance(inst, weight, this);
                    } else {

                        // Treat numeric attributes
                        if (inst.value(m_Attribute) < m_SplitPoint) {
                            m_Successors[0].insertHoldOutInstance(inst, weight, this);
                        } else {
                            m_Successors[1].insertHoldOutInstance(inst, weight, this);
                        }
                    }
                }
            }
        }

        /**
         * Backfits data from holdout set.
         *
         * @throws Exception if insertion fails
         */
        protected void backfitHoldOutSet() {

            // Insert instance into hold-out class distribution
            if (m_Info.classAttribute().isNominal()) {
                // Nominal case
                if (m_ClassProbs == null) {
                    m_ClassProbs = new double[m_Info.numClasses()];
                }
                System.arraycopy(m_Distribution, 0, m_ClassProbs, 0, m_Info.numClasses());
                for (int i = 0; i < m_HoldOutDist.length; i++) {
                    m_ClassProbs[i] += m_HoldOutDist[i];
                }
                if (Utils.sum(m_ClassProbs) > 0) {
                    Utils.normalize(m_ClassProbs);
                } else {
                    //noinspection AssignmentToNull
                    m_ClassProbs = null;
                }
            } else {

                // Numeric case
                final double sumOfWeightsTrainAndHoldout = m_Distribution[1] + m_HoldOutDist[0];
                if (sumOfWeightsTrainAndHoldout <= 0) {
                    return;
                }
                if (m_ClassProbs == null) {
                    m_ClassProbs = new double[1];
                } else {
                    m_ClassProbs[0] *= m_Distribution[1];
                }
                m_ClassProbs[0] += m_HoldOutDist[1];
                m_ClassProbs[0] /= sumOfWeightsTrainAndHoldout;
            }

            // The process is recursive
            if (m_Attribute != -1) {
                for (Tree m_Successor : m_Successors) {
                    m_Successor.backfitHoldOutSet();
                }
            }
        }
    }
}
