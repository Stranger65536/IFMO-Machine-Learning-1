import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.math3.util.KthSelector;
import org.apache.commons.math3.util.MathArrays.Function;
import org.apache.commons.math3.util.MedianOf3PivotingStrategy;
import org.jscience.mathematics.number.Rational;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.*;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.*;
import weka.classifiers.meta.nestedDichotomies.ClassBalancedND;
import weka.classifiers.meta.nestedDichotomies.DataNearBalancedND;
import weka.classifiers.meta.nestedDichotomies.ND;
import weka.classifiers.misc.HyperPipes;
import weka.classifiers.misc.VFI;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.core.Instances;

import java.io.*;
import java.security.SecureRandom;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Processor {
    private static final int BOOTSTRAP_COUNT = 100;
    private static final double ZERO_THRESHOLD = 1.0E-32;
    private static final StandardDeviation STANDARD_DEVIATION = new StandardDeviation();
    private static final Function MEAN = new Mean();

    private static final Collection<Classifier> CLASSIFIERS = Arrays.asList(
            new LibSVM(),
            new AdaBoostM1(),
            new SimpleLogistic(),
            new RandomTree(),
            new NaiveBayes(),
            new PART(),
            new Grading(),
            new Dagging(),
            new Stacking(),
            new IB1(),
            new KStar(),
            new ClassificationViaRegression(),
            new RacedIncrementalLogitBoost(),
            new RBFNetwork(),
            new IBk(),
            new RotationForest(),
            new ND(),
            new ZeroR(),
            new MultiClassClassifier(),
            new FT(),
            new RandomSubSpace(),
            new MultiScheme(),
            new DecisionStump(),
            new ConjunctiveRule(),
            new J48(),
            new END(),
            new ClassBalancedND(),
            new J48graft(),
            new SMO(),
            new HyperPipes(),
            new StackingC(),
            new JRip(),
            new NNge(),
            new MultilayerPerceptron(),
            new DecisionTable(),
            new FilteredClassifier(),
            new Bagging(),
            new Logistic(),
            new LogitBoost(),
            new MultiBoostAB(),
            new Vote(),
            new VFI(),
            new AttributeSelectedClassifier(),
            new OneR(),
            new CVParameterSelection(),
            new DTNB(),
            new BFTree(),
            new LWL(),
            new DataNearBalancedND(),
            new LMT(),
            new Ridor(),
            new Decorate(),
            new NBTree(),
            new REPTree(),
            new LADTree(),
            new RandomCommittee(),
            new ClassificationViaClustering()
    );

    public static void main(final String[] args) {
        if (args.length < 1) {
            throw new IllegalArgumentException("Usage: 'java Processor filename...'");
        }

        for (String fileName : args) {
            final String baseName = FilenameUtils.getBaseName(fileName);
            //noinspection ImplicitDefaultCharsetUsage,ObjectAllocationInLoop,MagicCharacter
            try (CSVPrinter out = CSVFormat.MYSQL.print(new PrintWriter(baseName + "_results" + '.' + "tsv"))) {
                processFile(fileName, out);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static void processFile(final String fileName, final CSVPrinter out) {
        final Instances data = readData(fileName);
        final BootstrapResults results = bootstrapAnalyzing(data);
        printBootstrapResults(results, out);
    }

    @SuppressWarnings({"OverlyLongMethod", "PrimitiveArrayArgumentToVariableArgMethod"})
    private static void printBootstrapResults(final BootstrapResults results, final CSVPrinter out) {
        try {
            Object[] result = null;
            //noinspection ConstantConditions
            result = ArrayUtils.add(ArrayUtils.nullToEmpty(result), "Classifier");
            result = ArrayUtils.add(result, "Micro f-measure mean");
            result = ArrayUtils.add(result, "Micro f-measure deviation");
            result = ArrayUtils.add(result, "Macro f-measure mean");
            result = ArrayUtils.add(result, "Macro f-measure deviation");
            result = ArrayUtils.add(result, "Correctly predicted percent mean");
            result = ArrayUtils.add(result, "Correctly predicted percent deviation");
            result = ArrayUtils.addAll(result, IntStream.range(1, results.getNcg().values().iterator().next().size() + 1).mapToObj(k -> "NCG for k = " + k).toArray());
            result = ArrayUtils.addAll(result, IntStream.range(1, results.getNcg().values().iterator().next().size() + 1).mapToObj(k -> "Quantile 10% for k = " + k).toArray());
            result = ArrayUtils.addAll(result, IntStream.range(1, results.getNcg().values().iterator().next().size() + 1).mapToObj(k -> "Quantile 90% for k = " + k).toArray());
            out.printRecord(Arrays.asList(result));
        } catch (IOException e) {
            e.printStackTrace();
        }

        CLASSIFIERS.stream().forEach(classifier -> {
            final double microDeviation = STANDARD_DEVIATION.evaluate(results.getMicroFMeasures().get(classifier));
            final double microMean = MEAN.evaluate(results.getMicroFMeasures().get(classifier));

            final double macroDeviation = STANDARD_DEVIATION.evaluate(results.getMacroFMeasures().get(classifier));
            final double macroMean = MEAN.evaluate(results.getMacroFMeasures().get(classifier));

            final double correctDeviation = STANDARD_DEVIATION.evaluate(results.getPctCorrect().get(classifier));
            final double correctMean = MEAN.evaluate(results.getPctCorrect().get(classifier));

            Object[] result = null;
            //noinspection ConstantConditions
            result = ArrayUtils.add(ArrayUtils.nullToEmpty(result), classifier.getClass().getSimpleName());
            result = ArrayUtils.add(result, microMean);
            result = ArrayUtils.add(result, microDeviation);
            result = ArrayUtils.add(result, macroMean);
            result = ArrayUtils.add(result, macroDeviation);
            result = ArrayUtils.add(result, correctMean);
            result = ArrayUtils.add(result, correctDeviation);
            result = ArrayUtils.addAll(result, results.getNcg().get(classifier).stream().map(Triple::getLeft).toArray());
            result = ArrayUtils.addAll(result, results.getNcg().get(classifier).stream().map(Triple::getMiddle).toArray());
            result = ArrayUtils.addAll(result, results.getNcg().get(classifier).stream().map(Triple::getRight).toArray());

            try {
                out.printRecord(Arrays.asList(result));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    @SuppressWarnings("OverlyLongMethod")
    private static BootstrapResults bootstrapAnalyzing(final Instances data) {
        final int hash = data.hashCode();

        final Map<Classifier, double[]> microFMeasures = new LinkedHashMap<>(CLASSIFIERS.size(), 1);
        final Map<Classifier, double[]> macroFMeasures = new LinkedHashMap<>(CLASSIFIERS.size(), 1);
        final Map<Classifier, double[]> pctCorrect = new LinkedHashMap<>(CLASSIFIERS.size(), 1);
        final Map<Classifier, List<List<Double>>> ncgRaw = new LinkedHashMap<>(CLASSIFIERS.size(), 1);
        final Map<Classifier, List<Triple<Double, Double, Double>>> ncg = new LinkedHashMap<>(CLASSIFIERS.size(), 1);

        CLASSIFIERS.stream().forEach(classifier -> {
            microFMeasures.put(classifier, new double[BOOTSTRAP_COUNT]);
            macroFMeasures.put(classifier, new double[BOOTSTRAP_COUNT]);
            pctCorrect.put(classifier, new double[BOOTSTRAP_COUNT]);
            pctCorrect.put(classifier, new double[BOOTSTRAP_COUNT]);
            ncgRaw.put(classifier, new ArrayList<>(BOOTSTRAP_COUNT));
            ncg.put(classifier, new ArrayList<>(BOOTSTRAP_COUNT));
        });

        int current = 1;
        final int total = BOOTSTRAP_COUNT;
        for (int bootstrapIteration = 0; bootstrapIteration < BOOTSTRAP_COUNT; bootstrapIteration++) {
            //noinspection MagicCharacter,SingleCharacterStringConcatenation,CharUsedInArithmeticContext
            System.err.println("[" + hash + "]" + " [INFO] " + "bootstrap iteration " + current++ + " of " + total);
            final Pair<Instances, Instances> bootsrapped = bootstrapInstances(data);
            final Instances testInstances = bootsrapped.getLeft();
            final Instances trainInstances = bootsrapped.getRight();
            processBootstrapClassifiers(microFMeasures, macroFMeasures, pctCorrect,
                    ncgRaw, bootstrapIteration, testInstances, trainInstances);
        }

        CLASSIFIERS.stream().forEach(classifier -> {
            final List<List<Double>> curNcgRaw = ncgRaw.get(classifier);
            final List<Triple<Double, Double, Double>> curNcg = ncg.get(classifier);

            final Percentile quantifier = new PercentileExcel();
            Collection<List<Double>> ncgTranspon = new ArrayList<>(data.numInstances());

            for (int k = 0; k < data.numInstances(); k++) {
                List<Double> values = new ArrayList<>(BOOTSTRAP_COUNT);

                for (int i = 0; i < BOOTSTRAP_COUNT; i++) {
                    values.add(curNcgRaw.get(i).get(k));
                }

                ncgTranspon.add(values);
            }

            ncgTranspon.stream().forEach(kList -> {
                quantifier.setData(kList.stream().mapToDouble(v -> v).toArray());
                final double quantize10th = quantifier.evaluate(10);
                //noinspection MagicNumber
                final double quantize90th = quantifier.evaluate(90);
                final double average = kList.stream().mapToDouble(v -> v).average().getAsDouble();

                curNcg.add(Triple.of(average, quantize10th, quantize90th));
            });
        });

        return new BootstrapResults(microFMeasures, macroFMeasures, pctCorrect, ncg);
    }

    @SuppressWarnings("MagicNumber")
    private static void processBootstrapClassifiers(
            final Map<Classifier, double[]> microFMeasures,
            final Map<Classifier, double[]> macroFMeasures,
            final Map<Classifier, double[]> pctCorrect,
            final Map<Classifier, List<List<Double>>> ncgRaw,
            final int bootstrapIteration,
            final Instances testInstances,
            final Instances trainInstances) {
        CLASSIFIERS.stream().forEach(classifier -> {
            try {
                final Evaluation evaluation = new Evaluation(trainInstances);

                classifier.buildClassifier(trainInstances);
                evaluation.evaluateModel(classifier, testInstances);

                final List<Double> predictedUtility = new ArrayList<>(evaluation.predictions().size());
                final List<Double> originalUtility = new ArrayList<>(evaluation.predictions().size());

                for (int i = 0; i < evaluation.predictions().size(); i++) {
                    predictedUtility.add(((NominalPrediction) evaluation.predictions().elementAt(i)).predicted());
                    originalUtility.add(((NominalPrediction) evaluation.predictions().elementAt(i)).actual());
                }

                final List<Double> ncg = getNCG(predictedUtility, originalUtility);
                ncgRaw.get(classifier).add(ncg);

                final double[][] confusionMatrix = evaluation.confusionMatrix();
                final List<Rational> precisions = precisions(confusionMatrix);
                final List<Rational> recalls = recalls(confusionMatrix);

                final double microAvgPrecision = microAverage(precisions);
                final double microAvgRecall = microAverage(recalls);
                final double microFMeasure = fMeasure(microAvgPrecision, microAvgRecall);
                microFMeasures.get(classifier)[bootstrapIteration] = microFMeasure;

                final double macroAvgPrecision = macroAverage(precisions);
                final double macroAvgRecall = macroAverage(recalls);
                final double macroFMeasure = fMeasure(macroAvgPrecision, macroAvgRecall);
                macroFMeasures.get(classifier)[bootstrapIteration] = macroFMeasure;

                pctCorrect.get(classifier)[bootstrapIteration] = evaluation.pctCorrect();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private static List<Double> getNCG(final List<Double> predictedUtility, final List<Double> originalUtility) {
        final Collection<Pair<Double, Double>> joined = new LinkedList<>();

        for (int i = 0; i < predictedUtility.size(); i++) {
            joined.add(Pair.of(originalUtility.get(i), predictedUtility.get(i)));
        }

        final List<Double> sortedByOriginal = originalUtility
                .stream()
                .sorted()
                .collect(Collectors.toList());
        Collections.reverse(sortedByOriginal);

        final List<Pair<Double, Double>> sortedByPredicted = joined.stream().collect(Collectors.toList());
        sortedByPredicted.sort((o1, o2) -> -o1.getRight().compareTo(o2.getRight()));

        final List<Double> result = new ArrayList<>(joined.size());

        for (int k = 1; k <= joined.size(); k++) {
            final List<Double> originalShort = sortedByOriginal
                    .stream()
                    .limit(k)
                    .collect(Collectors.toList());
            final List<Double> predictedShort = sortedByPredicted
                    .stream()
                    .limit(k)
                    .mapToDouble(Pair::getLeft)
                    .boxed()
                    .collect(Collectors.toList());

            final double divident = predictedShort.stream().mapToDouble(p -> p).sum();
            final double divisor = originalShort.stream().mapToDouble(p -> p).sum();

            if (divident > divisor) {
                throw new IllegalArgumentException("Divident greater that divisor!");
            }

            result.add(divident / divisor);
        }

        return result;
    }

    private static Instances readData(final String fileName) {
        try (final BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileName), "UTF8"))) {
            final Instances data = new Instances(reader);
            data.setClass(data.attribute(data.numAttributes() - 1));
            return data;
        } catch (final FileNotFoundException e) {
            throw new IllegalArgumentException("Can't find file " + fileName, e);
        } catch (final IOException e) {
            throw new IllegalArgumentException("Error parsing file " + fileName, e);
        }
    }

    @SuppressWarnings("NumericCastThatLosesPrecision")
    private static Pair<Instances, Instances> bootstrapInstances(final Instances data) {
        return Pair.of(data, data.resample(new SecureRandom()));
    }

    private static double fMeasure(final double precision, final double recall) {
        return Math.abs(precision + recall) < ZERO_THRESHOLD
                ? 0
                : 2 * precision * recall / (precision + recall);
    }

    private static List<Rational> precisions(final double[][] confusionMatrix) {
        final List<Rational> result = new LinkedList<>();

        IntStream.range(0, confusionMatrix.length).forEach(classIndex -> {
            double correct = 0;
            double total = 0;

            for (int i = 0; i < confusionMatrix.length; i++) {
                if (i == classIndex) {
                    correct += confusionMatrix[i][classIndex];
                }
                total += confusionMatrix[i][classIndex];
            }

            if (Math.abs(total) < ZERO_THRESHOLD) {
                result.add(Rational.valueOf(0, 1));
            } else {
                result.add(Rational.valueOf(new Double(correct).longValue(), new Double(total).longValue()));
            }
        });

        return result;
    }

    private static List<Rational> recalls(final double[][] confusionMatrix) {
        final List<Rational> result = new LinkedList<>();

        IntStream.range(0, confusionMatrix.length).forEach(classIndex -> {
            double correct = 0;
            double total = 0;

            for (int i = 0; i < confusionMatrix.length; i++) {
                if (i == classIndex) {
                    correct += confusionMatrix[classIndex][i];
                }
                total += confusionMatrix[classIndex][i];
            }

            if (Math.abs(total) < ZERO_THRESHOLD) {
                result.add(Rational.valueOf(0, 1));
            } else {
                result.add(Rational.valueOf(new Double(correct).longValue(), new Double(total).longValue()));
            }
        });

        return result;
    }

    private static double macroAverage(final Collection<Rational> rationals) {
        return rationals
                .stream()
                .mapToDouble(Rational::doubleValue)
                .sum();
    }

    private static double microAverage(final Collection<Rational> rationals) {
        final double dividendSum = rationals
                .stream()
                .filter(rational -> Math.abs(rational.doubleValue()) > ZERO_THRESHOLD)
                .mapToDouble(rational -> rational.getDividend().longValue())
                .sum();
        final double divisorSum = rationals
                .stream()
                .filter(rational -> Math.abs(rational.doubleValue()) > ZERO_THRESHOLD)
                .mapToDouble(rational -> rational.getDivisor().longValue())
                .sum();
        return Math.abs(divisorSum) < ZERO_THRESHOLD ? 0 : dividendSum / divisorSum;
    }

    static class PercentileExcel extends Percentile {
        /**
         * @throws MathIllegalArgumentException
         */
        @SuppressWarnings("MagicNumber")
        public PercentileExcel() {
            super(50.0,
                    EstimationType.R_7, // use excel style interpolation
                    NaNStrategy.REMOVED,
                    new KthSelector(new MedianOf3PivotingStrategy()));
        }
    }

    private static class BootstrapResults {
        private final Map<Classifier, double[]> microFMeasures;
        private final Map<Classifier, double[]> macroFMeasures;
        private final Map<Classifier, double[]> pctCorrect;
        private final Map<Classifier, List<Triple<Double, Double, Double>>> ncg;

        public BootstrapResults(
                final Map<Classifier, double[]> microFMeasures,
                final Map<Classifier, double[]> macroFMeasures,
                final Map<Classifier, double[]> pctCorrect,
                final Map<Classifier, List<Triple<Double, Double, Double>>> ncg) {
            this.microFMeasures = microFMeasures;
            this.macroFMeasures = macroFMeasures;
            this.pctCorrect = pctCorrect;
            this.ncg = ncg;
        }

        public Map<Classifier, double[]> getMicroFMeasures() {
            return microFMeasures;
        }

        public Map<Classifier, double[]> getMacroFMeasures() {
            return macroFMeasures;
        }

        public Map<Classifier, double[]> getPctCorrect() {
            return pctCorrect;
        }

        public Map<Classifier, List<Triple<Double, Double, Double>>> getNcg() {
            return ncg;
        }
    }
}