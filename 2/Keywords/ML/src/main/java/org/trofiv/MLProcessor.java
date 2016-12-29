package org.trofiv;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math.stat.descriptive.rank.Percentile;
import weka.core.Instances;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

@SuppressWarnings("WeakerAccess")
public class MLProcessor {
    private static final int BOOTSTRAP_COUNT = 1000;
    private static final double QUANTILE_10_TH = 10.0;
    private static final double QUANTILE_90_TH = 90.0;
    private static final double MEDIAN_QUANTILE = 50.0;
    private static final int NUM_BOOSTING_ITERATIONS = 5000;

    private static final Random RANDOM = new SecureRandom();
    private static final Percentile QUANTILE = new Percentile();
    private static final String INPUT_DIRECTORIES_LOCATION = "../";
    private static final Pattern INPUT_FILES = Pattern.compile(".*arff");
    private static final Pattern INPUT_DIRECTORIES = Pattern.compile("keywords.*");
    private static final Collection<ImprovedClassifier> CLASSIFIERS = new LinkedList<>();

    static {
        final RegressionDecisionTree tree = new RegressionDecisionTree();

        final BoostingAdditiveRegressionClassifier boosting = new BoostingAdditiveRegressionClassifier(tree);
        boosting.setNumIterations(NUM_BOOSTING_ITERATIONS);

        //noinspection TypeMayBeWeakened
        final LinearRegressionClassifier regression = new LinearRegressionClassifier();

        CLASSIFIERS.add(regression);
        CLASSIFIERS.add(boosting);
    }

    public static void main(final String[] args) {
        try {
            final File workDirectory = Paths.get(INPUT_DIRECTORIES_LOCATION).toRealPath().toFile();
            final File[] keywordDirectories = workDirectory.listFiles(
                    path -> path.isDirectory() && INPUT_DIRECTORIES.matcher(path.getName()).matches());

            if (keywordDirectories.length == 0) {
                throw new IllegalArgumentException("No one input directory found!");
            }

            for (File dir : keywordDirectories) {
                final File[] fileToProcess = dir.listFiles(path ->
                        path.isFile() && INPUT_FILES.matcher(path.getName()).matches());

                if (fileToProcess.length == 0) {
                    throw new IllegalArgumentException("No one input file found!");
                }

                for (File file : fileToProcess) {
                    processFile(file);
                }
            }
        } catch (IOException e) {
            System.out.println("Can't access working directory or missing dataset files");
            System.out.println(e.getLocalizedMessage());
        }
    }

    private static void processFile(final File fileName) {
        final Instances data = readData(fileName);
        System.out.println("Process file " + fileName + ':');
        final List<BootstrapResults> results = analyze(data);

        for (BootstrapResults result : results) {
            printResults(fileName, result);
        }
    }

    private static Instances readData(final File file) {
        try (final BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            final Instances data = new Instances(reader);
            data.setClass(data.attribute(data.numAttributes() - 1));
            return data;
        } catch (final FileNotFoundException e) {
            throw new IllegalArgumentException("Can't find file " + file, e);
        } catch (final IOException e) {
            throw new IllegalArgumentException("Error parsing file " + file, e);
        }
    }

    private static List<BootstrapResults> analyze(final Instances data) {
        final List<BootstrapResults> result = new LinkedList<>();

        for (ImprovedClassifier classifier : CLASSIFIERS) {
            try {
                System.out.println("    Processing " + classifier.getClass().getName() + ':');

                final double[][] quality = new double[BOOTSTRAP_COUNT][];
                final double[][] ncg = new double[BOOTSTRAP_COUNT][];

                for (int i = 0; i < BOOTSTRAP_COUNT; i++) {
                    System.out.print("      Bootstrap iteration " + i + " of " + BOOTSTRAP_COUNT + '\r');

                    final Instances trainData = data.resample(RANDOM);
                    classifier.evaluate(trainData, data);

                    quality[i] = classifier.attributeQuality();
                    ncg[i] = classifier.ncg();
                }

                final List<double[]> zippedNcg = Utils.zip(ncg);
                final List<double[]> zippedQuality = Utils.zip(quality);
                final List<Pair<String, Double>> rangedAttributes = new LinkedList<>();

                for (int attrIndex = 0; attrIndex < zippedQuality.size(); attrIndex++) {
                    final double median = QUANTILE.evaluate(zippedQuality.get(attrIndex), MEDIAN_QUANTILE);
                    rangedAttributes.add(Pair.of(data.attribute(attrIndex).name(), median));
                }

                final List<Triple<Double, Double, Double>> quantileNcg = extractNcg(zippedNcg);

                rangedAttributes.sort((o1, o2) -> -o1.getRight().compareTo(o2.getRight()));

                result.add(new BootstrapResults(classifier.getClass().getSimpleName(), rangedAttributes, quantileNcg));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private static void printResults(final File fileName, final BootstrapResults result) {
        final String baseName = fileName.getAbsolutePath().substring(0, fileName.getAbsolutePath().lastIndexOf('.'));
        final String resultFileName = baseName + '_' + result.getClassifierName() + ".tsv";

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(resultFileName), StandardCharsets.UTF_8)) {
            final CSVPrinter out = CSVFormat.MYSQL.print(writer);

            out.printRecord("Results for " + result.getClassifierName());
            out.printRecord("Ranged attributes (zero means exclude from model)");
            out.printRecord("Attribute", "Importance");

            for (Pair<String, Double> attributes : result.getRangedAttributes()) {
                out.printRecord(attributes.getLeft(), attributes.getRight());
            }

            out.printRecord("Post count", "Quantile 10%", "Quantile 50%", "Quantile 90%");

            int count = 0;
            for (Triple<Double, Double, Double> quantiles : result.getQuantiles()) {
                out.printRecord(count++, quantiles.getLeft(), quantiles.getMiddle(), quantiles.getRight());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<Triple<Double, Double, Double>> extractNcg(final Iterable<double[]> zippedNcg) {
        final List<Triple<Double, Double, Double>> quantileNcg = new LinkedList<>();

        for (double[] ncgForPostCount : zippedNcg) {
            final double median = QUANTILE.evaluate(ncgForPostCount, MEDIAN_QUANTILE);
            final double quantile10th = QUANTILE.evaluate(ncgForPostCount, QUANTILE_10_TH);
            final double quantile10thDelta = median - quantile10th;
            final double quantile90th = QUANTILE.evaluate(ncgForPostCount, QUANTILE_90_TH);
            final double quantile90thDelta = quantile90th - median;
            quantileNcg.add(Triple.of(quantile10thDelta, median, quantile90thDelta));
        }
        return quantileNcg;
    }

    private static final class BootstrapResults {
        private final String classifierName;
        private final List<Pair<String, Double>> rangedAttributes;
        private final List<Triple<Double, Double, Double>> quantiles;

        private BootstrapResults(
                final String classifierName,
                final List<Pair<String, Double>> rangedAttributes,
                final List<Triple<Double, Double, Double>> quantiles) {
            this.classifierName = classifierName;
            this.rangedAttributes = rangedAttributes;
            this.quantiles = quantiles;
        }

        public String getClassifierName() {
            return classifierName;
        }

        @SuppressWarnings("TypeMayBeWeakened")
        public List<Pair<String, Double>> getRangedAttributes() {
            return rangedAttributes;
        }

        @SuppressWarnings("TypeMayBeWeakened")
        public List<Triple<Double, Double, Double>> getQuantiles() {
            return quantiles;
        }
    }
}