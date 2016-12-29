package org.trofiv;

import com.google.common.base.Preconditions;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math.stat.descriptive.rank.Percentile;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.*;
import weka.classifiers.misc.HyperPipes;
import weka.classifiers.misc.VFI;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.core.Instances;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("WeakerAccess")
public class MLProcessor {
    private static final int BOOTSTRAP_COUNT = 100;
    private static final double QUANTILE_10_TH = 10.0;
    private static final double QUANTILE_90_TH = 90.0;
    private static final double QUANTILE_50_TH = 50.0;

    private static final Random RANDOM = new SecureRandom();
    private static final Percentile QUANTILE = new Percentile();
    private static final String INPUT_DIRECTORIES_LOCATION = "../";
    private static final Collection<Future> TASKS = new LinkedList<>();
    private static final Pattern INPUT_FILES = Pattern.compile(".*arff");
    private static final Pattern INPUT_DIRECTORIES = Pattern.compile("keywords.*");
    private static final List<Pair<ClassifierGroup, Classifier>> LISTED_CLASSIFIERS = new LinkedList<>();
    private static final Map<ClassifierGroup, Collection<Classifier>> CLASSIFIER_GROUPS = new EnumMap<>(ClassifierGroup.class);
    private static final ExecutorService THREAD_POOL = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    static {
        final Collection<Classifier> decisionTrees = Arrays.asList(
                new BFTree(),
                new FT(),
                new J48(),
                new J48graft(),
                new JRip(),
                new LADTree(),
                new NBTree(),
                new NNge(),
                new PART(),
                new RandomTree(),
                new DecisionTable(),
                new DTNB()
        );

        final Collection<Classifier> regressionClassifiers = Arrays.asList(
                new ClassificationViaRegression(),
                new Logistic(),
                new LogitBoost(),
                new LMT(),
                new Ridor()
        );

        final Collection<Classifier> kNearestClassifiers = Arrays.asList(
                new IB1(),
                new KStar(),
                new IBk()
        );

        final Collection<Classifier> metaClassifiers = Arrays.asList(
                new ConjunctiveRule(),
                new Dagging(),
                new DecisionStump(),
                new Grading(),
                new OneR(),
                new Stacking(),
                new StackingC(),
                new Vote(),
                new ZeroR()
        );

        final Collection<Classifier> vfi = Collections.singletonList(new VFI());
        final Collection<Classifier> svm = Collections.singletonList(new LibSVM());
        final Collection<Classifier> boosting = Collections.singletonList(new REPTree());
        final Collection<Classifier> hyperPipes = Collections.singletonList(new HyperPipes());
        final Collection<Classifier> clustering = Collections.singletonList(new ClassificationViaClustering());

        CLASSIFIER_GROUPS.put(ClassifierGroup.SVM, svm);
        CLASSIFIER_GROUPS.put(ClassifierGroup.VFI, vfi);
        CLASSIFIER_GROUPS.put(ClassifierGroup.BOOSTING, boosting);
        CLASSIFIER_GROUPS.put(ClassifierGroup.META, metaClassifiers);
        CLASSIFIER_GROUPS.put(ClassifierGroup.CLUSTERING, clustering);
        CLASSIFIER_GROUPS.put(ClassifierGroup.HYPER_PIPES, hyperPipes);
        CLASSIFIER_GROUPS.put(ClassifierGroup.DECISION_TREES, decisionTrees);
        CLASSIFIER_GROUPS.put(ClassifierGroup.REGRESSION, regressionClassifiers);
        CLASSIFIER_GROUPS.put(ClassifierGroup.NEAREST_NEIGHBOURS, kNearestClassifiers);

        CLASSIFIER_GROUPS.entrySet().stream().forEach(entry ->
                entry.getValue().forEach(classifier ->
                        LISTED_CLASSIFIERS.add(Pair.of(entry.getKey(), classifier))));
    }

    public static void main(final String[] args) {
        try {
            final File workDirectory = Paths.get(INPUT_DIRECTORIES_LOCATION).toRealPath().toFile();
            final File[] keywordDirectories = workDirectory.listFiles(
                    path -> path.isDirectory() && INPUT_DIRECTORIES.matcher(path.getName()).matches());

            if (keywordDirectories.length == 0) {
                throw new IllegalArgumentException("No one input directory found!");
            }

            final List<GroupedClassificationResults> rawTotal = new LinkedList<>();

            for (File dir : keywordDirectories) {
                final File[] fileToProcess = dir.listFiles(path ->
                        path.isFile() && INPUT_FILES.matcher(path.getName()).matches());

                if (fileToProcess.length == 0) {
                    throw new IllegalArgumentException("No one input file found!");
                }

                for (File file : fileToProcess) {
                    final List<GroupedClassificationResults> rawResults = processFile(file);
                    rawTotal.addAll(rawResults);
                }
            }

            final List<GroupedClassificationResults> total = groupResults(rawTotal);
            printResults(new File("total.tsv"), total);
        } catch (IOException e) {
            System.out.println("Can't access working directory or missing dataset files");
            System.out.println(e.getLocalizedMessage());
        }
    }

    private static List<GroupedClassificationResults> processFile(final File fileName) {
        final Instances data = readData(fileName);
        return analyze(data);
    }

    private static List<GroupedClassificationResults> groupResults(final List<GroupedClassificationResults> source) {
        Preconditions.checkNotNull(source, "Source results can't be null!");
        Preconditions.checkArgument(!source.contains(null), "Sources can't contains null values!");

        final int size = source.isEmpty() ? 0 : source.get(0).getQuantiles().size();

        final Map<ClassifierGroup, Triple<List<List<Double>>, List<List<Double>>, List<List<Double>>>> mapped =
                fillClassifierGroupsMap(size);

        for (GroupedClassificationResults classifierResult : source) {
            final ClassifierGroup group = classifierResult.getClassifierGroup();

            final List<Double> quantile10th = classifierResult.getQuantiles()
                    .stream()
                    .map(Triple::getLeft)
                    .collect(Collectors.toCollection(ArrayList<Double>::new));

            final List<Double> quantile50th = classifierResult.getQuantiles()
                    .stream()
                    .map(Triple::getMiddle)
                    .collect(Collectors.toCollection(ArrayList<Double>::new));

            final List<Double> quantile90th = classifierResult.getQuantiles()
                    .stream()
                    .map(Triple::getRight)
                    .collect(Collectors.toCollection(ArrayList<Double>::new));

            mapped.get(group).getLeft().add(quantile10th);
            mapped.get(group).getMiddle().add(quantile50th);
            mapped.get(group).getRight().add(quantile90th);
        }

        return mapped.entrySet().stream().map(entry -> {
            final List<List<Double>> zippedQuantile10th = Utils.zipLists(entry.getValue().getLeft());
            final List<List<Double>> zippedQuantile50th = Utils.zipLists(entry.getValue().getMiddle());
            final List<List<Double>> zippedQuantile90th = Utils.zipLists(entry.getValue().getRight());

            final List<Triple<Double, Double, Double>> ranged = IntStream
                    .range(0, size)
                    .mapToObj(k -> Triple.of(
                            zippedQuantile10th.get(k).stream().mapToDouble(i -> i).min().getAsDouble(),
                            zippedQuantile50th.get(k).stream().mapToDouble(i -> i).average().getAsDouble(),
                            zippedQuantile90th.get(k).stream().mapToDouble(i -> i).max().getAsDouble()))
                    .collect(Collectors.toList());

            return new GroupedClassificationResults(entry.getKey(), entry.getKey().getValue(), ranged);
        }).collect(Collectors.toList());
    }

    private static void printResults(final File fileName, final List<GroupedClassificationResults> result) {
        Preconditions.checkNotNull(result, "Results can't be null!");
        Preconditions.checkArgument(!result.contains(null), "Results can't contains null values!");

        final String baseName = fileName.getAbsolutePath().substring(0, fileName.getAbsolutePath().lastIndexOf('.'));
        final String resultFileName = baseName + "_results" + ".tsv";

        result.sort((o1, o2) -> o1.getClassifierName().compareTo(o2.getClassifierName()));

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(resultFileName), StandardCharsets.UTF_8)) {
            final CSVPrinter out = CSVFormat.MYSQL.print(writer);

            final List<List<Double>> quantiles10thDelta = result.stream()
                    .map(ClassificationResults::getQuantiles)
                    .map(i -> i.stream()
                            .map(j -> j.getMiddle() - j.getLeft())
                            .collect(Collectors.toCollection(ArrayList<Double>::new)))
                    .collect(Collectors.toList());

            final List<List<Double>> quantiles50th = result.stream()
                    .map(ClassificationResults::getQuantiles)
                    .map(i -> i.stream()
                            .map(Triple::getMiddle)
                            .collect(Collectors.toCollection(ArrayList<Double>::new)))
                    .collect(Collectors.toList());

            final List<List<Double>> quantiles90thDelta = result.stream()
                    .map(ClassificationResults::getQuantiles)
                    .map(i -> i.stream()
                            .map(j -> j.getRight() - j.getMiddle())
                            .collect(Collectors.toCollection(ArrayList<Double>::new)))
                    .collect(Collectors.toList());


            final Collection<String> groupsNames = getHeaderRow(result);

            out.printRecords(Collections.singletonList("Quantiles 50th"));
            out.printRecord(groupsNames);
            printTable(out, quantiles50th);

            out.printRecords(Collections.singletonList("Quantiles 10th delta"));
            out.printRecord(groupsNames);
            printTable(out, quantiles10thDelta);

            out.printRecords(Collections.singletonList("Quantiles 90th delta"));
            out.printRecord(groupsNames);
            printTable(out, quantiles90thDelta);

            THREAD_POOL.shutdown();
        } catch (IOException e) {
            e.printStackTrace();
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

    private static List<GroupedClassificationResults> analyze(final Instances data) {
        Preconditions.checkNotNull(data, "Dataset can't be null!");

        final List<GroupedClassificationResults> result = new CopyOnWriteArrayList<>();

        LISTED_CLASSIFIERS.stream().forEachOrdered(classifierPair -> {
            final Runnable runnable = () -> {
                try {
                    final ClassifierGroup group = classifierPair.getLeft();
                    final Classifier classifier = classifierPair.getRight();

                    final double[][] ncg = new double[BOOTSTRAP_COUNT][];

                    for (int i = 0; i < BOOTSTRAP_COUNT; i++) {
                        System.out.format("%d of %d for %s complete\n", i, BOOTSTRAP_COUNT,
                                classifier.getClass().getSimpleName());

                        final Instances trainData = data.resample(RANDOM);
                        final Evaluation evaluation = new Evaluation(trainData);

                        classifier.buildClassifier(trainData);
                        evaluation.evaluateModel(classifier, data);
                        ncg[i] = ncg(evaluation);
                    }

                    final List<double[]> zippedNcg = Utils.zip(ncg);
                    final List<Triple<Double, Double, Double>> quantileNcg = extractNcg(zippedNcg);

                    result.add(new GroupedClassificationResults(group, classifier.getClass().getSimpleName(), quantileNcg));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            };

            TASKS.add(THREAD_POOL.submit(runnable));
        });

        waitForTaskCompleting();
        return result;
    }

    private static Map<ClassifierGroup, Triple<
            List<List<Double>>, // quantiles 10th for each classifier
            List<List<Double>>, // quantiles 50th for each classifier
            List<List<Double>>  // quantiles 90th for each classifier
            >> fillClassifierGroupsMap(final int size) {
        final Map<ClassifierGroup, Triple<List<List<Double>>, List<List<Double>>, List<List<Double>>>> result =
                new EnumMap<>(ClassifierGroup.class);

        for (ClassifierGroup group : ClassifierGroup.values()) {
            result.put(group, Triple.of(new ArrayList<>(size), new ArrayList<>(size), new ArrayList<>(size)));
        }

        return result;
    }

    private static Collection<String> getHeaderRow(final Collection<GroupedClassificationResults> result) {
        Preconditions.checkNotNull(result, "Result list can't be null!");
        Preconditions.checkArgument(!result.contains(null), "Result list can't contains null values!");

        final Collection<String> groupsNames = new LinkedList<>();
        groupsNames.add("");

        groupsNames.addAll(result.stream()
                .map(GroupedClassificationResults::getClassifierName)
                .collect(Collectors.toList()));
        return groupsNames;
    }

    private static void printTable(final CSVPrinter out, final List<List<Double>> table) throws IOException {
        Preconditions.checkNotNull(table, "Table can't be null!");
        Preconditions.checkArgument(!table.contains(null), "Table can't contains null values!");

        int count = 1;

        for (List<Double> row : Utils.zipLists(table)) {
            final List<String> line = row.stream()
                    .mapToDouble(i -> i)
                    .mapToObj(Double::toString)
                    .collect(Collectors.toList());
            line.add(0, Integer.toString(count++));
            out.printRecord(line);
        }
    }

    public static double[] ncg(final Evaluation evaluation) {
        Preconditions.checkNotNull(evaluation, "Evaluation can't be null!");

        final List<Double> originalUtility = new LinkedList<>();
        final List<Double> predictedUtility = new LinkedList<>();

        for (int i = 0; i < evaluation.numInstances(); i++) {
            originalUtility.add(((NominalPrediction) evaluation.predictions().elementAt(i)).actual());
            predictedUtility.add(((NominalPrediction) evaluation.predictions().elementAt(i)).predicted());
        }

        return Utils.ncg(predictedUtility, originalUtility).stream().mapToDouble(r -> r).toArray();
    }

    private static List<Triple<Double, Double, Double>> extractNcg(final Collection<double[]> zippedNcg) {
        Preconditions.checkNotNull(zippedNcg, "NCG list can't be null!");
        Preconditions.checkArgument(!zippedNcg.contains(null), "NCG list can't contains null values!");

        final List<Triple<Double, Double, Double>> quantileNcg = new ArrayList<>(zippedNcg.size());

        for (double[] ncgForPostCount : zippedNcg) {
            final double quantile10th = QUANTILE.evaluate(ncgForPostCount, QUANTILE_10_TH);
            final double quantile50th = QUANTILE.evaluate(ncgForPostCount, QUANTILE_50_TH);
            final double quantile90th = QUANTILE.evaluate(ncgForPostCount, QUANTILE_90_TH);
            quantileNcg.add(Triple.of(quantile10th, quantile50th, quantile90th));
        }
        return quantileNcg;
    }

    private static void waitForTaskCompleting() {
        for (Future task : TASKS) {
            try {
                task.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }
}