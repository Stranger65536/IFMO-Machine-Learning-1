package org.trofiv;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FilenameUtils;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

@SuppressWarnings("ThrowCaughtLocally")
public class Main {
    private static final String INPUT_DIRECTORIES_LOCATION = "../";
    private static final Pattern INPUT_FILES = Pattern.compile(".*_soc_graph.*normalized.tsv");
    private static final Pattern INPUT_DIRECTORIES = Pattern.compile("keywords.*");
    private static final String FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW = "File must contains more that one row!";

    @SuppressWarnings({"ImplicitDefaultCharsetUsage", "ConstantConditions"})
    public static void main(final String[] args) {
        try {
            final File workDirectory = Paths.get(INPUT_DIRECTORIES_LOCATION).toRealPath().toFile();
            final File[] keywordDirectories = workDirectory.listFiles(
                    path -> path.isDirectory() && INPUT_DIRECTORIES.matcher(path.getName()).matches());

            if (keywordDirectories.length == 0) {
                throw new IllegalArgumentException("No one input directory found!");
            }

            for (File dir : keywordDirectories) {
                final File[] fileToProcess = dir.listFiles(
                        path -> path.isFile() && INPUT_FILES.matcher(path.getName()).matches() &&
                                !path.getName().contains("betweenness"));

                if (fileToProcess.length == 0) {
                    throw new IllegalArgumentException("No one input file found!");
                }

                for (File file : fileToProcess) {
                    processFile(file);
                }

            }
        } catch (IOException e) {
            System.out.println("Can't access working directory or missing social graph files");
            System.out.println(e.getLocalizedMessage());
        }
    }

    private static void processFile(final File file) {
        final String fileName = FilenameUtils.getBaseName(file.toString());
        final String extension = FilenameUtils.getExtension(file.toString());

        //noinspection ImplicitDefaultCharsetUsage
        try (Reader in = new FileReader(file);
             CSVPrinter out = CSVFormat.MYSQL.print(
                     new PrintWriter(file.getParent() + File.separator + "betweenness_" + fileName + '.' + extension))) {
            final GraphWithAuthors graphWithAuthors = new GraphWithAuthors(in).invoke();
            final AdjMatrixEdgeWeightedDigraph graph = graphWithAuthors.getGraph();
            final List<String> authors = graphWithAuthors.getAuthors();

            final FloydWarshall floydWarshall = new FloydWarshall(graph);

            if (file.getName().contains("sentim")) {
                final Map<Integer, Double> betweenness = new LinkedHashMap<>(authors.size(), 1);
                IntStream.range(0, authors.size()).forEach(i -> betweenness.put(i, 0.0));
                calculateBetweennessSent(floydWarshall, betweenness);

                for (int i = 0; i < authors.size(); i++) {
                    out.printRecord(authors.get(i), betweenness.get(i));
                }
            } else {
                final Map<Integer, Integer> betweenness = new LinkedHashMap<>(authors.size(), 1);
                IntStream.range(0, authors.size()).forEach(i -> betweenness.put(i, 0));
                calculateBetweenness(floydWarshall, betweenness);

                for (int i = 0; i < authors.size(); i++) {
                    out.printRecord(authors.get(i), betweenness.get(i));
                }
            }
        } catch (IOException e) {
            System.out.println("Can't process graph file!");
            System.out.println(e.getLocalizedMessage());
        }
    }

    private static void calculateBetweenness(
            final FloydWarshall floydWarshall,
            final Map<Integer, Integer> betweenness) {
        for (int from = 0; from < betweenness.size(); from++) {
            for (int to = 0; to < betweenness.size(); to++) {
                final List<DirectedEdge> path = (Stack<DirectedEdge>) floydWarshall.path(from, to);

                if (path != null && !path.isEmpty()) {
                    Collections.reverse(path);
                    final Collection<Integer> edges = new HashSet<>(path.size() * 2, 1);

                    path.stream().forEach(directedEdge -> {
                        edges.add(directedEdge.from());
                        edges.add(directedEdge.to());
                    });

                    edges.stream().forEach(edge -> {
                        final int old = betweenness.get(edge);
                        betweenness.put(edge, old + 1);
                    });
                }
            }
        }
    }

    private static void calculateBetweennessSent(
            final FloydWarshall floydWarshall,
            final Map<Integer, Double> betweenness) {
        for (int from = 0; from < betweenness.size(); from++) {
            for (int to = 0; to < betweenness.size(); to++) {
                final List<DirectedEdge> path = (Stack<DirectedEdge>) floydWarshall.path(from, to);

                if (path != null && !path.isEmpty()) {
                    Collections.reverse(path);

                    path.stream().forEach(directedEdge -> {
                        final double oldFrom = betweenness.get(directedEdge.from());
                        final double oldTo = betweenness.get(directedEdge.to());

                        betweenness.put(directedEdge.from(), oldFrom + directedEdge.weight());
                        betweenness.put(directedEdge.to(), oldTo + directedEdge.weight());
                    });
                }
            }
        }
    }

    @SuppressWarnings("InstanceVariableMayNotBeInitialized")
    private static class GraphWithAuthors {
        private final Reader in;
        private List<String> authors;
        private AdjMatrixEdgeWeightedDigraph graph;

        public GraphWithAuthors(final Reader in) {
            this.in = in;
        }

        public List<String> getAuthors() {
            return authors;
        }

        public AdjMatrixEdgeWeightedDigraph getGraph() {
            return graph;
        }

        public GraphWithAuthors invoke() throws IOException {
            final List<CSVRecord> records = CSVFormat.MYSQL.parse(in).getRecords();

            if (records.size() < 1) {
                throw new IllegalArgumentException(FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW);
            }

            final Iterator<CSVRecord> it = records.iterator();
            final CSVRecord firstLine = it.next();

            if (firstLine.size() < 1) {
                throw new IllegalArgumentException(FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW);
            }

            final Iterator<String> firstRowIt = firstLine.iterator();
            firstRowIt.next();

            authors = new ArrayList<>(firstLine.size());
            while (firstRowIt.hasNext()) {
                authors.add(firstRowIt.next());
            }

            graph = new AdjMatrixEdgeWeightedDigraph(authors.size());

            int row = 0;
            while (it.hasNext()) {
                final Iterator<String> rowIt = it.next().iterator();
                rowIt.next();

                int column = 0;
                while (rowIt.hasNext()) {
                    final double value = Double.valueOf(rowIt.next());

                    if (row == column) {
                        graph.addEdge(new DirectedEdge(row, column, 0));
                    } else {
                        if (value >= Integer.MAX_VALUE) {
                            graph.addEdge(new DirectedEdge(row, column, Double.POSITIVE_INFINITY));
                        } else {
                            graph.addEdge(new DirectedEdge(row, column, value));
                        }
                    }
                    column++;
                }
                row++;
            }
            return this;
        }
    }
}