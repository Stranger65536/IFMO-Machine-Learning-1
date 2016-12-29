import com.sun.media.sound.InvalidFormatException;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FilenameUtils;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.IntStream;

@SuppressWarnings("ThrowCaughtLocally")
public class Main {
    public static final List<Path> FILES = Arrays.asList(
            Paths.get("stat_post_1_naive_quoted_soc_graph_clean_normalized.tsv"),
            Paths.get("stat_post_1_naive_quoted_soc_graph_sentim_normalized.tsv"),
            Paths.get("stat_post_1_soc_graph_clean_normalized.tsv"),
            Paths.get("stat_post_1_soc_graph_sentim_normalized.tsv"),
            Paths.get("stat_post_2_naive_quoted_soc_graph_clean_normalized.tsv"),
            Paths.get("stat_post_2_naive_quoted_soc_graph_sentim_normalized.tsv"),
            Paths.get("stat_post_2_soc_graph_clean_normalized.tsv"),
            Paths.get("stat_post_2_soc_graph_sentim_normalized.tsv")
    );
    private static final String FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW = "File must contains more that one row!";

    @SuppressWarnings({"ImplicitDefaultCharsetUsage", "ConstantConditions"})
    public static void main(final String[] args) {
        for (Path path : FILES) {
            final String fileName = FilenameUtils.getBaseName(path.toString());
            final String extension = FilenameUtils.getExtension(path.toString());

            try (Reader in = new FileReader(path.toFile());
                 CSVPrinter out = CSVFormat.MYSQL.print(new PrintWriter(fileName + "_betweenness" + '.' + extension))) {
                final GraphWithAuthors graphWithAuthors = new GraphWithAuthors(in).invoke();
                final AdjMatrixEdgeWeightedDigraph graph = graphWithAuthors.getGraph();
                final List<String> authors = graphWithAuthors.getAuthors();

                final FloydWarshall floydWarshall = new FloydWarshall(graph);

                if (path.getFileName().toString().contains("sentim")) {
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
                e.printStackTrace();
            }
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
                        int old = betweenness.get(edge);
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
                        double oldFrom = betweenness.get(directedEdge.from());
                        double oldTo = betweenness.get(directedEdge.to());

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
                throw new InvalidFormatException(FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW);
            }

            final Iterator<CSVRecord> it = records.iterator();
            final CSVRecord firstLine = it.next();

            if (firstLine.size() < 1) {
                throw new InvalidFormatException(FILE_MUST_CONTAINS_MORE_THAT_ONE_ROW);
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