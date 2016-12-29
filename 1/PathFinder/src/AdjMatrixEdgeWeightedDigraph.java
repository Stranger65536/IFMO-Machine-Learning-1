import java.util.Iterator;
import java.util.NoSuchElementException;

@SuppressWarnings("ALL")
public class AdjMatrixEdgeWeightedDigraph {
    private static final String NEWLINE = System.getProperty("line.separator");

    private final int v;
    private final DirectedEdge[][] adj;
    private int e;

    /**
     * Initializes an empty edge-weighted digraph with <tt>v</tt> vertices and 0 edges.
     * param v the number of vertices
     *
     * @throws IllegalArgumentException if <tt>v</tt> < 0
     */
    public AdjMatrixEdgeWeightedDigraph(final int V) {
        if (V < 0) {
            //noinspection ProhibitedExceptionThrown
            throw new RuntimeException("Number of vertices must be nonnegative");
        }
        this.v = V;
        this.e = 0;
        this.adj = new DirectedEdge[V][V];
    }

    /**
     * Initializes a random edge-weighted digraph with <tt>v</tt> vertices and <em>e</em> edges.
     * param v the number of vertices
     * param e the number of edges
     *
     * @throws IllegalArgumentException if <tt>v</tt> < 0
     * @throws IllegalArgumentException if <tt>e</tt> < 0
     */
    public AdjMatrixEdgeWeightedDigraph(final int V, final int E) {
        this(V);
        if (E < 0) {
            //noinspection ProhibitedExceptionThrown
            throw new RuntimeException("Number of edges must be nonnegative");
        }
        if (E > V * V) {
            //noinspection ProhibitedExceptionThrown
            throw new RuntimeException("Too many edges");
        }

        // can be inefficient
        while (this.e != E) {
            int v = StdRandom.uniform(V);
            int w = StdRandom.uniform(V);
            double weight = Math.round(100 * StdRandom.uniform()) / 100.0;
            addEdge(new DirectedEdge(v, w, weight));
        }
    }

    /**
     * Returns the number of vertices in the edge-weighted digraph.
     *
     * @return the number of vertices in the edge-weighted digraph
     */
    public int V() {
        return v;
    }

    /**
     * Returns the number of edges in the edge-weighted digraph.
     *
     * @return the number of edges in the edge-weighted digraph
     */
    public int E() {
        return e;
    }

    /**
     * Adds the directed edge <tt>e</tt> to the edge-weighted digraph (if there
     * is not already an edge with the same endpoints).
     *
     * @param e the edge
     */
    public void addEdge(final DirectedEdge e) {
        final int v = e.from();
        final int w = e.to();
        if (adj[v][w] == null) {
            this.e++;
            adj[v][w] = e;
        }
    }

    /**
     * Returns the directed edges incident from vertex <tt>v</tt>.
     *
     * @param v the vertex
     * @return the directed edges incident from vertex <tt>v</tt> as an Iterable
     * @throws IndexOutOfBoundsException unless 0 <= v < v
     */
    public Iterable<DirectedEdge> adj(final int v) {
        return new AdjIterator(v);
    }

    /**
     * Returns a string representation of the edge-weighted digraph. This method takes
     * time proportional to <em>v</em><sup>2</sup>.
     *
     * @return the number of vertices <em>v</em>, followed by the number of edges <em>e</em>,
     * followed by the <em>v</em> adjacency lists of edges
     */
    public String toString() {
        final StringBuilder s = new StringBuilder();
        s.append(v + " " + e + NEWLINE);
        for (int v = 0; v < this.v; v++) {
            s.append(v + ": ");
            for (DirectedEdge e : adj(v)) {
                s.append(e + "  ");
            }
            s.append(NEWLINE);
        }
        return s.toString();
    }

    // support iteration over graph vertices
    @SuppressWarnings({"NewExceptionWithoutArguments", "ReturnOfInnerClass", "InnerClassFieldHidesOuterClassField", "InstanceVariableMayNotBeInitialized"})
    private class AdjIterator implements Iterator<DirectedEdge>, Iterable<DirectedEdge> {
        private final int v;
        private int w;

        public AdjIterator(final int v) {
            this.v = v;
        }

        @Override
        public Iterator<DirectedEdge> iterator() {
            return this;
        }

        @Override
        public boolean hasNext() {
            while (w < AdjMatrixEdgeWeightedDigraph.this.v) {
                if (adj[v][w] != null) {
                    return true;
                }
                w++;
            }
            return false;
        }

        @Override
        public DirectedEdge next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return adj[v][w++];
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}