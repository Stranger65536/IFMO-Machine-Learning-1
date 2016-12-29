public class DirectedEdge {
    private static final String VERTEX_NAMES_MUST_BE_NONNEGATIVE_INTEGERS = "Vertex names must be nonnegative integers";
    private final int v;
    private final int w;
    private final double weight;

    /**
     * Initializes a directed edge from vertex <tt>v</tt> to vertex <tt>w</tt> with
     * the given <tt>weight</tt>.
     *
     * @param v      the tail vertex
     * @param w      the head vertex
     * @param weight the weight of the directed edge
     * @throws IndexOutOfBoundsException if either <tt>v</tt> or <tt>w</tt>
     *                                   is a negative integer
     * @throws IllegalArgumentException  if <tt>weight</tt> is <tt>NaN</tt>
     */
    public DirectedEdge(final int v, final int w, final double weight) {
        if (v < 0) {
            throw new IndexOutOfBoundsException(VERTEX_NAMES_MUST_BE_NONNEGATIVE_INTEGERS);
        }
        if (w < 0) {
            throw new IndexOutOfBoundsException(VERTEX_NAMES_MUST_BE_NONNEGATIVE_INTEGERS);
        }
        if (Double.isNaN(weight)) {
            throw new IllegalArgumentException("Weight is NaN");
        }
        this.v = v;
        this.w = w;
        this.weight = weight;
    }

    /**
     * Returns the tail vertex of the directed edge.
     *
     * @return the tail vertex of the directed edge
     */
    public int from() {
        return v;
    }

    /**
     * Returns the head vertex of the directed edge.
     *
     * @return the head vertex of the directed edge
     */
    public int to() {
        return w;
    }

    /**
     * Returns the weight of the directed edge.
     *
     * @return the weight of the directed edge
     */
    public double weight() {
        return weight;
    }

    /**
     * Returns a string representation of the directed edge.
     *
     * @return a string representation of the directed edge
     */
    public String toString() {
        return v + "->" + w + ' ' + String.format("%5.2f", weight);
    }
}