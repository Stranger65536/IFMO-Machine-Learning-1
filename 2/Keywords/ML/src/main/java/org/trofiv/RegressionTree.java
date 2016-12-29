package org.trofiv;

import java.util.List;
import java.util.Map;

interface RegressionTree {
    Map<Integer, List<Double>> sse();
}
