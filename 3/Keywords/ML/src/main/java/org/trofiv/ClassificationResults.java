package org.trofiv;

import org.apache.commons.lang3.tuple.Triple;

import java.util.List;

public class ClassificationResults {
    protected final String classifierName;
    protected final List<Triple<Double, Double, Double>> quantiles;

    public ClassificationResults(
            final String classifierName,
            final List<Triple<Double, Double, Double>> quantiles) {
        this.classifierName = classifierName;
        this.quantiles = quantiles;
    }

    public String getClassifierName() {
        return classifierName;
    }

    public List<Triple<Double, Double, Double>> getQuantiles() {
        return quantiles;
    }
}