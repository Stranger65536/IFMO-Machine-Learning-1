package org.trofiv;

import org.apache.commons.lang3.tuple.Triple;

import java.util.List;

public class GroupedClassificationResults extends ClassificationResults {
    private final ClassifierGroup classifierGroup;

    public GroupedClassificationResults(
            final ClassifierGroup classifierGroup,
            final String classifierName,
            final List<Triple<Double, Double, Double>> quantiles) {
        super(classifierName, quantiles);
        this.classifierGroup = classifierGroup;
    }

    public ClassifierGroup getClassifierGroup() {
        return classifierGroup;
    }
}