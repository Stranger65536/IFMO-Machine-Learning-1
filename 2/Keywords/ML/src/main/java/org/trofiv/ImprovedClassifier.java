package org.trofiv;

import weka.core.Instances;

interface ImprovedClassifier {
    @SuppressWarnings("ProhibitedExceptionDeclared")
    void evaluate(final Instances train, final Instances test) throws Exception;

    double[] attributeQuality();

    @SuppressWarnings("ProhibitedExceptionDeclared")
    double[] ncg() throws Exception;
}