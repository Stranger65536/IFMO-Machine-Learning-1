package org.trofiv;

public enum ClassifierGroup {
    DECISION_TREES("Decision tree based classifiers"),
    BOOSTING("Boosting"),
    REGRESSION("Regression based classifiers"),
    NEAREST_NEIGHBOURS("Nearest neighbours based classifiers"),
    META("Meta classifiers"),
    SVM("Support vector machine classifiers"),
    HYPER_PIPES("Hyper pipes classifiers"),
    VFI("Voting Feature Intervals classifiers"),
    CLUSTERING("Clustering based classifiers");

    private final String value;

    ClassifierGroup(final String s) {
        value = s;
    }

    public String getValue() {
        return value;
    }
}
