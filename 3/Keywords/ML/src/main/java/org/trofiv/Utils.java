package org.trofiv;

import com.google.common.base.Preconditions;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Utils {
    public static List<Double> ncg(final List<Double> predictedUtility, final List<Double> originalUtility) {
        Preconditions.checkNotNull(predictedUtility, "Predicted values list must be provided!");
        Preconditions.checkNotNull(originalUtility, "Original values list must be provided!");
        Preconditions.checkArgument(predictedUtility.size() == originalUtility.size(), "List must have the same size!");
        Preconditions.checkArgument(!predictedUtility.contains(null), "Predicted utilities can't be null");
        Preconditions.checkArgument(!originalUtility.contains(null), "Original utilities can't be null");

        final Collection<Pair<Double, Double>> joined = new LinkedList<>();

        for (int i = 0; i < predictedUtility.size(); i++) {
            joined.add(Pair.of(originalUtility.get(i), predictedUtility.get(i)));
        }

        final List<Double> sortedByOriginal = originalUtility.stream().collect(Collectors.toList());
        sortedByOriginal.sort((o1, o2) -> -o1.compareTo(o2));

        final List<Pair<Double, Double>> sortedByPredicted = joined.stream().collect(Collectors.toList());
        sortedByPredicted.sort((o1, o2) -> -o1.getRight().compareTo(o2.getRight()));

        final List<Double> result = new ArrayList<>(joined.size());

        for (int k = 1; k <= joined.size(); k++) {
            final List<Double> originalShort = sortedByOriginal
                    .stream()
                    .limit(k)
                    .collect(Collectors.toList());
            final List<Double> predictedShort = sortedByPredicted
                    .stream()
                    .limit(k)
                    .mapToDouble(Pair::getLeft)
                    .boxed()
                    .collect(Collectors.toList());

            final double divident = predictedShort.stream().mapToDouble(p -> p).sum();
            final double divisor = originalShort.stream().mapToDouble(p -> p).sum();

            if (divident > divisor) {
                throw new IllegalArgumentException("Divident greater that divisor!");
            }

            result.add(divident / divisor);
        }

        return result;
    }

    public static List<double[]> zip(final double[]... lists) {
        Preconditions.checkNotNull(lists, "Arrays can't be null!");

        final int size = ArrayUtils.nullToEmpty(lists[0]).length;
        final List<double[]> zipped = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            zipped.add(new double[lists.length]);
        }

        for (int i = 0; i < lists.length; i++) {
            final double[] list = lists[i];

            if (list.length != size) {
                throw new IllegalArgumentException("All arrays must have equal length!");
            }

            for (int j = 0; j < size; j++) {
                zipped.get(j)[i] = list[j];
            }
        }
        return zipped;
    }


    public static <T> List<List<T>> zipLists(final List<List<T>> lists) {
        Preconditions.checkNotNull(lists, "Lists can't be null!");
        Preconditions.checkArgument(!lists.contains(null), "List can't be null!");

        final int size = lists.isEmpty() || CollectionUtils.isEmpty(lists.get(0)) ? 0 : lists.get(0).size();
        final List<List<T>> zipped = new ArrayList<>(size);

        for (List<T> list : lists) {
            for (int i = 0, listSize = list.size(); i < listSize; i++) {
                final List<T> list2;

                if (i >= zipped.size()) {
                    zipped.add(list2 = new ArrayList<>(size));
                } else {
                    list2 = zipped.get(i);
                }

                list2.add(list.get(i));
            }
        }
        return zipped;
    }
}
