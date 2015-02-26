package org.itshared.flink.naivebayes;

import java.util.List;
import java.util.Map;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Table;

public class NaiveBayesClassifier {

	private final int smoothing;

	private final Map<String, Double> priors = Maps.newHashMap();
	private final Map<String, Integer> countPerCategory = Maps.newHashMap();
	private final Table<String, String, Integer> conditionalWordCounts = HashBasedTable.create();
	private final List<String> labels = Lists.newArrayList();

	private int distinctWordCount;

	public NaiveBayesClassifier(int smoothing) {
		this.smoothing = smoothing;
	}

	public String predict(String[] words) {
		double maxLog = Double.NEGATIVE_INFINITY;

		String predictionLabel = "";
		for (String label : labels) {
			double logProb = calculateLogP(label, words);
			if (logProb > maxLog) {
				maxLog = logProb;
				predictionLabel = label;
			}
		}

		return predictionLabel;
	}

	private double calculateLogP(String label, String[] words) {
		double logProb = priors.get(label);
		Map<String, Integer> countsPerLabel = conditionalWordCounts.row(label);

		// numerator terms
		for (String word : words) {
			Integer count = countsPerLabel.get(word);
			if (count != null) {
				logProb = logProb + Math.log(count + smoothing);
			} else {
				logProb = logProb + Math.log(smoothing);
			}
		}

		// denominator terms
		double denom = countPerCategory.get(label) + smoothing * distinctWordCount;
		logProb = logProb - words.length * Math.log(denom);
		return logProb;
	}

	public void init(List<Tuple2<String, Double>> priors,
			List<Tuple3<String, String, Integer>> conditionalWordCounts, 
			List<Tuple2<String, Integer>> countPerCategory) {
		initCountPerCategory(countPerCategory);
		initConditionalWordCounts(conditionalWordCounts);
		initPriors(priors);
	}

	private void initCountPerCategory(List<Tuple2<String, Integer>> input) {
		for (Tuple2<String, Integer> tuple : input) {
			countPerCategory.put(tuple.f0, tuple.f1);
		}
	}

	private void initPriors(List<Tuple2<String, Double>> input) {
		for (Tuple2<String, Double> tuple : input) {
			priors.put(tuple.f0, Math.log(tuple.f1));
		}
	}

	private void initConditionalWordCounts(List<Tuple3<String, String, Integer>> input) {
		for (Tuple3<String, String, Integer> tuple : input) {
			conditionalWordCounts.put(tuple.f0, tuple.f1, tuple.f2);
		}
		labels.addAll(conditionalWordCounts.rowKeySet());
		distinctWordCount = conditionalWordCounts.columnKeySet().size();
	}

}
