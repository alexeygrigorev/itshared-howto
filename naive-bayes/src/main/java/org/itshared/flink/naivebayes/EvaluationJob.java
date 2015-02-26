package org.itshared.flink.naivebayes;

import java.util.List;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;

public class EvaluationJob {

	public static void main(String[] args) throws Exception {
		ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		DataSet<Tuple2<String, Double>> priors = 
				env.readTextFile(Config.OUT_PRIOR).map(new PriorsReaderMapper());
		DataSet<Tuple2<String, Integer>> totalCountPerCategory = 
				env.readTextFile(Config.OUT_TOTAL_COUNT_PER_CAT).map(new TotalCountPerCategoryMapper());
		DataSet<Tuple3<String, String, Integer>> counts = 
				env.readTextFile(Config.OUT_COND_COUNT).map(new CountsReaderMapper());

		DataSource<String> testSet = env.readTextFile(Config.TEST_DATA);

		int smoothing = 1;
		DataSet<Tuple2<String, String>> predictions = testSet.map(new ClassifierMapper(smoothing))
				.withBroadcastSet(totalCountPerCategory, "countPerCategory")
				.withBroadcastSet(priors, "priors")
				.withBroadcastSet(counts, "counts");

		predictions.map(new MatchMapper())
					.reduce(new PerformaceEvaluatorReducer())
					.map(new PerformaceMapper())
					.print();
		
		env.execute("Evaluation Job");
	}

	public static class MatchMapper implements MapFunction<Tuple2<String, String>, Tuple2<Integer, Integer>> {
		@Override
		public Tuple2<Integer, Integer> map(Tuple2<String, String> value) throws Exception {
			if (value.f0.equals(value.f1)) {
				return new Tuple2<>(1, 1);
			} else {
				return new Tuple2<>(0, 1);
			}
		}

	}

	public static class PerformaceEvaluatorReducer implements ReduceFunction<Tuple2<Integer, Integer>> {
		@Override
		public Tuple2<Integer, Integer> reduce(Tuple2<Integer, Integer> value1,
				Tuple2<Integer, Integer> value2) throws Exception {
			return new Tuple2<>(value1.f0 + value2.f0, value1.f1 + value2.f1);
		}
	}

	public static class PerformaceMapper implements MapFunction<Tuple2<Integer, Integer>, Double> {
		@Override
		public Double map(Tuple2<Integer, Integer> value) throws Exception {
			return (double) value.f0 / (double) value.f1;
		}
	}

	public static class ClassifierMapper extends RichMapFunction<String, Tuple2<String, String>> {
		private int smooting;
		private NaiveBayesClassifier classifier;

		public ClassifierMapper(int smooting) {
			this.smooting = smooting;
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			super.open(parameters);
			RuntimeContext ctx = getRuntimeContext();
			List<Tuple2<String, Integer>> countPerCategory = ctx.getBroadcastVariable("countPerCategory");
			List<Tuple2<String, Double>> priors = ctx.getBroadcastVariable("priors");
			List<Tuple3<String, String, Integer>> counts = ctx.getBroadcastVariable("counts");

			classifier = new NaiveBayesClassifier(smooting);
			classifier.init(priors, counts, countPerCategory);
		}

		@Override
		public Tuple2<String, String> map(String value) throws Exception {
			String[] split = value.split("\t");
			String actualLabel = split[0];

			String[] words = split[1].split(",");
			String predictedLabel = classifier.predict(words);

			return new Tuple2<>(actualLabel, predictedLabel);
		}
	}

	public static class PriorsReaderMapper implements MapFunction<String, Tuple2<String, Double>> {
		@Override
		public Tuple2<String, Double> map(String value) throws Exception {
			String[] split = value.split("\t");
			return new Tuple2<>(split[0], Double.valueOf(split[1]));
		}
	}

	public static class TotalCountPerCategoryMapper implements MapFunction<String, Tuple2<String, Integer>> {
		@Override
		public Tuple2<String, Integer> map(String value) throws Exception {
			String[] split = value.split("\t");
			return new Tuple2<>(split[0], Integer.valueOf(split[1]));
		}
	}
	
	public static class CountsReaderMapper implements MapFunction<String, Tuple3<String, String, Integer>> {
		@Override
		public Tuple3<String, String, Integer> map(String value) throws Exception {
			String[] split = value.split("\t");
			return new Tuple3<>(split[0], split[1], Integer.valueOf(split[2]));
		}
	}

}
