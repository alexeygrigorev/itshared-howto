package org.itshared.flink.naivebayes;

import java.util.List;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.tuple.Tuple1;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.FileSystem.WriteMode;
import org.apache.flink.util.Collector;

public class WordCounterJob {

	public static void main(String[] args) throws Exception {
		ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		DataSource<String> input = env.readTextFile(Config.TRAIN_DATA);

		// estimate P(class=c)
		DataSet<Tuple2<String, Long>> labelFrequencies = 
				input.map(new LabelExtraction()).groupBy(0).sum(1);
		DataSet<Tuple1<Long>> totalSum = labelFrequencies.sum(1).project(1);

		DataSet<Tuple2<String, Double>> priors = 
				labelFrequencies.map(new NormalizationMapper()).withBroadcastSet(totalSum, "totalSum");

		// counts to estimate P(word | class=c)
		DataSet<Tuple3<String, String, Integer>> labelledWords = 
				input.flatMap(new TokenReaderMapper());
		DataSet<Tuple3<String, String, Integer>> wordCount = 
				labelledWords.groupBy(1, 0).sum(2);
		
		priors.writeAsCsv(Config.OUT_PRIOR, "\n", "\t", WriteMode.OVERWRITE);
		wordCount.writeAsCsv(Config.OUT_COND_COUNT, "\n", "\t", WriteMode.OVERWRITE);
		
		env.execute("Naive Bayes Job");
	}

	public static class NormalizationMapper extends
			RichMapFunction<Tuple2<String, Long>, Tuple2<String, Double>> {
		private long totalSum;
		
		@Override
		public void open(Configuration parameters) throws Exception {
			super.open(parameters);
			List<Tuple1<Long>> totalSumList = getRuntimeContext().getBroadcastVariable("totalSum");
			this.totalSum = totalSumList.get(0).f0;
		}

		@Override
		public Tuple2<String, Double> map(Tuple2<String, Long> value) throws Exception {
			return new Tuple2<>(value.f0, ((double) value.f1) / totalSum);
		}
	}

	public static class LabelExtraction implements MapFunction<String, Tuple2<String, Long>> {
		@Override
		public Tuple2<String, Long> map(String value) throws Exception {
			return new Tuple2<>(value.split("\t")[0], 1L);
		}
	}

	public static class TokenReaderMapper implements 
				FlatMapFunction<String, Tuple3<String, String, Integer>> {

		@Override
		public void flatMap(String inputValue, Collector<Tuple3<String, String, Integer>> out)
				throws Exception {
			String[] split = inputValue.split("\t");
			String category = split[0];

			for (String word : split[1].split(",")) {
				out.collect(new Tuple3<>(category, word, 1));
			}
		}

	}

}
