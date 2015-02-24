package org.itshared.flink.naivebayes;

import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.FileSystem.WriteMode;
import org.apache.flink.util.Collector;

public class NlpPreprocessionJob {

	public static void main(String[] args) throws Exception {
		ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		DataSource<String> inputTrain = env.readTextFile(Config.TRAIN_DATA_RAW);
		DataSet<Tuple2<String, String>> train = inputTrain.flatMap(new NlpProcessingMapper());
		train.writeAsCsv(Config.TRAIN_DATA, "\n", "\t", WriteMode.OVERWRITE);

		DataSource<String> inputTest = env.readTextFile(Config.TEST_DATA_RAW);
		DataSet<Tuple2<String, String>> test = inputTest.flatMap(new NlpProcessingMapper());
		test.writeAsCsv(Config.TRAIN_DATA, "\n", "\t", WriteMode.OVERWRITE);

		env.execute("NLP Preprocession");
	}

	public static class NlpProcessingMapper extends RichFlatMapFunction<String, Tuple2<String, String>> {

		private NlpProcessor processor;

		@Override
		public void open(Configuration parameters) throws Exception {
			super.open(parameters);
			processor = NlpProcessor.create();
		}

		@Override
		public void flatMap(String value, Collector<Tuple2<String, String>> out) throws Exception {
			String[] split = value.split("\t");
			if (split.length < 3) {
				return;
			}

			String category = split[0];

			List<String> words = processor.processBody(split[2]);
			if (!words.isEmpty()) {
				out.collect(new Tuple2<>(category, StringUtils.join(words, ",")));
			}
		}

	}
}
