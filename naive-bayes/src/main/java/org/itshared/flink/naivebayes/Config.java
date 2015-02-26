package org.itshared.flink.naivebayes;

public class Config {
	private static final String FOLDER = "C:/tmp/20news-bydate/";
	public static final String TRAIN_DATA_RAW = FOLDER + "train.tab";
	public static final String TEST_DATA_RAW = FOLDER + "test.tab";

	public static final String TRAIN_DATA = FOLDER + "train/";
	public static final String TEST_DATA = FOLDER + "test/";

	private static final String OUT_FOLDER = FOLDER + "OUT/";
	public static final String OUT_PRIOR = OUT_FOLDER + "prior/";
	public static final String OUT_COND_COUNT = OUT_FOLDER + "cond/";
	public static final String OUT_TOTAL_COUNT_PER_CAT = OUT_FOLDER + "countPerCat/";
}
