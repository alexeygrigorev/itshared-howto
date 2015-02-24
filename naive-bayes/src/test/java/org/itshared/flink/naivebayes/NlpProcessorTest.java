package org.itshared.flink.naivebayes;

import static org.junit.Assert.assertTrue;

import java.io.InputStream;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.junit.Test;

public class NlpProcessorTest {

	@Test
	public void test() throws Exception {
		InputStream testStream = NlpProcessorTest.class.getResourceAsStream("test.txt");
		List<String> testData = IOUtils.readLines(testStream);
		String first = testData.get(0);
		String[] split = first.split("\t");

		NlpProcessor processor = NlpProcessor.create();

		List<String> result = processor.processBody(split[2]);
		System.out.println(result);
		assertTrue(result.size() > 0);
	}
}
