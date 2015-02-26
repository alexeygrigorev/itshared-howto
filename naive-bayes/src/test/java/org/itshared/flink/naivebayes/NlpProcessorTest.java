package org.itshared.flink.naivebayes;

import static org.junit.Assert.*;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.junit.Test;

public class NlpProcessorTest {

	NlpPreprocessor processor = NlpPreprocessor.create();
	
	@Test
	public void test() throws Exception {
		InputStream testStream = NlpProcessorTest.class.getResourceAsStream("test.txt");
		List<String> testData = IOUtils.readLines(testStream);
		String first = testData.get(0);
		String[] split = first.split("\t");

		List<String> result = processor.processBody(split[2]);
		System.out.println(result);
		assertTrue(result.size() > 0);
	}
	
	@Test
	public void testBadSymbols() throws Exception {
		String body = "~~~~~~~~ Hello there ~~~~~~~~ !!!";
		List<String> result = processor.processBody(body);
		System.out.println(result);
		assertTrue(result.size() > 0);
	}

	@Test
	public void testStopWordsRemoved() throws Exception {
		String body = "Do not use a phone.";
		List<String> result = processor.processBody(body);
		System.out.println(result);
		assertEquals(Arrays.asList("use", "phone"), result);
	}

	@Test
	public void testCapitalStopwords() throws Exception {
		InputStream testStream = NlpProcessorTest.class.getResourceAsStream("test.txt");
		List<String> testData = IOUtils.readLines(testStream);
		String first = testData.get(2);
		String[] split = first.split("\t");

		List<String> result = processor.processBody(split[2]);
		assertFalse(result.contains("do"));
	}
	
	@Test
	public void valid() {
		assertTrue(processor.valid("processor"));
		assertFalse(processor.valid("do"));
		assertFalse(processor.valid("don't"));
		assertFalse(processor.valid("some@email.com"));
		assertFalse(processor.valid("123"));
	}
}
