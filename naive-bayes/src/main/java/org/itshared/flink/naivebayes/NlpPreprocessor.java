package org.itshared.flink.naivebayes;

import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Properties;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;

import com.google.common.base.Throwables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class NlpPreprocessor {

	private StanfordCoreNLP pipeline;
	private HashSet<String> stopwords;

	private NlpPreprocessor() {
	}

	public static NlpPreprocessor create() {
		try {
			Properties props = new Properties();
			props.put("annotators", "tokenize, ssplit, pos, lemma");
			StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

			NlpPreprocessor nlpProcessor = new NlpPreprocessor();
			nlpProcessor.pipeline = pipeline;

			InputStream stopwordsStream = NlpPreprocessor.class.getResourceAsStream("en-stopwords.txt");
			nlpProcessor.stopwords = Sets.newHashSet(IOUtils.readLines(stopwordsStream));

			return nlpProcessor;
		} catch (Exception e) {
			throw Throwables.propagate(e);
		}
	}

	public List<String> processBody(String input) {
		String body = removeBadSymbols(input);
		Annotation document = new Annotation(body);
		pipeline.annotate(document);

		List<CoreLabel> tokenized = document.get(TokensAnnotation.class);
		List<String> result = Lists.newArrayListWithCapacity(tokenized.size());

		for (CoreLabel token : tokenized) {
			String lemma = token.get(LemmaAnnotation.class);
			String lowerCaseLemma = lemma.toLowerCase(Locale.ENGLISH);
			if (valid(lowerCaseLemma)) {
				result.add(lowerCaseLemma);
			}
		}

		return result;
	}

	public boolean valid(String lemma) {
		if (lemma.length() < 2) {
			return false;
		}

		if (stopwords.contains(lemma)) {
			return false;
		}

		return StringUtils.isAlpha(lemma);
	}

	private static String removeBadSymbols(String body) {
		return body.replaceAll("[~^=<>&\\_/]", "");
	}

}
