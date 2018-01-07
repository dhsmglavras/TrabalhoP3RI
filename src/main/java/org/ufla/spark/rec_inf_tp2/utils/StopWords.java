package org.ufla.spark.rec_inf_tp2.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Responsável por ler as stopwords do arquivo stoplist.txt.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class StopWords {

	/**
	 * Arquivo com as stopwords
	 */
	private static String ARQUIVO_STOP_WORDS = "stoplist.txt";
	/**
	 * Leitor bufferizado para o arquivo.
	 */
	private BufferedReader in;

	/**
	 * Recupera um array com as stopwords.
	 * 
	 * @return array com as stopwords
	 * @throws IOException
	 */
	public String[] getStopWords() throws IOException {
		List<String> stopWords = new ArrayList<>();
		FileReader file = new FileReader(ARQUIVO_STOP_WORDS);
		in = new BufferedReader(file);
		String linha;
		while ((linha = in.readLine()) != null) {
			String[] tokens = linha.split("\\s");
			for (String token : tokens) {
				if (token.isEmpty()) {
					continue;
				}
				stopWords.add(token);
			}
		}
		in.close();
		return stopWords.toArray(new String[0]);
	}

	/**
	 * Recupera um conjunto com as stopwords.
	 * 
	 * @return conjunto com as stopwords.
	 * @throws IOException
	 */
	public Set<String> getSetStopWords() throws IOException {
		return new HashSet<>(Arrays.asList(getStopWords()));
	}
}
