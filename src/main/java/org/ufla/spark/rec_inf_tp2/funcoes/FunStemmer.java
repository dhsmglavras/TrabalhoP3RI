package org.ufla.spark.rec_inf_tp2.funcoes;

import java.io.StringReader;
import java.util.List;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.ufla.spark.rec_inf_tp2.utils.Stemmer;

/**
 * Responsável por transformar a string de uma coluna de uma Row em uma nova
 * string onde as palavras tenham apenas seus radicais (stemmer).
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class FunStemmer implements MapFunction<Row, Row> {

	private static final long serialVersionUID = 1L;

	/**
	 * Coluna em que deve aplicar o pré-processamento.
	 */
	private int colunaEntrada;
	/**
	 * Coluna em que deve salvar string após o pré-processamento.
	 */
	private int colunaSaida;;

	public FunStemmer(int colunaEntrada, int colunaSaida) {
		this.colunaEntrada = colunaEntrada;
		this.colunaSaida = colunaSaida;
	}

	/**
	 * Transforma a string de uma coluna de uma Row (colunaEntrada) em uma nova
	 * string onde as palavras tenham apenas seus radicais (stemmer) (colunaSaida).
	 */
	@Override
	public Row call(Row row) throws Exception {
		int n = row.length();
		if (n == colunaSaida) {
			n++;
		}
		Object[] campos = new Object[n];
		for (int i = 0; i < row.length(); i++) {
			campos[i] = row.get(i);
		}
		String entrada = row.getString(colunaEntrada);
		StringReader stringReader = new StringReader(entrada);
		Stemmer stemmer = new Stemmer();
		List<String> radicais = stemmer.runStem(stringReader);
		StringBuilder saida = new StringBuilder();
		for (String radical : radicais) {
			saida.append(radical).append(' ');
		}
		//saida.deleteCharAt(saida.length() - 1);
		campos[colunaSaida] = saida.toString();
		return RowFactory.create(campos);
	}

}
