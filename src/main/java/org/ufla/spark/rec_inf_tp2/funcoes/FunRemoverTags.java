package org.ufla.spark.rec_inf_tp2.funcoes;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

/**
 * Responsável por transformar a string de uma coluna de uma Row em uma nova
 * string sem tags.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class FunRemoverTags implements MapFunction<Row, Row> {

	private static final long serialVersionUID = 1L;

	/**
	 * Coluna em que deve aplicar o pré-processamento.
	 */
	private int colunaEntrada;
	/**
	 * Coluna em que deve salvar string após o pré-processamento.
	 */
	private int colunaSaida;

	public FunRemoverTags(int colunaEntrada, int colunaSaida) {
		this.colunaEntrada = colunaEntrada;
		this.colunaSaida = colunaSaida;
	}

	/**
	 * Transforma a string de uma coluna de uma Row (colunaEntrada) em uma nova
	 * string sem tags (colunaSaida).
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
		char[] entrada = row.getString(colunaEntrada).toCharArray();
		StringBuilder saida = new StringBuilder();
		boolean tag = false;
		int indTag = 0;
		for (int i = 0; i < entrada.length; i++) {
			if (entrada[i] == '<') {
				saida.append(entrada, indTag, i - indTag);
				indTag = i;
				tag = true;
			} else if (entrada[i] == '>' && tag) {
				indTag = i + 1;
				tag = false;
			} else if (entrada[i] == '\n' && tag) {
				saida.append(entrada, indTag, i + 1 - indTag);
				indTag = i + 1;
				tag = false;
			}
		}
		saida.append(entrada, indTag, entrada.length - indTag);
		campos[colunaSaida] = saida.toString();
		return RowFactory.create(campos);
	}

}
