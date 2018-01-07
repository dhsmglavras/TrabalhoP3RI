package org.ufla.spark.rec_inf_tp2.funcoes;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

/**
 * Responsável por converter uma string de uma coluna em uma string somente com
 * letras minúsculas.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class FunMapMinuscula implements MapFunction<Row, Row> {

	private static final long serialVersionUID = 1L;

	/**
	 * Coluna em que deve aplicar o pré-processamento.
	 */
	private int colunaEntrada;
	/**
	 * Coluna em que deve salvar string após o pré-processamento.
	 */
	private int colunaSaida;

	public FunMapMinuscula(int colunaEntrada, int colunaSaida) {
		this.colunaEntrada = colunaEntrada;
		this.colunaSaida = colunaSaida;
	}

	/**
	 * Aplica a conversão da string em letras minúsculas salva na coluna de saída e
	 * retorna nova linha.
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
		campos[colunaSaida] = row.getString(colunaEntrada).toLowerCase();
		return RowFactory.create(campos);
	}

}
