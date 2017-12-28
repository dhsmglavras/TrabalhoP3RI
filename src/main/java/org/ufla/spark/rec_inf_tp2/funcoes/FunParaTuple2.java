package org.ufla.spark.rec_inf_tp2.funcoes;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import scala.Tuple2;

/**
 * Responsável por converter uma Row em uma tupla com a predição da classe e a
 * classe documento.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class FunParaTuple2 implements MapFunction<Row, Tuple2<Object, Object>> {

	private static final long serialVersionUID = 1L;

	/**
	 * Indice da coluna onde está a predição do classe do documento.
	 */
	private int colunaPredicao;
	/**
	 * Indice da coluna onde está a classe do documento.
	 */
	private int colunaLabel;

	public FunParaTuple2(int colunaPredicao, int colunaLabel) {
		this.colunaPredicao = colunaPredicao;
		this.colunaLabel = colunaLabel;
	}

	/**
	 * Converte a Row em uma tupla com a predição da classe e a classe documento.
	 */
	@Override
	public Tuple2<Object, Object> call(Row row) throws Exception {
		Object predicao = row.get(colunaPredicao);
		Object label = row.get(colunaLabel);
		return new Tuple2<Object, Object>(predicao, label);
	}

}
