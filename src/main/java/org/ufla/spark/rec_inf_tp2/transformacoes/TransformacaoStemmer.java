package org.ufla.spark.rec_inf_tp2.transformacoes;

import org.ufla.spark.rec_inf_tp2.funcoes.FunStemmer;

/**
 * Responsável por aplicar a transformação de uma string de uma determinada
 * coluna de um dataset em uma string em que todas as palavras tenham apenas
 * seus radicais (stemmer). Aplica transformação em todo dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class TransformacaoStemmer extends TransformacaoGenerica<FunStemmer> {

	private static final long serialVersionUID = 1L;

	@Override
	protected String getLabeluid() {
		return "TransformacaoStemmer";
	}

	@Override
	public TransformacaoGenerica<FunStemmer> criarTransformacao() {
		return new TransformacaoStemmer();
	}

	@Override
	public FunStemmer criarFuncao(int colunaEntrada, int indiceColSaida) {
		return new FunStemmer(colunaEntrada, indiceColSaida);
	}

}
