package org.ufla.spark.rec_inf_tp2.transformacoes;

import org.ufla.spark.rec_inf_tp2.funcoes.FunMapMinuscula;

/**
 * Responsável por aplicar a transformação de uma string de uma determinada
 * coluna de um dataset em uma string somente com letras minúsculas. Aplica
 * transformação em todo dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class TransformacaoMinuscula extends TransformacaoGenerica<FunMapMinuscula> {

	private static final long serialVersionUID = 1L;

	@Override
	protected String getLabeluid() {
		return "TransformacaoMinuscula";
	}

	@Override
	public TransformacaoGenerica<FunMapMinuscula> criarTransformacao() {
		return new TransformacaoMinuscula();
	}

	@Override
	public FunMapMinuscula criarFuncao(int indiceColEntrada, int indiceColSaida) {
		return new FunMapMinuscula(indiceColEntrada, indiceColSaida);
	}

}
