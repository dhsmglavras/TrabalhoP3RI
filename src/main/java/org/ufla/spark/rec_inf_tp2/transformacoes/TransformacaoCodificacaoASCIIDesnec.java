package org.ufla.spark.rec_inf_tp2.transformacoes;

import org.ufla.spark.rec_inf_tp2.funcoes.FunCodificacaoASCIIDesnec;

/**
 * Responsável por aplicar a transformação de uma string de uma determinada
 * coluna de um dataset em uma string codificada em ASCII e sem símbolos
 * desnecessários, apenas com letras. Aplica transformação em todo dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class TransformacaoCodificacaoASCIIDesnec extends TransformacaoGenerica<FunCodificacaoASCIIDesnec> {

	private static final long serialVersionUID = 1L;

	@Override
	protected String getLabeluid() {
		return "TransformacaoCodificacaoASCIIDesnec";
	}

	@Override
	public TransformacaoGenerica<FunCodificacaoASCIIDesnec> criarTransformacao() {
		return new TransformacaoCodificacaoASCIIDesnec();
	}

	@Override
	public FunCodificacaoASCIIDesnec criarFuncao(int indiceColEntrada, int indiceColSaida) {
		return new FunCodificacaoASCIIDesnec(indiceColEntrada, indiceColSaida);
	}

}
