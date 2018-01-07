package org.ufla.spark.rec_inf_tp2.transformacoes;

import java.io.IOException;
import java.util.Set;

import org.ufla.spark.rec_inf_tp2.funcoes.FunRemocaoStopWords;
import org.ufla.spark.rec_inf_tp2.funcoes.FunStemmer;
import org.ufla.spark.rec_inf_tp2.utils.StopWords;

/**
 * Responsável por aplicar a transformação de uma string de uma determinada
 * coluna de um dataset em uma string que não contém stop words. Aplica
 * transformação em todo dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class TransformacaoRemocaoStopWords extends TransformacaoGenerica<FunRemocaoStopWords> {

	private static final long serialVersionUID = 1L;

	/**
	 * Conjunto de stopwords.
	 */
	private Set<String> setStopWords;

	@Override
	protected String getLabeluid() {
		return "TransformacaoRemocaoStopWords";
	}

	@Override
	public TransformacaoGenerica<FunStemmer> criarTransformacao() {
		return new TransformacaoStemmer();
	}

	@Override
	public FunRemocaoStopWords criarFuncao(int indiceColEntrada, int indiceColSaida) {
		if (setStopWords == null) {
			try {
				setStopWords = new StopWords().getSetStopWords();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return new FunRemocaoStopWords(indiceColEntrada, indiceColSaida, setStopWords);
	}

}
