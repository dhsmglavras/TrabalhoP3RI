package leitura;

import java.io.Serializable;

/**
 * Classe usada a partir do formato do Dataset 
 * de entrada.
 * @author Ítalo Della Garza SIlva
 * @author Douglas Henrique Silva
 * @author Carlos Henrique Pereira
 */
public final class Documento implements Serializable {
	private int PhraseId;
	private int SentenceId;
	private String Phrase;
	private int Sentiment;
	
	public int getPhraseId() {
		return PhraseId;
	}
	
	public int getSentenceId() {
		return SentenceId;
	}
	
	public String getPhrase() {
		return Phrase;
	}
	
	public int getSentiment() {
		return Sentiment;
	}
}
