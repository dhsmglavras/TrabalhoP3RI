package leitura;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

/**
 * Classe usada para leitura do Dataset.
 * @author Ítalo Della Garza SIlva
 * @author Douglas Henrique Silva
 * @author Carlos Henrique Pereira
 */
public class Leitor {
	/**
	 * Encoder para ser usado pelo Dataset do Spark
	 */
	private static Encoder<Documento> docEncoder = Encoders.bean(Documento.class);
	/**
	 * Método para leitura na base de dados.
	 * @param nomeArquivo nome do arquivo a ser lido.
	 * @return Base de dados lida em formato Dataset do Spark
	 */
	public static Dataset<Documento> lerDeDoc(String nomeArquivo) {
		SparkSession spark = new SparkSession.Builder().getOrCreate();
		Dataset<Documento> dados = spark.read()
				.option("header", true)
				.option("sep", "\t")
				.csv(nomeArquivo)
				.as(docEncoder);
		return dados;
	}
}
