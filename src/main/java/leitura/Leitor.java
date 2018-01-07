package leitura;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;

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
	
	/**
	 * Aplica um determinado pré-processamento em um determinado dataset.
	 * 
	 * @param dataset
	 *            dataset a ser aplicado o pré-processamento
	 * @param preProcessamento
	 *            pré-processamento a ser aplicado no dataset
	 * @return dataset após a aplicação do pré-processamento
	 */
	public Dataset<Row> aplicarPreProcessamento(Dataset<Row> dataset, Transformer preProcessamento) {
		Transformer preProcessamentos = preProcessamento;
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {preProcessamentos});
		return pipeline.fit(dataset).transform(dataset);
	}
	
}
