package classificacao;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import leitura.*;


/**
 * Classe principal.
 * @author Ítalo Della Garza SIlva
 * @author Douglas Henrique Silva
 * @author Carlos Henrique Pereira
 */
public class Main {

	public static void main(String[] args) {
		SparkSession spark = new SparkSession
				.Builder()
				.appName("TrabalhoP3")
				.master("local[*]")
				.getOrCreate();
		
		// Teste para leitura do dataset
		Dataset<Documento> dados = Leitor.lerDeDoc("data/test.tsv");
		dados.show();
	}

}
