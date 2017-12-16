package classificacao;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import leitura.*;

public class Main {

	public static void main(String[] args) {
		SparkSession spark = new SparkSession
				.Builder()
				.appName("TrabalhoP3")
				.master("local[*]")
				.getOrCreate();
		
		// Teste para leitura do dataset
		Dataset<Row> dados = spark.read()
				.option("header", true)
				.option("sep", "\t")
				.csv("data/train.tsv");
		dados.show();
	}

}
