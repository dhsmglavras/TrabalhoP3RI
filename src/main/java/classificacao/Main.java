package classificacao;

import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import leitura.*;
import scala.Tuple2;


/**
 * Classe principal.
 * @author Ítalo Della Garza SIlva
 * @author Douglas Henrique Silva
 * @author Carlos Henrique Pereira
 */
public class Main {

	public static void main(String[] args) {
		
	    System.setProperty("spark.executor.memory", "5G");
		
		SparkSession spark = new SparkSession
				.Builder()
				.appName("TrabalhoP3")
				.master("local[*]")
				.getOrCreate();
		
		// Teste para leitura do dataset
		Dataset<Documento> dados = Leitor.lerDeDoc("data/train.tsv");
		
		
		Dataset<Documento>[] splits = dados.randomSplit(new double[]{0.9, 0.1}, 1234L);
		Dataset<Row> training = splits[0].toDF();
		Dataset<Row> test = splits[1].toDF();

		
		StringIndexer categoryIndexer = new StringIndexer()
			      .setInputCol("Sentiment")
			      .setOutputCol("label");
					
		Tokenizer tokenizer = new Tokenizer()
				  .setInputCol("Phrase")
				  .setOutputCol("words");
		
		
		HashingTF hashingTF = new HashingTF()
				  .setInputCol(tokenizer.getOutputCol())
				  .setOutputCol("features");
		
		IDF idf = new IDF()
				.setInputCol(hashingTF.getOutputCol())
				.setOutputCol("features_col");
				
		NaiveBayes nb = new NaiveBayes()
				.setSmoothing(1.0)
				.setModelType("multinomial");
		
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {categoryIndexer,tokenizer, hashingTF, idf, nb});
		
		
		MulticlassClassificationEvaluator mEvaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction");
		
		ParamMap[] paramGrid = new ParamGridBuilder()
				  .addGrid(hashingTF.numFeatures(), new int[] {1000, 10000, 100000})
				  .build();
		
		CrossValidator cv = new CrossValidator().setEstimator(pipeline)
				.setEstimatorParamMaps(paramGrid)
				.setEvaluator(mEvaluator)
				.setNumFolds(10);
		
		CrossValidatorModel cvModel = cv.fit(training);
		
		double[] avgs = cvModel.avgMetrics();
		
		for(double a:avgs)
			System.out.println(a);

	}

}
