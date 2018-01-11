package classificacao;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.ufla.spark.rec_inf_tp2.funcoes.FunParaTuple2;
import org.ufla.spark.rec_inf_tp2.transformacoes.TransformacaoCodificacaoASCIIDesnec;
import org.ufla.spark.rec_inf_tp2.transformacoes.TransformacaoGenerica;
import org.ufla.spark.rec_inf_tp2.transformacoes.TransformacaoMinuscula;
import org.ufla.spark.rec_inf_tp2.transformacoes.TransformacaoRemoverTags;
import org.ufla.spark.rec_inf_tp2.transformacoes.TransformacaoStemmer;
import org.ufla.spark.rec_inf_tp2.utils.DatasetUtils;

import leitura.Documento;
import leitura.Leitor;
import scala.Tuple2;

/**
 * Classe principal.
 * @author Ítalo Della Garza SIlva
 * @author Douglas Henrique Silva
 * @author Carlos Henrique Pereira
 */
public class Main {

	public static void main(String[] args) throws IOException, InstantiationException, IllegalAccessException {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		Logger.getLogger("").setLevel(Level.ERROR);
		
	    System.setProperty("spark.executor.memory", "5G");
		
		SparkSession spark = new SparkSession
				.Builder()
				.appName("TrabalhoP3")
				.master("local[*]")
				.getOrCreate();
		
		// Teste para leitura do dataset
		Dataset<Documento> dataInput = Leitor.lerDeDoc("data/train.tsv");
		Dataset<Row> dataOutput = dataInput.toDF();
		
		TransformacaoGenerica teste = TransformacaoMinuscula.class.newInstance().criarTransformacao()
				.setColunaEntrada("Phrase")
				.setColunaSaida("coluna_min")
				.setEsquemaEntrada(dataOutput.schema());
				
		dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
		teste = TransformacaoRemoverTags.class.newInstance().criarTransformacao()
				.setColunaEntrada(teste.getColunaSaida())
				.setColunaSaida("coluna_sem_tags")
				.setEsquemaEntrada(dataOutput.schema());
				
		dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
		teste = TransformacaoCodificacaoASCIIDesnec.class.newInstance().criarTransformacao()
				.setColunaEntrada(teste.getColunaSaida())
				.setColunaSaida("coluna_sem_cod_asc_desnec")
				.setEsquemaEntrada(dataOutput.schema());
				
		dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
		teste = TransformacaoCodificacaoASCIIDesnec.class.newInstance().criarTransformacao()
				.setColunaEntrada(teste.getColunaSaida())
				.setColunaSaida("coluna_sem_stop")
				.setEsquemaEntrada(dataOutput.schema());
				
		dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
		
		teste = TransformacaoStemmer.class.newInstance().criarTransformacao()
				.setColunaEntrada(teste.getColunaSaida())
				.setColunaSaida("coluna_stemmer")
				.setEsquemaEntrada(dataOutput.schema());
				
		dataOutput = Leitor.class.newInstance().aplicarPreProcessamento(dataOutput, teste);
				
		Dataset<Row>[] splits = dataOutput.randomSplit(new double[]{0.9, 0.1}, 1234L);
		Dataset<Row> training = splits[0].toDF();
		Dataset<Row> test = splits[1].toDF();
		
		StringIndexer categoryIndexer = new StringIndexer()
			      .setInputCol("Sentiment")
			      .setOutputCol("label");
					
		Tokenizer tokenizer = new Tokenizer()
				  .setInputCol(teste.getColunaSaida())
				  .setOutputCol("words");
				
		HashingTF hashingTF = new HashingTF()
				  .setInputCol(tokenizer.getOutputCol())
				  .setOutputCol("features");
				
		NaiveBayes nb = new NaiveBayes()
				.setSmoothing(1.0)
				.setModelType("multinomial");
		
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {categoryIndexer, tokenizer, hashingTF, nb});
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction");
						
		ParamMap[] paramGrid = new ParamGridBuilder()
				  .addGrid(hashingTF.numFeatures(), new int[] {11000,12200,13200,13500,13600,13700,14700,15000,18000,18300})
				  .build();
				
		String nomeDoModelo = "modeloCV_00";
		
		CrossValidatorModel cvModel;
		
		String predicoesNome = "predicoes_" + nomeDoModelo;
		File diretorioModelo = new File(nomeDoModelo);
		File diretorioPredicoes = new File(predicoesNome);
		if (diretorioPredicoes.exists() && diretorioModelo.exists()) {
			cvModel = CrossValidatorModel.read().load(nomeDoModelo);
		} else {
			if (diretorioModelo.exists()) {
				diretorioModelo.delete();
			}
			if (diretorioPredicoes.exists()) {
				diretorioPredicoes.delete();
			}
									
			CrossValidator cv = new CrossValidator().setEstimator(pipeline)
					.setEstimatorParamMaps(paramGrid)
					.setEvaluator(evaluator)
					.setNumFolds(3);
			
			cvModel = cv.fit(training);
			cvModel.write().save(nomeDoModelo);
		}
		
		Dataset<Row> predicoes;

		if (new File(predicoesNome).exists()) {
			predicoes = spark.read().load(predicoesNome);
		} else {
			predicoes = cvModel.bestModel().transform(test);
			predicoes.write().save(predicoesNome);
		}
		
		Dataset<Tuple2<Object, Object>> predicoesTuple2 = paraTuple2(predicoes, "prediction", "label");

		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predicoesTuple2.rdd());

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("\nConfusion matrix: \n" + confusion);

		// Stats by labels
		for (int i = 0; i < metrics.labels().length; i++) {
			System.out.format("\n\nClass %f precision = %f\n", metrics.labels()[i],
					metrics.precision(metrics.labels()[i]));
			System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(metrics.labels()[i]));
			System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(metrics.labels()[i]));
		}

		// Weighted stats
		System.out.println("\n\nMétricas calculadas com org.apache.spark.mllib.evaluation.MulticlassMetrics\n");
		System.out.format("Accuracy = %f\n", metrics.accuracy());
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
		System.out.format("Weighted true positive rate = %f\n", metrics.weightedTruePositiveRate());

		System.out
				.println("\n\nMétricas calculadas org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator\n");
		MulticlassClassificationEvaluator mEvaluator = new MulticlassClassificationEvaluator();
		System.out.println("F1 -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("accuracy");
		System.out.println("Accuracy -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("weightedPrecision");
		System.out.println("weightedPrecision -> " + mEvaluator.evaluate(predicoes));
		mEvaluator.setMetricName("weightedRecall");
		System.out.println("weightedRecall -> " + mEvaluator.evaluate(predicoes));

		System.out.println(
				"\nmedia das métricas F1 para cross validator em treino -> " + Arrays.toString(cvModel.avgMetrics()));	

	}
	
	/**
	 * Converte um Dataset<Row> em um Dataset<Tuple2<Object, Object>>.
	 * 
	 * @param dataset
	 *            dataset a ser convertido
	 * @param primeiroElemento
	 *            nome da coluna que será o primeiro elemento da tupla
	 * @param segundoElemento
	 *            nome da coluna que será o segundo elemento da tupla
	 * @return dataset convertido para Tuple2<Object, Object>
	 */
	private static Dataset<Tuple2<Object, Object>> paraTuple2(Dataset<Row> dataset, String primeiroElemento,
			String segundoElemento) {
		@SuppressWarnings("unchecked")
		Class<Tuple2<Object, Object>> tuple2ObjectClasse = (Class<
				Tuple2<Object, Object>>) new Tuple2<Object, Object>(primeiroElemento, segundoElemento).getClass();
		return dataset.map(new FunParaTuple2(DatasetUtils.getIndiceColuna(dataset, primeiroElemento),
				DatasetUtils.getIndiceColuna(dataset, segundoElemento)), Encoders.bean(tuple2ObjectClasse));
	}

}
